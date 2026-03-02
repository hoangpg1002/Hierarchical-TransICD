import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from utils import *
from constants import *


def remove_stopwords(text):
    stpwords = set([stopword for stopword in stopwords.words('english')])
    stpwords.update({'admission', 'birth', 'date', 'discharge', 'service', 'sex'})
    tokens = text.strip().split()
    tokens = [token for token in tokens if token not in stpwords]
    return ' '.join(tokens)


def load_dataset(data_setting, batch_size, split):
    data = pd.read_csv(f'{GENERATED_DIR}/{split}_{data_setting}.csv', dtype={'LENGTH': int})
    len_stat = data['LENGTH'].describe()
    logging.info(f'{split} set length stats:\n{len_stat}')

    if data_setting == FULL:
        code_df = pd.read_csv(f'{CODE_FREQ_PATH}', dtype={'code': str})
        all_codes = ';'.join(map(str, code_df['code'].values.tolist()))
        data = pd.concat(
            [data, pd.DataFrame([{'HADM_ID': -1, 'TEXT': 'remove', 'LABELS': all_codes, 'LENGTH': 6}])],
            ignore_index=True
        )

    mlb = MultiLabelBinarizer()
    data['LABELS'] = data['LABELS'].apply(lambda x: str(x).split(';'))
    code_counts = list(data['LABELS'].str.len())
    avg_code_counts = sum(code_counts) / len(code_counts)
    logging.info(f'In {split} set, average code counts per discharge summary: {avg_code_counts}')
    mlb.fit(data['LABELS'])
    temp = mlb.transform(data['LABELS'])
    if mlb.classes_[-1] == 'nan':
        mlb.classes_ = mlb.classes_[:-1]
    logging.info(f'Final number of labels/codes: {len(mlb.classes_)}')

    for i, x in enumerate(mlb.classes_):
        data[x] = temp[:, i]
    data.drop(['LABELS', 'LENGTH'], axis=1, inplace=True)

    if data_setting == FULL:
        data = data[:-1]

    code_list = list(mlb.classes_)
    label_freq = list(data[code_list].sum(axis=0))
    hadm_ids = data['HADM_ID'].values.tolist()
    texts = data['TEXT'].values.tolist()
    labels = data[code_list].values.tolist()
    item_count = (len(texts) // batch_size) * batch_size
    logging.info(f'{split} set true item count: {item_count}\n\n')
    return {'hadm_ids': hadm_ids[:item_count],
            'texts': texts[:item_count],
            'targets': labels[:item_count],
            'labels': code_list,
            'label_freq': label_freq}


def get_all_codes(train_path, dev_path, test_path):
    all_codes = set()
    splits_path = {'train': train_path, 'dev': dev_path, 'test': test_path}
    for split, file_path in splits_path.items():
        split_df = pd.read_csv(file_path, dtype={'HADM_ID': str})
        split_codes = set()
        for codes in split_df['LABELS'].values:
            for code in str(codes).split(';'):
                split_codes.add(code)
        logging.info(f'{split} set has {len(split_codes)} unique codes')
        all_codes.update(split_codes)
    logging.info(f'In total, there are {len(all_codes)} unique codes')
    return list(all_codes)


def load_datasets(data_setting, batch_size):
    train_raw = load_dataset(data_setting, batch_size, split='train')
    dev_raw = load_dataset(data_setting, batch_size, split='dev')
    test_raw = load_dataset(data_setting, batch_size, split='test')

    if train_raw['labels'] != dev_raw['labels'] or dev_raw['labels'] != test_raw['labels']:
        raise ValueError("Train dev test labels don't match!")

    return train_raw, dev_raw, test_raw


def load_embedding_weights():
    W = []
    with open(EMBED_FILE_PATH) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float64)
            W.append(vec)
    logging.info(f'Total token count (including PAD, UNK, EOS) of full preprocessed discharge summaries: {len(W)}')
    weights = torch.tensor(W, dtype=torch.float)
    return weights


def load_label_embedding(labels, pad_index):
    code_desc = []
    desc_dt = {}
    max_desc_len = 0
    with open(f'{CODE_DESC_VECTOR_PATH}', 'r') as fin:
        for line in fin:
            items = line.strip().split()
            code = items[0]
            if code in labels:
                desc_dt[code] = list(map(int, items[1:]))
                max_desc_len = max(max_desc_len, len(desc_dt[code]))
    for code in labels:
        pad_len = max_desc_len - len(desc_dt[code])
        code_desc.append(desc_dt[code] + [pad_index] * pad_len)
    code_desc = torch.tensor(code_desc, dtype=torch.long)
    return code_desc


# =============================================================================
# Standard (flat) token indexing — used by Transformer and TransICD
# =============================================================================

def index_text(data, indexer, max_len, split):
    """
    Converts a list of text strings into a flat 2D token index matrix.
    Output shape: (num_samples, max_len).
    """
    data_indexed = []
    lens = []
    oov_word_frac = []
    for text in data:
        num_oov_words = 0
        text_indexed = [indexer.index_of(PAD_SYMBOL)] * max_len
        tokens = str(text).split()
        text_len = max_len if len(tokens) > max_len else len(tokens)
        lens.append(text_len)
        for i in range(text_len):
            if indexer.index_of(tokens[i]) >= 0:
                text_indexed[i] = indexer.index_of(tokens[i])
            else:
                num_oov_words += 1
                text_indexed[i] = indexer.index_of(UNK_SYMBOL)
        oov_word_frac.append(num_oov_words / text_len)
        data_indexed.append(text_indexed)
    logging.info(
        f'{split} dataset has on average {sum(oov_word_frac)/len(oov_word_frac):.4f} '
        f'oov word fraction per discharge summary'
    )
    return data_indexed, lens


# =============================================================================
# Hierarchical token indexing — used by HierarchicalTransICD
# =============================================================================

def index_text_hierarchical(data, indexer, max_num_sents, max_sent_len, split):
    """
    Converts a list of text strings (with <EOS> sentence delimiters) into a
    3D token index tensor of shape (num_samples, max_num_sents, max_sent_len).

    The text is expected to have been preprocessed so that each sentence ends
    with the special token '<EOS>', inserted by preprocessor.py.

    Strategy:
        1. Split the flat token list at every '<EOS>' occurrence.
        2. Truncate to max_num_sents sentences.
        3. Pad each sentence to max_sent_len.
        4. Pad the sentence list to max_num_sents.

    This results in a 3D integer tensor that can be passed directly to the
    HierarchicalTransICD forward pass as `inputs`.

    Args:
        data          : list of str — preprocessed discharge note texts.
        indexer       : Indexer — maps tokens to integer indices.
        max_num_sents : int — maximum number of sentences per document.
        max_sent_len  : int — maximum number of tokens per sentence.
        split         : str — dataset split name (for logging).

    Returns:
        data_hierarchical : list of 2D lists of shape (max_num_sents, max_sent_len).
        sent_counts       : list of int — actual sentence count per document.
    """
    eos_symbol = EOS_SYMBOL
    pad_idx = indexer.index_of(PAD_SYMBOL)
    unk_idx = indexer.index_of(UNK_SYMBOL)

    data_hierarchical = []
    sent_counts = []
    oov_fracs = []

    for text in data:
        tokens = str(text).split()
        num_oov = 0
        total_real_tokens = 0

        # Split flat token list into sentences at <EOS> boundaries
        sentences = []
        current_sentence = []
        for tok in tokens:
            if tok == eos_symbol:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(tok)
        # Handle last sentence that may not be followed by <EOS>
        if current_sentence:
            sentences.append(current_sentence)

        # Truncate to max_num_sents
        sentences = sentences[:max_num_sents]
        sent_counts.append(len(sentences))

        # Build 2D padded matrix: (max_num_sents, max_sent_len)
        doc_matrix = []
        for sent_tokens in sentences:
            row = [pad_idx] * max_sent_len
            sent_len = min(len(sent_tokens), max_sent_len)
            total_real_tokens += sent_len
            for i in range(sent_len):
                tok = sent_tokens[i]
                idx = indexer.index_of(tok)
                if idx >= 0:
                    row[i] = idx
                else:
                    num_oov += 1
                    row[i] = unk_idx
            doc_matrix.append(row)

        # Pad with entirely-PAD sentences up to max_num_sents
        while len(doc_matrix) < max_num_sents:
            doc_matrix.append([pad_idx] * max_sent_len)

        data_hierarchical.append(doc_matrix)
        if total_real_tokens > 0:
            oov_fracs.append(num_oov / total_real_tokens)

    avg_oov = sum(oov_fracs) / len(oov_fracs) if oov_fracs else 0.0
    avg_sents = sum(sent_counts) / len(sent_counts)
    logging.info(
        f'{split} hierarchical dataset | avg sentences/doc: {avg_sents:.1f} '
        f'(max: {max_num_sents}) | avg OOV fraction: {avg_oov:.4f}'
    )
    return data_hierarchical, sent_counts


# =============================================================================
# Dataset classes
# =============================================================================

class ICD_Dataset(Dataset):
    """
    Standard (flat) dataset for Transformer and TransICD models.
    Each sample has a 1D token index tensor of shape (max_len,).
    """
    def __init__(self, hadm_ids, texts, lens, labels):
        self.hadm_ids = hadm_ids
        self.texts = texts
        self.lens = lens
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def get_code_count(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        hadm_id = torch.tensor(self.hadm_ids[index])
        text = torch.tensor(self.texts[index], dtype=torch.long)
        length = torch.tensor(self.lens[index], dtype=torch.long)
        codes = torch.tensor(self.labels[index], dtype=torch.float)
        return {'hadm_id': hadm_id, 'text': text, 'length': length, 'codes': codes}


class HierarchicalICD_Dataset(Dataset):
    """
    Hierarchical dataset for HierarchicalTransICD.
    Each sample has a 2D token index tensor of shape (max_num_sents, max_sent_len).
    This 3D batch tensor (B, N_s, T_w) is what the HierarchicalTransICD model expects.
    """
    def __init__(self, hadm_ids, texts_2d, sent_counts, labels):
        """
        Args:
            hadm_ids    : list of int — hospital admission IDs.
            texts_2d    : list of 2D lists (max_num_sents, max_sent_len) of int indices.
            sent_counts : list of int — actual (non-padded) sentence count per document.
            labels      : list of lists — multi-hot label vectors.
        """
        self.hadm_ids = hadm_ids
        self.texts_2d = texts_2d
        self.sent_counts = sent_counts
        self.labels = labels

    def __len__(self):
        return len(self.texts_2d)

    def get_code_count(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        hadm_id = torch.tensor(self.hadm_ids[index])
        # text: (max_num_sents, max_sent_len)
        text = torch.tensor(self.texts_2d[index], dtype=torch.long)
        num_sents = torch.tensor(self.sent_counts[index], dtype=torch.long)
        codes = torch.tensor(self.labels[index], dtype=torch.float)
        return {'hadm_id': hadm_id, 'text': text, 'num_sents': num_sents, 'codes': codes}


# =============================================================================
# Unified dataset preparation
# =============================================================================

def prepare_datasets(data_setting, batch_size, max_len,
                     hierarchical=False, max_num_sents=None, max_sent_len=None):
    """
    Prepares train/dev/test datasets for either flat or hierarchical model.

    Args:
        data_setting  : str — '50' or 'full'.
        batch_size    : int.
        max_len       : int — used for flat models (TransICD, Transformer).
        hierarchical  : bool — if True, builds HierarchicalICD_Dataset.
        max_num_sents : int — max sentences per doc (hierarchical mode only).
        max_sent_len  : int — max tokens per sentence (hierarchical mode only).

    Returns:
        train_set, dev_set, test_set, labels, label_freq, input_indexer
    """
    train_data, dev_data, test_data = load_datasets(data_setting, batch_size)

    # Build vocabulary indexer
    input_indexer = Indexer()
    input_indexer.add_and_get_index(PAD_SYMBOL)
    input_indexer.add_and_get_index(UNK_SYMBOL)
    input_indexer.add_and_get_index(EOS_SYMBOL)   # <EOS> must be at index 2
    with open(VOCAB_FILE_PATH, 'r') as fin:
        for line in fin:
            word = line.strip()
            if word in (PAD_SYMBOL, UNK_SYMBOL, EOS_SYMBOL):
                continue
            input_indexer.add_and_get_index(word)

    logging.info(f'Vocabulary size (PAD, UNK, EOS + words): {len(input_indexer)}')

    if hierarchical:
        if max_num_sents is None or max_sent_len is None:
            raise ValueError("Hierarchical mode requires max_num_sents and max_sent_len.")

        train_h, train_sc = index_text_hierarchical(
            train_data['texts'], input_indexer, max_num_sents, max_sent_len, 'train'
        )
        dev_h, dev_sc = index_text_hierarchical(
            dev_data['texts'], input_indexer, max_num_sents, max_sent_len, 'dev'
        )
        test_h, test_sc = index_text_hierarchical(
            test_data['texts'], input_indexer, max_num_sents, max_sent_len, 'test'
        )

        train_set = HierarchicalICD_Dataset(train_data['hadm_ids'], train_h, train_sc, train_data['targets'])
        dev_set   = HierarchicalICD_Dataset(dev_data['hadm_ids'],   dev_h,   dev_sc,   dev_data['targets'])
        test_set  = HierarchicalICD_Dataset(test_data['hadm_ids'],  test_h,  test_sc,  test_data['targets'])
    else:
        train_text_indexed, train_lens = index_text(train_data['texts'], input_indexer, max_len, 'train')
        dev_text_indexed,   dev_lens   = index_text(dev_data['texts'],   input_indexer, max_len, 'dev')
        test_text_indexed,  test_lens  = index_text(test_data['texts'],  input_indexer, max_len, 'test')

        train_set = ICD_Dataset(train_data['hadm_ids'], train_text_indexed, train_lens, train_data['targets'])
        dev_set   = ICD_Dataset(dev_data['hadm_ids'],   dev_text_indexed,   dev_lens,   dev_data['targets'])
        test_set  = ICD_Dataset(test_data['hadm_ids'],  test_text_indexed,  test_lens,  test_data['targets'])

    return train_set, dev_set, test_set, train_data['labels'], train_data['label_freq'], input_indexer
