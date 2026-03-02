import logging
import numpy as np
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import constants
from models import Transformer, TransICD, HierarchicalTransICD
from data import prepare_datasets, load_embedding_weights, load_label_embedding
from trainer import train
import random
import os
import torch


def get_hyper_params_combinations(args):
    params = OrderedDict(
        learning_rate=args.learning_rate,
        num_epoch=args.num_epoch
    )
    HyperParams = namedtuple('HyperParams', params.keys())
    hyper_params_list = []
    for v in product(*params.values()):
        hyper_params_list.append(HyperParams(*v))
    return hyper_params_list


def run(args, device):
    is_hierarchical = (args.model == 'HierarchicalTransICD')

    # ----------------------------------------------------------------------- #
    # Dataset preparation                                                       #
    # ----------------------------------------------------------------------- #
    if is_hierarchical:
        train_set, dev_set, test_set, train_labels, train_label_freq, input_indexer = prepare_datasets(
            args.data_setting,
            args.batch_size,
            max_len=args.max_num_sents * args.max_sent_len,  # approx total tokens
            hierarchical=True,
            max_num_sents=args.max_num_sents,
            max_sent_len=args.max_sent_len,
        )
    else:
        train_set, dev_set, test_set, train_labels, train_label_freq, input_indexer = prepare_datasets(
            args.data_setting,
            args.batch_size,
            max_len=args.max_len,
        )

    logging.info(f'Training labels: {train_labels}\n')
    embed_weights = load_embedding_weights()
    label_desc = None

    # ----------------------------------------------------------------------- #
    # Model instantiation & training                                            #
    # ----------------------------------------------------------------------- #
    for hyper_params in get_hyper_params_combinations(args):
        if args.model == 'Transformer':
            model = Transformer(
                embed_weights, args.embed_size, args.freeze_embed, args.max_len,
                args.num_trans_layers, args.num_attn_heads, args.trans_forward_expansion,
                train_set.get_code_count(), args.dropout_rate, device
            )

        elif args.model == 'TransICD':
            model = TransICD(
                embed_weights, args.embed_size, args.freeze_embed, args.max_len,
                args.num_trans_layers, args.num_attn_heads, args.trans_forward_expansion,
                train_set.get_code_count(), args.label_attn_expansion,
                args.dropout_rate, label_desc, device, train_label_freq
            )

        elif args.model == 'HierarchicalTransICD':
            model = HierarchicalTransICD(
                embed_weights=embed_weights,
                embed_size=args.embed_size,
                freeze_embed=args.freeze_embed,
                max_sent_len=args.max_sent_len,
                max_num_sents=args.max_num_sents,
                word_num_layers=args.word_num_layers,
                word_num_heads=args.word_num_heads,
                sent_num_layers=args.sent_num_layers,
                sent_num_heads=args.sent_num_heads,
                forward_expansion=args.trans_forward_expansion,
                output_size=train_set.get_code_count(),
                attn_expansion=args.label_attn_expansion,
                dropout_rate=args.dropout_rate,
                device=device,
                label_freq=train_label_freq,
                pad_idx=input_indexer.index_of(constants.PAD_SYMBOL),
                eos_idx=input_indexer.index_of(constants.EOS_SYMBOL),
            )

        else:
            raise ValueError(
                f"Unknown --model value '{args.model}'. "
                f"Choose from: Transformer, TransICD, HierarchicalTransICD"
            )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Model: {args.model} | Trainable parameters: {num_params:,}')
        logging.info(f'Training configuration: {hyper_params}')
        model.to(device)
        train(model, train_set, dev_set, test_set, hyper_params, args.batch_size, device)


if __name__ == "__main__":
    args = constants.get_args()
    if not os.path.exists('../results'):
        os.makedirs('../results')
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(
        filename='../results/app.log', filemode='w',
        format=FORMAT, level=getattr(logging, args.log.upper())
    )
    logging.info(f'{args}\n')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f'Using device: {device}')
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)
    run(args, device)
