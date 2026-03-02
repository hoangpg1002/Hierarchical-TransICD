import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# =============================================================================
# Shared modules
# =============================================================================

class Attention(nn.Module):
    """
    Code-wise Attention module.

    Given a sequence of hidden states H (B x S x H), computes a separate
    attention distribution for each ICD code, producing code-specific
    document representations.

    Attention score: e_i = W2 * tanh(W1 * h_i)
    Attention weight: alpha_i = softmax(e_i)  [over position i]
    Code representation: c_k = sum_i alpha_{k,i} * h_i

    Args:
        hidden_size   : Dimensionality of input hidden states
        output_size   : Number of ICD codes (one attention head per code)
        attn_expansion: Expansion factor for the intermediate linear layer
        dropout_rate  : Dropout rate
    """
    def __init__(self, hidden_size, output_size, attn_expansion, dropout_rate):
        super(Attention, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size * attn_expansion)
        self.tnh = nn.Tanh()
        self.l2 = nn.Linear(hidden_size * attn_expansion, output_size)

    def forward(self, hidden, attn_mask=None):
        # hidden: B x S x H
        # output_1: B x S x (attn_expansion * H)
        output_1 = self.tnh(self.l1(hidden))

        # output_2: B x S x output_size(O)
        output_2 = self.l2(output_1)

        # Mask padded positions before softmax
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x O x S  (one distribution per code)
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x O x S) @ (B x S x H) -> B x O x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class LabelAttention(nn.Module):
    """
    Label-description guided attention.

    Uses pretrained embeddings of ICD code descriptions as query vectors
    to attend over the document's token-level hidden states.
    """
    def __init__(self, hidden_size, label_embed_size, dropout_rate):
        super(LabelAttention, self).__init__()
        self.l1 = nn.Linear(hidden_size, label_embed_size)
        self.tnh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, label_embeds, attn_mask=None):
        # hidden: B x S x H
        # output_1: B x S x E
        output_1 = self.tnh(self.l1(hidden))
        output_1 = self.dropout(output_1)

        # output_2: (B x S x E) @ (E x L) -> B x S x L
        output_2 = torch.matmul(output_1, label_embeds.t())

        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x L x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x L x S) @ (B x S x H) -> B x L x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in:
    Vaswani et al., "Attention is All You Need", NeurIPS 2017.

    Adds a fixed positional signal to the token embeddings.
    """
    def __init__(self, d_model, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


# =============================================================================
# Baseline: Vanilla Transformer classifier (mean pooling, no attention)
# =============================================================================

class Transformer(nn.Module):
    """
    Baseline multi-label ICD classifier using a Transformer encoder
    with mean pooling and per-code linear heads.
    """
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len,
                 num_layers, num_heads, forward_expansion, output_size,
                 dropout_rate, device, pad_idx=0):
        super(Transformer, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=forward_expansion * embed_size, dropout=dropout_rate
        )
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(output_size)])

    def forward(self, inputs, targets=None):
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # (B, S)

        # (B, S, E) → scale → dropout → permute to (S, B, E) for pos encoder
        embeds = self.embedder(inputs) * math.sqrt(self.embed_size)    # (B, S, E)
        embeds = self.dropout(embeds)
        embeds = self.pos_encoder(embeds.permute(1, 0, 2))              # (S, B, E)

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        encoded_inputs = encoded_inputs.permute(1, 0, 2)               # (B, S, E)

        pooled_outputs = encoded_inputs.mean(dim=1)                    # (B, E)
        outputs = torch.zeros((pooled_outputs.size(0), self.output_size), device=self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code + 1] = fc(pooled_outputs)

        return outputs, None, None


# =============================================================================
# TransICD: Token-level Transformer + Code-wise Attention + LDAM loss
# =============================================================================

class TransICD(nn.Module):
    """
    TransICD model (Biswas et al., 2021).

    Encodes a discharge note with a Transformer encoder, then applies
    code-wise label attention to generate per-code representations.
    LDAM (Label-Distribution-Aware Margin) loss is supported for
    handling class imbalance common in clinical coding tasks.

    Reference:
        Biplob Biswas, Hoang Pham, Ping Zhang.
        "TransICD: Transformer Based Code-wise Attention Model for
        Explainable ICD Coding." AIME 2021.
    """
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len,
                 num_layers, num_heads, forward_expansion, output_size,
                 attn_expansion, dropout_rate, label_desc, device,
                 label_freq=None, C=3.0, pad_idx=0):
        super(TransICD, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        # LDAM margin buffer
        if label_freq is not None:
            class_margin = torch.tensor(label_freq, dtype=torch.float32) ** 0.25
            class_margin = class_margin.masked_fill(class_margin == 0, 1)
            self.register_buffer('class_margin', 1.0 / class_margin)
            self.C = C
        else:
            self.class_margin = None
            self.C = 0

        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=forward_expansion * embed_size, dropout=dropout_rate
        )
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        self.attn = Attention(embed_size, output_size, attn_expansion, dropout_rate)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(output_size)])

    def forward(self, inputs, targets=None):
        # attn_mask: (B, S, 1) — 0 for PAD positions
        attn_mask = (inputs != self.pad_idx).unsqueeze(2).to(self.device)
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # (B, S)

        embeds = self.embedder(inputs) * math.sqrt(self.embed_size)    # (B, S, E)
        embeds = self.dropout(embeds)
        embeds = self.pos_encoder(embeds.permute(1, 0, 2))             # (S, B, E)

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        encoded_inputs = encoded_inputs.permute(1, 0, 2)               # (B, S, E)

        weighted_outputs, attn_weights = self.attn(encoded_inputs, attn_mask)

        outputs = torch.zeros((weighted_outputs.size(0), self.output_size), device=self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code + 1] = fc(weighted_outputs[:, code, :])

        if targets is not None and self.class_margin is not None and self.C > 0:
            ldam_outputs = outputs - targets * self.class_margin * self.C
        else:
            ldam_outputs = None

        return outputs, ldam_outputs, attn_weights


# =============================================================================
# HierarchicalTransICD: Token -> Sentence -> Document
# =============================================================================

class HierarchicalTransICD(nn.Module):
    """
    Hierarchical Transformer for automatic ICD coding from long clinical notes.

    Motivation:
        The vanilla TransICD model applies self-attention across all tokens of a
        document simultaneously (complexity: O(L^2), where L is token length).
        For long discharge summaries (L > 2500), this is computationally
        prohibitive. Furthermore, token-level attention weights are difficult
        to interpret clinically; clinicians reason at the sentence level.

    Architecture (Token -> Sentence -> Document):
        ┌──────────────────────────────────────────────┐
        │  Input: (B, N_s, T_w)                        │
        │  B = batch size                              │
        │  N_s = number of sentences per document      │
        │  T_w = max tokens per sentence               │
        └──────────┬───────────────────────────────────┘
                   │  Flatten to (B*N_s, T_w)
                   ▼
        ┌──────────────────────────────────────────────┐
        │  Level 1: Word-level Transformer Encoder     │
        │  - Embedding (E) + Positional Encoding       │
        │  - Transformer Encoder layers (word-level)   │
        │  - Mean-pool over non-PAD tokens             │
        │  Output: sentence vector (B*N_s, E)          │
        │  Reshaped to (B, N_s, E)                     │
        └──────────┬───────────────────────────────────┘
                   ▼
        ┌──────────────────────────────────────────────┐
        │  Level 2: Sentence-level Transformer Encoder │
        │  - Positional Encoding (sentence index)      │
        │  - Transformer Encoder layers (sent-level)   │
        │  Output: contextualized sentence vectors     │
        │          (B, N_s, E)                         │
        └──────────┬───────────────────────────────────┘
                   ▼
        ┌──────────────────────────────────────────────┐
        │  Level 3: Code-wise Attention (sentence-lvl) │
        │  - Per-code attention weights over sentences │
        │  - attn_weights: (B, num_codes, N_s)         │
        │     → identifies evidence sentences per code │
        └──────────┬───────────────────────────────────┘
                   ▼
        ┌──────────────────────────────────────────────┐
        │  Level 4: Per-code Linear Classifiers        │
        │  Output: logits (B, num_codes)               │
        │  LDAM margin adjustment (train-time)         │
        └──────────────────────────────────────────────┘

    Memory complexity:
        Word-level self-attention: O(B * N_s * T_w^2)
        Sent-level self-attention: O(B * N_s^2)
        Total: O(B * (N_s * T_w^2 + N_s^2))  << O(B * (N_s * T_w)^2)
        Example: N_s=150, T_w=30 -> 150*900 + 150^2 = 157,500
                 vs. flat 4500^2 = 20,250,000
    """

    def __init__(
        self,
        embed_weights,         # FloatTensor: (vocab_size, embed_size)
        embed_size,            # Token embedding dimensionality (d_model)
        freeze_embed,          # If True, pretrained embeddings are not fine-tuned
        max_sent_len,          # T_w: max tokens per sentence (word encoder horizon)
        max_num_sents,         # N_s: max sentences per document (sent encoder horizon)
        word_num_layers,       # Depth of the word-level Transformer Encoder
        word_num_heads,        # Multi-head attention heads for word encoder
        sent_num_layers,       # Depth of the sentence-level Transformer Encoder
        sent_num_heads,        # Multi-head attention heads for sentence encoder
        forward_expansion,     # FFN width multiplier: d_ff = forward_expansion * d_model
        output_size,           # Number of ICD code classification heads
        attn_expansion,        # Width multiplier in code-wise attention MLP
        dropout_rate,          # Applied consistently across embedding, encoder, attention
        device,
        label_freq=None,       # List of per-code training frequencies for LDAM
        C=3.0,                 # LDAM margin scaling constant (Cao et al., 2019)
        pad_idx=0,             # Vocabulary index of <PAD>
        eos_idx=2,             # Vocabulary index of <EOS> (sentence boundary marker)
    ):
        super(HierarchicalTransICD, self).__init__()

        if embed_size % word_num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be divisible by word_num_heads ({word_num_heads})"
            )
        if embed_size % sent_num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be divisible by sent_num_heads ({sent_num_heads})"
            )

        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.output_size = output_size
        self.max_sent_len = max_sent_len
        self.max_num_sents = max_num_sents

        # ------------------------------------------------------------------ #
        # LDAM: Label-Distribution-Aware Margin (Cao et al., NeurIPS 2019)   #
        # Computes per-class margin: Delta_j = C / (n_j ^ 0.25)             #
        # where n_j is the training frequency of class j.                    #
        # ------------------------------------------------------------------ #
        if label_freq is not None:
            class_margin = torch.tensor(label_freq, dtype=torch.float32) ** 0.25
            class_margin = class_margin.masked_fill(class_margin == 0, 1.0)
            self.register_buffer('class_margin', 1.0 / class_margin)
            self.C = C
        else:
            self.class_margin = None
            self.C = 0.0

        # ------------------------------------------------------------------ #
        # Token Embeddings (shared across both levels)                       #
        # ------------------------------------------------------------------ #
        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)
        self.embed_dropout = nn.Dropout(dropout_rate)

        # ------------------------------------------------------------------ #
        # Level 1 — Word-level Transformer Encoder                           #
        # Processes each sentence independently, bounded by max_sent_len.    #
        # ------------------------------------------------------------------ #
        self.word_pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_sent_len)
        word_enc_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=word_num_heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout_rate,
            batch_first=False,
        )
        self.word_encoder = TransformerEncoder(word_enc_layer, num_layers=word_num_layers)

        # ------------------------------------------------------------------ #
        # Level 2 — Sentence-level Transformer Encoder                       #
        # Captures inter-sentence dependencies across the document.          #
        # ------------------------------------------------------------------ #
        self.sent_pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_num_sents)
        sent_enc_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=sent_num_heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout_rate,
            batch_first=False,
        )
        self.sent_encoder = TransformerEncoder(sent_enc_layer, num_layers=sent_num_layers)

        # ------------------------------------------------------------------ #
        # Level 3 — Code-wise Attention over sentence representations        #
        # Each ICD code independently attends over all sentence vectors,     #
        # producing an interpretable distribution: which sentences matter    #
        # most for predicting each specific diagnosis code.                  #
        # ------------------------------------------------------------------ #
        self.code_attn = Attention(embed_size, output_size, attn_expansion, dropout_rate)

        # ------------------------------------------------------------------ #
        # Level 4 — Per-code binary classifiers                              #
        # ------------------------------------------------------------------ #
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(output_size)])

    def forward(self, inputs, targets=None):
        """
        Forward pass through the Hierarchical TransICD model.

        Args:
            inputs  (LongTensor): shape (B, N_s, T_w)
                Tokenized and padded input, pre-split into sentence chunks.
                Sentences are delimited (in preprocessing) by <EOS> tokens;
                here the structure is passed as a 3-D index tensor.
            targets (FloatTensor): shape (B, num_codes), optional.
                Multi-hot label vector. Required for LDAM margin during training.

        Returns:
            outputs (FloatTensor): (B, num_codes) — raw logits before sigmoid.
            ldam_outputs (FloatTensor | None): (B, num_codes) — margin-adjusted
                logits for LDAM loss. None during evaluation.
            sent_attn_weights (FloatTensor): (B, num_codes, N_s) — interpretable
                sentence-level attention weights per ICD code.
                High weight -> sentence strongly supports that code prediction.
        """
        batch_size, num_sents, sent_len = inputs.size()

        # ================================================================== #
        # LEVEL 1: Word-level encoding                                        #
        # ================================================================== #

        # Flatten: treat all sentences as independent sequences in one batch.
        # flat_inputs: (B * N_s, T_w)
        flat_inputs = inputs.view(batch_size * num_sents, sent_len)

        # Word-level key padding mask (True = PAD position, to be ignored)
        # flat_pad_mask: (B * N_s, T_w)
        flat_pad_mask = (flat_inputs == self.pad_idx).to(self.device)

        # Identify fully-padded sentence slots (all tokens are PAD).
        # PyTorch TransformerEncoder produces NaN when all keys are masked.
        # We handle this by temporarily un-masking one token position for
        # all-PAD sentences, then zero-ing out those sentence vectors after pooling.
        # fully_pad_sent: (B * N_s,) — True if the entire sentence is PAD
        fully_pad_sent = flat_pad_mask.all(dim=1)  # (B*N_s,)

        # Safe mask: un-mask position 0 for all-PAD sentences to prevent NaN
        safe_pad_mask = flat_pad_mask.clone()
        safe_pad_mask[fully_pad_sent, 0] = False   # keep at least one key unmasked

        # Token embedding with embedding scale (Vaswani et al., 2017)
        # flat_embeds: (B*N_s, T_w, E)
        flat_embeds = self.embedder(flat_inputs.to(self.device)) * math.sqrt(self.embed_size)
        flat_embeds = self.embed_dropout(flat_embeds)

        # PositionalEncoding expects (T_w, B*N_s, E)
        flat_embeds = flat_embeds.permute(1, 0, 2)          # (T_w, B*N_s, E)
        flat_embeds = self.word_pos_encoder(flat_embeds)

        # Word-level self-attention: complexity O(B * N_s * T_w^2)
        word_encoded = self.word_encoder(
            flat_embeds, src_key_padding_mask=safe_pad_mask
        )                                                   # (T_w, B*N_s, E)
        word_encoded = word_encoded.permute(1, 0, 2)        # (B*N_s, T_w, E)

        # ------------------------------------------------------------------
        # Aggregate token states to a sentence vector via masked mean pooling.
        # non_pad_mask: (B*N_s, T_w, 1) — 1.0 for real tokens
        # ------------------------------------------------------------------
        non_pad_mask = (~flat_pad_mask).unsqueeze(-1).float()
        # Masked mean: sum(h_i * m_i) / sum(m_i)
        sent_vecs = (word_encoded * non_pad_mask).sum(dim=1) / (non_pad_mask.sum(dim=1) + 1e-9)

        # Zero out sentence vectors for fully-padded sentence slots
        # (their mean-pooled value is undefined / noise from 1 unmasked PAD token)
        sent_vecs[fully_pad_sent] = 0.0

        # sent_vecs: (B*N_s, E) -> (B, N_s, E)
        sent_vecs = sent_vecs.view(batch_size, num_sents, self.embed_size)

        # ================================================================== #
        # LEVEL 2: Sentence-level encoding                                    #
        # ================================================================== #

        # Construct sentence-level padding mask.
        # A sentence slot is padding if ALL its token positions are PAD
        # (occurs when a document contains fewer than max_num_sents sentences).
        # flat_pad_mask: (B * N_s, T_w) -> (B, N_s, T_w)
        sent_is_pad = flat_pad_mask.view(batch_size, num_sents, sent_len).all(dim=2)  # (B, N_s)

        # Sentence-level positional encoding: (N_s, B, E)
        sent_vecs = sent_vecs.permute(1, 0, 2)             # (N_s, B, E)
        sent_vecs = self.sent_pos_encoder(sent_vecs)

        # Sentence-level self-attention: complexity O(B * N_s^2)
        sent_encoded = self.sent_encoder(
            sent_vecs, src_key_padding_mask=sent_is_pad
        )                                                   # (N_s, B, E)
        sent_encoded = sent_encoded.permute(1, 0, 2)       # (B, N_s, E)

        # ================================================================== #
        # LEVEL 3: Code-wise Attention (sentence-level)                       #
        # ================================================================== #

        # sent_attn_mask: (B, N_s, 1) — 0.0 for fully-padded sentence slots
        sent_attn_mask = (~sent_is_pad).unsqueeze(2).to(self.device).float()

        # weighted_outputs: (B, num_codes, E)
        # sent_attn_weights: (B, num_codes, N_s)  ← the interpretability output
        weighted_outputs, sent_attn_weights = self.code_attn(sent_encoded, sent_attn_mask)

        # ================================================================== #
        # LEVEL 4: Per-code linear classification heads                       #
        # ================================================================== #
        outputs = torch.zeros((batch_size, self.output_size), device=self.device)
        for code_idx, fc in enumerate(self.fcs):
            # weighted_outputs[:, code_idx, :]: (B, E) -> fc -> (B, 1)
            outputs[:, code_idx:code_idx + 1] = fc(weighted_outputs[:, code_idx, :])

        # ================================================================== #
        # LDAM margin adjustment (applied only during training)               #
        # Effective loss: BCE(logit - y * C / n_j^0.25, label)               #
        # ================================================================== #
        if targets is not None and self.class_margin is not None and self.C > 0:
            ldam_outputs = outputs - targets * self.class_margin * self.C
        else:
            ldam_outputs = None

        return outputs, ldam_outputs, sent_attn_weights
