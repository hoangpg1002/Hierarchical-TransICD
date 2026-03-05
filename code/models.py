import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Attention(nn.Module):
    def __init__(self, hidden_size, output_size, attn_expansion, dropout_rate):
        super(Attention, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size*attn_expansion)
        self.tnh = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size*attn_expansion, output_size)

    def forward(self, hidden, attn_mask=None):
        # output_1: B x S x H -> B x S x attn_expansion*H
        output_1 = self.tnh(self.l1(hidden))
        # output_1 = self.dropout(output_1)

        # output_2: B x S x attn_expansion*H -> B x S x output_size(O)
        output_2 = self.l2(output_1)

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x output_size(O) -> B x O x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x O x S) @ (B x S x H) -> B x O x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class LabelAttention(nn.Module):
    def __init__(self, hidden_size, label_embed_size, dropout_rate):
        super(LabelAttention, self).__init__()
        self.l1 = nn.Linear(hidden_size, label_embed_size)
        self.tnh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, label_embeds, attn_mask=None):
        # output_1: B x S x H -> B x S x E
        output_1 = self.tnh(self.l1(hidden))
        output_1 = self.dropout(output_1)

        # output_2: (B x S x E) x (E x L) -> B x S x L
        output_2 = torch.matmul(output_1, label_embeds.t())

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x L -> B x L x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x L x S) @ (B x S x H) -> B x L x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class CodeWiseAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CodeWiseAttention, self).__init__()
        self.query = nn.Parameter(torch.randn(output_size, hidden_size))
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, attn_mask=None):
        # hidden: B x S x H
        projected_hidden = self.key_proj(hidden)
        # scores: B x O x S
        scores = torch.einsum('oh,bsh->bos', self.query, projected_hidden)

        if attn_mask is not None:
            # attn_mask: B x S -> B x 1 x S
            expanded_mask = attn_mask.unsqueeze(1).float()
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        else:
            expanded_mask = None

        attn_weights = F.softmax(scores, dim=-1)
        if expanded_mask is not None:
            attn_weights = attn_weights * expanded_mask
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # weighted_output: B x O x H
        weighted_output = torch.bmm(attn_weights, hidden)
        return weighted_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len, num_layers, num_heads, forward_expansion,
                 output_size, dropout_rate, device, pad_idx=0):
        super(Transformer, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*embed_size, dropout=dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for code in range(output_size)])

    def forward(self, inputs, targets=None):
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S

        embeds = self.pos_encoder(self.embedder(inputs) * math.sqrt(self.embed_size))  # N x S x E
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # S x N x E

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # T x N x E
        encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x T x E

        pooled_outputs = encoded_inputs.mean(dim=1)
        outputs = torch.zeros((pooled_outputs.size(0), self.output_size)).to(self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code+1] = fc(pooled_outputs)

        return outputs, None, None


class TransICD(nn.Module):
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len, num_layers, num_heads, forward_expansion,
                 output_size, attn_expansion, dropout_rate, label_desc, device, label_freq=None, C=3.0,  pad_idx=0):
        super(TransICD, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size
        # self.register_buffer('label_desc', label_desc)
        # self.register_buffer('label_desc_mask', (self.label_desc != self.pad_idx)*1.0)

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
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*embed_size, dropout=dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        self.attn = Attention(embed_size, output_size, attn_expansion, dropout_rate)
        # self.label_attn = LabelAttention(embed_size, embed_size, dropout_rate)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for code in range(output_size)])

    def embed_label_desc(self):
        label_embeds = self.embedder(self.label_desc).transpose(1, 2).matmul(self.label_desc_mask.unsqueeze(2))
        label_embeds = torch.div(label_embeds.squeeze(2), torch.sum(self.label_desc_mask, dim=-1).unsqueeze(1))
        return label_embeds

    def forward(self, inputs, targets=None):
        # attn_mask: B x S -> B x S x 1
        attn_mask = (inputs != self.pad_idx).unsqueeze(2).to(self.device)
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S

        embeds = self.pos_encoder(self.embedder(inputs) * math.sqrt(self.embed_size))  # N x S x E
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # S x N x E

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # T x N x E
        encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x T x E

        # encoded_inputs is of shape: batch_size, seq_len, embed_size
        weighted_outputs, attn_weights = self.attn(encoded_inputs, attn_mask)
        # label_embeds = self.embed_label_desc()
        # weighted_outputs, attn_weights = self.label_attn(encoded_inputs, label_embeds, attn_mask)

        outputs = torch.zeros((weighted_outputs.size(0), self.output_size)).to(self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code+1] = fc(weighted_outputs[:, code, :])

        if targets is not None and self.class_margin is not None and self.C > 0:
            ldam_outputs = outputs - targets * self.class_margin * self.C
        else:
            ldam_outputs = None

        return outputs, ldam_outputs, attn_weights


class HierarchicalTransICD(nn.Module):
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len, num_layers, num_heads, forward_expansion,
                 output_size, sentence_len, sentence_num_layers, dropout_rate, device, label_freq=None, C=3.0,
                 pad_idx=0):
        super(HierarchicalTransICD, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        if sentence_len <= 0:
            raise ValueError("sentence_len should be > 0")

        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size
        self.sentence_len = sentence_len
        self.max_sentence_count = int(math.ceil(max_len / sentence_len))

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

        # Local encoding inside each sentence chunk
        self.token_pos_encoder = PositionalEncoding(embed_size, dropout_rate, sentence_len)
        token_encoder_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout_rate
        )
        self.token_encoder = TransformerEncoder(token_encoder_layer, num_layers)

        # Global encoding across sentence vectors
        self.sentence_pos_encoder = PositionalEncoding(embed_size, dropout_rate, self.max_sentence_count)
        sent_encoder_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout_rate
        )
        self.sentence_encoder = TransformerEncoder(sent_encoder_layer, sentence_num_layers)

        self.attn = CodeWiseAttention(embed_size, output_size)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for code in range(output_size)])

    def _to_sentence_chunks(self, inputs):
        batch_size, total_tokens = inputs.size()
        sentence_count = int(math.ceil(total_tokens / self.sentence_len))
        padded_token_count = sentence_count * self.sentence_len
        if padded_token_count != total_tokens:
            pad_len = padded_token_count - total_tokens
            pad_values = torch.full(
                (batch_size, pad_len),
                self.pad_idx,
                dtype=inputs.dtype,
                device=inputs.device
            )
            inputs = torch.cat([inputs, pad_values], dim=1)

        return inputs.view(batch_size, sentence_count, self.sentence_len)

    def _pool_sentence_vectors(self, token_vectors, token_mask):
        # token_vectors: (B*S) x L x H, token_mask: (B*S) x L
        token_mask = token_mask.float()
        masked_sum = (token_vectors * token_mask.unsqueeze(-1)).sum(dim=1)
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sentence_vectors = masked_sum / denom
        return sentence_vectors

    def forward(self, inputs, targets=None):
        # B x T -> B x S x L
        sentence_chunks = self._to_sentence_chunks(inputs)
        batch_size, sentence_count, sentence_len = sentence_chunks.size()

        token_mask = (sentence_chunks != self.pad_idx)  # B x S x L
        sentence_mask = token_mask.any(dim=-1)  # B x S

        flat_tokens = sentence_chunks.reshape(batch_size * sentence_count, sentence_len)  # (B*S) x L
        flat_token_mask = token_mask.reshape(batch_size * sentence_count, sentence_len)  # (B*S) x L

        # Token-level transformer per sentence chunk
        token_embeds = self.embedder(flat_tokens) * math.sqrt(self.embed_size)  # (B*S) x L x E
        token_embeds = self.token_pos_encoder(token_embeds)
        token_embeds = self.dropout(token_embeds).permute(1, 0, 2)  # L x (B*S) x E

        encoded_tokens = self.token_encoder(
            token_embeds,
            src_key_padding_mask=(~flat_token_mask).to(self.device)
        )  # L x (B*S) x E
        encoded_tokens = encoded_tokens.permute(1, 0, 2)  # (B*S) x L x E

        sentence_vectors = self._pool_sentence_vectors(encoded_tokens, flat_token_mask)
        sentence_vectors = sentence_vectors.view(batch_size, sentence_count, self.embed_size)  # B x S x E

        # Sentence-level transformer
        sentence_vectors = self.sentence_pos_encoder(sentence_vectors)
        sentence_vectors = self.dropout(sentence_vectors).permute(1, 0, 2)  # S x B x E

        encoded_sentences = self.sentence_encoder(
            sentence_vectors,
            src_key_padding_mask=(~sentence_mask).to(self.device)
        )  # S x B x E
        encoded_sentences = encoded_sentences.permute(1, 0, 2)  # B x S x E

        weighted_outputs, attn_weights = self.attn(encoded_sentences, sentence_mask)
        outputs = torch.zeros((weighted_outputs.size(0), self.output_size), device=self.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code+1] = fc(weighted_outputs[:, code, :])

        if targets is not None and self.class_margin is not None and self.C > 0:
            ldam_outputs = outputs - targets * self.class_margin * self.C
        else:
            ldam_outputs = None

        return outputs, ldam_outputs, attn_weights
