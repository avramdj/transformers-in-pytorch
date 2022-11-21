import torch
from torch import nn
from torch.nn import functional as F

from base.positional_base import LearnedPositionalEncoding
from base.transformer_base import Encoder


class BertEmbedding(nn.Module):
    def __init__(self, n_tokens, d_model=768):
        super().__init__()
        self.word_embeddings = nn.Embedding(n_tokens, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model)

    def forward(self, ids):
        x = self.word_embeddings(ids)
        x = x + self.pos_emb(x)
        return x


class BertBase(nn.Module):
    def __init__(self, n_tokens, d_model=768, n_layers=12, n_heads=12, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbedding(n_tokens=n_tokens, d_model=d_model)
        self.encoder = Encoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self):
        """
        initialization from https://huggingface.co/bert-base-uncased/raw/main/config.json
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, ids, mask=None):
        x = self.embeddings(ids)
        x = self.dropout(x)
        x = self.encoder(x, mask=mask)
        return x


class BertLMPredictionHead(nn.Module):
    def __init__(self, d_model, vocab_size, word_embedding_layer=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        # tying weights between final dense and input embeddings
        if word_embedding_layer is not None:
            self.decoder.weight = word_embedding_layer.weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def tie_embeddings(self, word_embs):
        self.decoder.weight = word_embs.weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        x = F.gelu(x)
        x = self.LayerNorm(x)
        return x
