from torch import nn

from base.positional_base import LearnedPositionalEncoding
from base.transformer_base import Decoder


class GPT2Embedding(nn.Module):
    def __init__(self, n_tokens, d_model=768):
        super().__init__()
        self.word_embeddings = nn.Embedding(n_tokens, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model)

    def forward(self, ids):
        x = self.word_embeddings(ids)
        x = x + self.pos_emb(x)
        return x

class GPT2Base(nn.Module):
    def __init__(self, n_tokens, d_model=768, n_layers=10, n_heads=6, dropout=0.1):
        super().__init__()
        self.emb = GPT2Embedding(n_tokens, d_model)
        self.decoder = Decoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ids, mask):
        x = self.emb(ids)
        x = self.dropout(x)
        x = self.decoder(x, mask)
        return x
