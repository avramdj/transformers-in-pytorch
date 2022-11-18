from torch import nn

from base.positional_base import LearnedPositionalEncoding
from base.transformer_base import Decoder


class GPT2Base(nn.Module):
    def __init__(self, n_tokens, d_model=768, n_layers=10, n_heads=6, dropout=0.1):
        super().__init__()
        self.word_emb = nn.Embedding(n_tokens, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model)
        self.decoder = Decoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ids, mask):
        x = self.word_emb(ids)
        x = self.pos_emb(x)
        x = self.dropout(x)
        x = self.decoder(x, mask)
        return x
