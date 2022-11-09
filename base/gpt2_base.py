from torch import nn

from base.transformer_base import Decoder, PositionalEncoding


class GPT2Base(nn.Module):
    def __init__(self, n_tokens, d_model=768, n_layers=10, n_heads=6):
        super().__init__()
        self.word_emb = nn.Embedding(n_tokens + 1, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.decoder = Decoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)

    def forward(self, ids, mask):
        x = self.word_emb(ids)
        x = self.pos_emb(x)
        x = self.decoder(x, mask)
        return x
