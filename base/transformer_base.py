import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MhaBlock(nn.Module):
    def __init__(self, d_model, n_heads=10):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), "embedding size `d_model` must be divisible by the number of attention heads `n_heads`"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_h = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return einops.rearrange(x, "b s (h j) -> b h s j", h=self.n_heads)

    def group_heads(self, x):
        return einops.rearrange(x, "b h s j  -> b s (h j)")

    def attention(self, q, k, v, mask):
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) / np.sqrt(self.d_k)
        if mask is not None:
            mask = einops.repeat(
                mask, "b s -> b h f s", h=self.n_heads, f=mask.shape[-1]
            )
            scores = scores.masked_fill(mask == 0, np.NINF)
        a = F.softmax(scores, dim=-1)
        x = a @ v
        return x

    def forward(self, x, q=None, mask=None):
        if q is None:
            q = x
        q = self.w_q(q)
        k = self.w_k(x)
        v = self.w_v(x)

        sq = self.split_heads(q)
        sk = self.split_heads(k)
        sv = self.split_heads(v)

        sh = self.attention(sq, sk, sv, mask)
        gh = self.group_heads(sh)
        out = self.w_h(gh)
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Linear(d_model, d_model)
        self.proj_2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.proj_2(self.act(self.proj_1(x)))


class Sublayer(nn.Module):
    def __init__(self, layer, d_model, dropout=0.1):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        l = self.layer(x, **kwargs)
        l = self.dropout(l)
        return self.norm(x + l)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha_sublayer = Sublayer(MhaBlock(d_model, n_heads), d_model, dropout)
        self.ffn_sublayer = Sublayer(PositionWiseFFN(d_model), d_model, dropout)

    def forward(self, x, mask=None):
        x = self.mha_sublayer(x, mask=mask)
        x = self.ffn_sublayer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mmha_sublayer = Sublayer(MhaBlock(d_model, n_heads), d_model, dropout)
        self.mha_sublayer = Sublayer(MhaBlock(d_model, n_heads), d_model, dropout)
        self.ffn_sublayer = Sublayer(PositionWiseFFN(d_model), d_model, dropout)

    def forward(self, x, mask=None):
        masked_q = self.mmha_sublayer(x, mask=mask)
        x = self.mha_sublayer(x, q=masked_q)
        x = self.ffn_sublayer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
