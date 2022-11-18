import numpy as np
import torch
from torch import nn

MAX_POS_LEN = 5000


class PositionalEncoding(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        pid = torch.arange(self.max_len)
        self.register_buffer("pid", pid)

    def seq_to_pos_idx(self, x):
        return self.pid[: x.size(1)]


class LearnedPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, max_len=MAX_POS_LEN):
        super().__init__(max_len)
        self.enc = nn.Embedding(self.max_len, d_model)

    def forward(self, x):
        pos_ids = self.seq_to_pos_idx(x)
        return self.enc(pos_ids)


class SinCosPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, max_len=MAX_POS_LEN):
        super().__init__(max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(self.max_len, d_model)
        pe[:, 0::2] = torch.sin(self.pid * div_term)
        pe[:, 1::2] = torch.cos(self.pid * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[: x.size(1)]
