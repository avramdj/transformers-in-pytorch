from torch import nn
from torch.nn import functional as F


class LabelSmoothing(nn.Module):
    def __init__(self, pad_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="mean")
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        vocab_size = x.size(-1)
        if x.shape != target.shape:
            target = F.one_hot(target, num_classes=vocab_size)
        target = self.confidence * target + self.smoothing / (vocab_size - 1)
        target[:, self.pad_idx] = 0.0
        return self.criterion(x, target)
