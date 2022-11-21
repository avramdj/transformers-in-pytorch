from torch.optim.lr_scheduler import LambdaLR

class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1, warmup_steps=1, verbose=False):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose)

def lr_warmup_lambda(current_step: int, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    return 1.0