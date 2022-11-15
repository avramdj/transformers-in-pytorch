import os

import einops
import pytorch_lightning as pl
import torch
import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from base.bert_base import BERT
from base.gpt2_base import GPT2Base


class BertMaskedLM(pl.LightningModule):
    def __init__(
        self,
        d_model=768,
        n_layers=12,
        n_heads=12,
        mask_prob=0.15,
        tokenizer_name="distilbert-base-uncased",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_str = self.tokenizer.special_tokens_map["mask_token"]
        self.mask_id = self.tokenizer.vocab[self.mask_str]
        self.mask_prob = mask_prob
        self.encode = BERT(
            self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.out = nn.Linear(d_model, self.tokenizer.vocab_size)

    def mask_random(self, attn_mask, prob=0.1):
        """
        Every attended-to token in the sequence has `prob` probability of being masked.
        """
        assert prob > 0 and prob < 1, "what the hell u doin"
        return torch.vstack(
            [torch.rand(row.shape) > (prob * row.sum() / len(row)) for row in attn_mask]
        )

    def step(self, x):
        t = self.tokenizer(x, return_tensors="pt", padding=True)
        ids = t["input_ids"]
        attn_mask = t["attention_mask"]

        if self.device.type == "cuda":
            ids = ids.cuda()
            attn_mask = attn_mask.cuda()

        rand_dist = torch.rand(ids.shape).to(attn_mask.device.type)
        # fill `self.mask_prob * 100` % of tokens with [MASK]
        fill_mask = rand_dist < self.mask_prob
        # fill 10% of the masked tokens with random words instead
        fill_mask_random = rand_dist < self.mask_prob * 0.20
        # leave 10% of the masked tokens unchanged
        fill_mask_unchanged = rand_dist < self.mask_prob * 0.10

        non_special_mask = (
            torch.isin(
                ids, torch.Tensor(self.tokenizer.all_special_ids).to(ids.device.type)
            )
            == 0
        )
        # we don't want to mask special tokens
        fill_mask = fill_mask * non_special_mask
        fill_mask_random = fill_mask_random * non_special_mask
        fill_mask_unchanged = fill_mask_unchanged * non_special_mask

        # get random tokens to fill in `fill_mask_random`
        fill_mask_random_values = torch.randint(
            1, self.vocab_size, fill_mask_random.shape
        ).to(attn_mask.device.type)

        # filling
        masked_ids = torch.clone(ids)
        masked_ids[fill_mask] = self.mask_id
        masked_ids[fill_mask_random] = fill_mask_random_values[fill_mask_random]
        masked_ids[fill_mask_unchanged] = ids[fill_mask_unchanged]

        o = self(masked_ids, attn_mask)

        o = einops.rearrange(o, "b s v -> (b s) v")
        target = einops.rearrange(ids, "b s -> (b s)")
        flat_mask = fill_mask.view(-1)
        o = o[flat_mask]
        target = target[flat_mask]
        loss = F.cross_entropy(o, target, reduction="mean")
        return loss

    def training_step(self, train_batch, batch_idx):

        if batch_idx % 100 == 0:
            self.print_prompt("You are [MASK] rude")
            self.print_prompt("The movie was very [MASK] and boring")

        s1, s2 = train_batch
        s1 = list(s1)
        s2 = list(s2)
        s = s1

        loss = self.step(s)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        s1, s2 = val_batch
        s1 = list(s1)
        s2 = list(s2)
        s = s1

        loss = self.step(s)

        self.log("val_loss", loss, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        save_path = os.path.join(
            self.trainer.log_dir, f"encoder-{self.trainer.global_step}"
        )
        with open(save_path, "wb") as f:
            torch.save(self.encode, f)
            print(f"Saved base checkpoint at {save_path}")

    def print_prompt(self, prompt):
        with torch.no_grad():
            t = self.tokenizer([prompt], return_tensors="pt", padding=True)
            ids = t["input_ids"]
            attn_mask = t["attention_mask"]

            if self.device.type == "cuda":
                ids = ids.cuda()
                attn_mask = attn_mask.cuda()

            o = self(ids, attn_mask)
            m = torch.topk(o[:, -1], 3, dim=-1)
            indices = m.indices[0]
            values = m.values[0]
            mask_idx = ids == self.mask_id
            print()
            for i, v in sorted(list(zip(indices, values)), key=lambda x: -x[1]):
                ids[mask_idx] = i
                print(f"p:{v:0.2f}\t{self.tokenizer.decode(ids[0])}")
            print()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 10000, self.trainer.max_epochs * 100000
        )
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "val_loss",
                }
            ],
        )

    def forward(self, ids, mask=None):
        x = self.encode(ids, mask=mask)
        x = self.out(x)
        x = F.softmax(x, dim=-1)
        return x


class BertClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        d_model=768,
        n_layers=12,
        n_heads=12,
        tokenizer_name="distilbert-base-uncased",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encode = BERT(
            self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.output = nn.Linear(d_model, n_classes)

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = list(x)
        y = y.float()

        o = self(x)
        loss = F.binary_cross_entropy(o, y)  # perplexity loss

        self.train_acc(o, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = list(x)
        y = y.float()

        o = self(x)
        loss = F.binary_cross_entropy(o, y)

        self.val_acc(o, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-05)

    def forward(self, x):
        if type(x) == "str":
            x = [x]
        t = self.tokenizer(x, return_tensors="pt", padding=True)
        x = t["input_ids"]
        attn_mask = t["attention_mask"]
        if self.device.type == "cuda":
            x = x.cuda()
            attn_mask = attn_mask.cuda()
        x = self.encode(x, mask=attn_mask)
        x = x[:, 0]  # take only [CLS] token
        x = self.output(x)
        x = F.softmax(x, dim=-1) if self.n_classes > 2 else torch.sigmoid(x)
        return x[:, 0]


class GPT2(pl.LightningModule):
    def __init__(self, d_model=768, n_layers=12, n_heads=12, tokenizer_name="gpt2"):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pad_str_token = "[PAD]"
        self.tokenizer.add_special_tokens({"pad_token": self.pad_str_token})
        self.eos_token = self.tokenizer.special_tokens_map["eos_token"]
        self.eos_id = self.tokenizer.vocab[self.eos_token]
        self.pad_id = self.tokenizer.vocab[self.pad_str_token]
        self.vocab_size = self.tokenizer.vocab_size + 1
        self.decode = GPT2Base(
            self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.output = nn.Linear(d_model, self.vocab_size)

    def step(self, x):
        t = self.tokenizer(x, return_tensors="pt", padding=True)
        ids = t["input_ids"]
        attn_mask = t["attention_mask"]
        if self.device.type == "cuda":
            ids = ids.cuda()
            attn_mask = attn_mask.cuda()

        o = self(ids, attn_mask)

        trg = torch.roll(ids, -1)
        trg[:, -1] = self.pad_id
        first_pad = (ids != self.pad_id) != (trg != self.pad_id)
        trg[first_pad] = self.eos_id

        o = einops.rearrange(o, "b s v -> (b s) v")
        trg = einops.rearrange(trg, "b s -> (b s)")

        filter_pads = trg != self.pad_id
        o = o[filter_pads]
        trg = trg[filter_pads]

        return F.cross_entropy(o, trg, reduction="mean")

    def training_step(self, train_batch, batch_idx):

        if batch_idx % 100 == 0:
            self.generate("You are so", n=3)
            self.generate("The movie was", n=3)

        x, _ = train_batch
        x = list(x)

        loss = self.step(x)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        x = list(x)

        loss = self.step(x)

        self.log("val_loss", loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10000
        scaled_warmup_steps = int(
            warmup_steps * self.trainer.target_batch_size // self.trainer.batch_size
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-04)
        # return optimizer
        scheduler = get_constant_schedule_with_warmup(optimizer, scaled_warmup_steps)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "val_loss",
                }
            ],
        )

    def generate(self, prompt, n=1):
        for _ in range(n):
            ids_completed = self._generate(prompt)
            prompt = self.tokenizer.decode(ids_completed, skip_special_tokens=True)
            if ids_completed[-1] == self.eos_id:
                break
        print(prompt)

    def _generate(self, prompt):
        with torch.no_grad():
            t = self.tokenizer([prompt], return_tensors="pt", padding=True)
            ids = t["input_ids"]
            attn_mask = t["attention_mask"]
            if self.device.type == "cuda":
                ids = ids.cuda()
                attn_mask = attn_mask.cuda()

            o = self(ids, attn_mask)
            o = o[0]
            next_token = torch.argmax(o, dim=-1)[-1]
            ids_completed = torch.concat([ids[0], next_token[None]])
            return ids_completed

    def on_train_epoch_end(self):
        self.generate("the meaning of ", n=10)

    def forward(self, ids, attn_mask):
        x = self.decode(ids, mask=attn_mask)
        x = self.output(x)
        x = F.softmax(x, dim=-1)
        return x
