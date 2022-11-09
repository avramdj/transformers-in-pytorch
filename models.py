import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from base.bert_base import BERT
from base.gpt2_base import GPT2Base


class BertMaskedLM(pl.LightningModule):
    def __init__(
        self,
        d_model=768,
        n_layers=12,
        n_heads=12,
        mask_prob=0.1,
        tokenizer_name="distilbert-base-uncased",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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

    def training_step(self, train_batch, batch_idx):

        if batch_idx % 200 == 0:
            self.print_prompt("The movie was very [MASK] and boring")

        x, _ = train_batch
        x = list(x)

        t = self.tokenizer(x, return_tensors="pt", padding=True)
        ids = t["input_ids"]
        attn_mask = t["attention_mask"]

        if self.device.type == "cuda":
            ids = ids.cuda()
            attn_mask = attn_mask.cuda()

        fill_mask = torch.rand(ids.shape).to(attn_mask.device.type) < self.mask_prob
        special_mask = (
            torch.isin(
                ids, torch.Tensor(self.tokenizer.all_special_ids).to(ids.device.type)
            )
            == 0
        )
        fill_mask = fill_mask * special_mask  # we don't want to mask special tokens

        masked_ids = torch.clone(ids)
        masked_ids[fill_mask] = self.mask_id

        o = self(masked_ids, attn_mask)

        oh_target = F.one_hot(ids, num_classes=self.tokenizer.vocab_size).type(o.dtype)
        loss = F.cross_entropy(o, oh_target)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        x = list(x)

        t = self.tokenizer(x, return_tensors="pt", padding=True)
        ids = t["input_ids"]
        attn_mask = t["attention_mask"]

        if self.device.type == "cuda":
            ids = ids.cuda()
            attn_mask = attn_mask.cuda()

        fill_mask = torch.rand(ids.shape).to(attn_mask.device.type) < self.mask_prob
        fill_mask = fill_mask * attn_mask  # we don't want to mask [PAD] tokens
        pre_mask = ids[fill_mask]

        ids[fill_mask] = self.mask_id
        o = self(ids, attn_mask)
        ids[fill_mask] = pre_mask

        oh_target = F.one_hot(ids, num_classes=self.tokenizer.vocab_size).type(o.dtype)
        loss = F.cross_entropy(o, oh_target)

        self.log("val_loss", loss, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        pass

    def print_prompt(self, prompt):
        with torch.no_grad():
            t = self.tokenizer([prompt], return_tensors="pt", padding=True)
            ids = t["input_ids"]
            attn_mask = t["attention_mask"]

            if self.device.type == "cuda":
                ids = ids.cuda()
                attn_mask = attn_mask.cuda()

            o = self(ids, attn_mask)
            m = torch.topk(o[:, -1], 5, dim=-1)
            indices = m.indices[0]
            values = m.values[0]
            mask_idx = ids == self.mask_id
            print()
            for i, v in sorted(list(zip(indices, values)), key=lambda x: -x[1]):
                ids[mask_idx] = i
                print(f"p:{v:0.2f}\t{self.tokenizer.decode(ids[0])}")
            print()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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
        loss = F.binary_cross_entropy(o, y)

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
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.vocab_size = self.tokenizer.vocab_size
        self.decode = GPT2Base(
            self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.output = nn.Linear(d_model, self.vocab_size)

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x = list(x)

        o = self(x)
        loss = F.cross_entropy(
            o,
        )
        # perplexity = torch.exp(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = list(x)
        y = y.float()

        o = self(x)
        loss = F.cross_entropy(o, y, reduction="mean")
        # perplexity = torch.exp(loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-05)

    def generate(self, prompt, n=1):
        pass

    def on_train_epoch_end(self):
        self.generate("the meaning of ", n=10)

    def forward(self, x):
        if type(x) == "str":
            x = [x]
        t = self.tokenizer(x, return_tensors="pt", padding=True)
        x = t["input_ids"]
        attn_mask = t["attention_mask"]
        if self.device.type == "cuda":
            x = x.cuda()
            attn_mask = attn_mask.cuda()
        x = self.decode(x, mask=attn_mask)
        x = x[:, -1]
        x = self.output(x)
        x = F.softmax(x, dim=-1)
        return x[:, -1]
