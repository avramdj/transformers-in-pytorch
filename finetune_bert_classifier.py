import json
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import data.datasets as datasets
from models import BertClassifier, BertMaskedLM

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

    with open("datasets_config.json") as f:
        ds_conf = json.load(f)

    parser = ArgumentParser()
    parser.add_argument("dataset", choices=list(ds_conf.keys()))
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--target_batch_size", default=16, type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--dataset_size", default=None, type=int)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument(
        "--num_workers", default=10, help="num threads to use for dataloaders"
    )
    args = parser.parse_args()

    assert args.epochs > 0, "more epochs pls"

    epochs = args.epochs
    batch_size = args.batch_size
    target_batch_size = args.target_batch_size
    overfit_batches = 0.0
    enable_checkpointing = True

    if args.sanity_check:
        batch_size = 1
        target_batch_size = 1
        overfit_batches = 1
        enable_checkpointing = False

    ds_class = getattr(sys.modules[datasets.__name__], ds_conf[args.dataset]["class"])
    ds_path = ds_conf[args.dataset]["path"]
    assert ds_path, "Didn't set the dataset path in config file"
    dataset = ds_class(ds_path, max_size=args.dataset_size)
    train_dataset, val_dataset = train_test_split(dataset, train_size=0.8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=args.num_workers
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=args.device,
        accumulate_grad_batches=target_batch_size // batch_size,
        overfit_batches=overfit_batches,
        enable_checkpointing=enable_checkpointing,
    )
    trainer.batch_size = args.batch_size
    trainer.target_batch_size = args.target_batch_size

    if args.checkpoint:
        model = BertClassifier.load_from_checkpoint(args.checkpoint)
        print("Loaded from checkpoint\n")
    else:
        model = BertClassifier(n_classes=2, d_model=768, n_heads=12, n_layers=12)
        if args.base_checkpoint:
            with open(args.base_checkpoint, "rb") as f:
                model.encode = torch.load(f)
            print("Loaded base BERT encoder from checkpoint")

    trainer.fit(model, train_loader, val_loader)
