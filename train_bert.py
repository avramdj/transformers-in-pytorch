import yaml
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.datasets import ConfigurableDataset
from models import BertMaskedLM

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "true")

    parser = ArgumentParser()
    parser.add_argument("dataset_config")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--target_batch_size", default=16, type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset_size", default=None, type=int)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument(
        "--num_workers", default=10, help="num threads to use for dataloaders"
    )
    args = parser.parse_args()
    assert args.epochs > 0, "more epochs pls"

    with open(args.dataset_config) as f:
        ds_conf = yaml.full_load(f)

    epochs = args.epochs
    batch_size = args.batch_size
    target_batch_size = args.target_batch_size
    overfit_batches = 0.0
    enable_checkpointing = True
    log_every_n_steps = 50

    if args.sanity_check:
        batch_size = 1
        target_batch_size = 1
        overfit_batches = 1
        enable_checkpointing = False
        log_every_n_steps = 1

    dataset = ConfigurableDataset(ds_conf)
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
        log_every_n_steps=log_every_n_steps,
    )

    if args.checkpoint:
        model = BertMaskedLM.load_from_checkpoint(args.checkpoint)
        print("Loaded from checkpoint\n")
    else:
        model = BertMaskedLM(d_model=768, n_heads=12, n_layers=12, label_smoothing=0.1)

    trainer.batch_size = args.batch_size
    trainer.target_batch_size = args.target_batch_size
    if args.sanity_check:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
