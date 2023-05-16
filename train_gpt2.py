import os

from pytorch_lightning.cli import LightningCLI
import torch
# from rich import traceback
# traceback.install(show_locals=True)


import data.datasets
import models

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "true")

    torch.set_float32_matmul_precision("medium")

    cli = LightningCLI(models.GPT2, data.datasets.ConfigurableDataModule)
