import os

from pytorch_lightning.cli import LightningCLI

import data.datasets
import models

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "true")

    cli = LightningCLI(models.BertMaskedLM, data.datasets.ConfigurableDataModule)
