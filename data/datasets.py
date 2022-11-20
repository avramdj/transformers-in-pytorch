import json
import os
import pathlib
from pydoc import locate
from typing import Iterable, List

import pandas as pd
import pytorch_lightning as pl
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

MAX_SEQ_LEN = 512


class Config(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    class _ConfigList(list):
        def __getitem__(*args):
            item = list.__getitem__(*args)
            return Config._cast(item)

        def __iter__(self):
            self.n = 0
            return self

        def __next__(self):
            if self.n < len(self):
                result = self[self.n]
                self.n += 1
                return result
            else:
                raise StopIteration

    @staticmethod
    def _cast(item):
        if type(item) is dict:
            return Config(item)
        if type(item) is list:
            return Config._ConfigList(item)
        return item

    def __getattr__(*args):
        item = dict.get(*args)
        return Config._cast(item)


def _parse_dataset(file_name, max_size=None, frac=None):
    ext = pathlib.Path(file_name).suffix[1:]
    actions = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "jsonl": lambda p: pd.read_json(p, lines=True),
    }
    assert ext in actions, "file type not supported"
    df = actions[ext](file_name)
    if max_size or frac:
        df = _cut_shuffle_df(df, max_size=max_size, frac=frac)
    return df


def _apply(col, fn):
    if not isinstance(col, Iterable) or isinstance(col, str):
        return fn(col)
    if isinstance(col, pd.Series):
        return col.map(fn)
    if isinstance(col, list):
        return type(col)(map(fn, col))
    raise NotImplementedError(f'Apply "{type(col)}" for not implemented')


def _cut_shuffle_df(df, max_size=None, frac=None):
    df = df.sample(n=max_size, frac=frac).reset_index(drop=True)
    return df


class _MetaWrapper(type):
    def __call__(cls, *args, **kwargs):
        dataset = super().__call__(*args, **kwargs)
        print(f"Dataset size: {len(dataset)}")
        return dataset


class WrappedDataset(Dataset, metaclass=_MetaWrapper):
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class ConfigurableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config: str,
        batch_size: int,
        validation_size: float,
        num_workers: int = 10,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.num_workers = num_workers
        self.train_dataset, self.val_dataset = None, None

    def prepare_data(self):
        dataset = ConfigurableDataset.from_yaml(self.dataset_config)
        splits = train_test_split(dataset, test_size=self.validation_size)
        self.train_dataset, self.val_dataset = splits

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ConfigurableDataset(WrappedDataset):
    DEFAULT_TYPE = str

    def __init__(self, config: dict):
        super().__init__()
        self.config = Config(config)
        self._col_type_map = self._get_col_type_map()
        self._col_transform_map = self._get_col_transform_map()
        self.df = _parse_dataset(self.config.path)

    def _get_col_type_map(self):
        return {
            x.name: locate(x.type) if "type" in x else self.DEFAULT_TYPE
            for x in self.config.cols
        }

    def _get_col_transform_map(self):
        for col in self.config.cols:
            if "transforms" in col and not bool(os.getenv("ALLOW_TRANSFORMS", False)):
                raise Exception(
                    """
                Transforms should be avoided because of potential arbitrary code execution.
                It's much better and safer to preprocess your datasets.
                To enable them, set the ALLOW_TRANSFORMS environment variable
                """
                )
        return {
            x.name: self._create_transform_pipe(x.transforms)
            if "transforms" in x
            else lambda x: x
            for x in self.config.cols
        }

    def _create_transform_pipe(self, strs_callable):
        pipe = []
        if not isinstance(strs_callable, list):
            strs_callable = [strs_callable]
        for str_callable in strs_callable:
            transform_fn = eval(str_callable)
            assert callable(transform_fn), "Not a valid transform"
            pipe.append(transform_fn)
        return pipe

    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path) as f:
            config = yaml.full_load(f)
        return cls(config)

    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as f:
            config = json.load(f)
        return cls(config)

    def _parse_col(self, col, index):
        items = self.df[col.name][index]
        pipe = self._col_transform_map[col.name]
        for fn in pipe:
            items = _apply(items, fn)
        items = self._col_type_map[col.name](items)
        return items

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return tuple([self._parse_col(col, index) for col in self.config.cols])


class ImdbDataset(WrappedDataset):
    def __init__(self, file_name, max_size=None):
        super().__init__()
        df = _parse_dataset(file_name, max_size=max_size)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.df.review[idx][-MAX_SEQ_LEN:].lower(),
            float(self.df.sentiment[idx] == "positive"),
        )


class TweetDataset(WrappedDataset):
    def __init__(self, file_name, max_size=None):
        super().__init__()
        df = _parse_dataset(file_name, max_size=max_size)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.df.tweet[idx][:MAX_SEQ_LEN].lower(),
            float(self.df.target[idx] == 4),
        )


class MNLI(WrappedDataset):
    def __init__(self, file_name, max_size=None):
        super().__init__()
        df = _parse_dataset(file_name, max_size=max_size)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.df.sentence1[idx][:MAX_SEQ_LEN].lower(),
            self.df.sentence2[idx][:MAX_SEQ_LEN].lower(),
        )


class MNLIClassification(MNLI):
    def __init__(self, file_name, max_size=None):
        super().__init__(file_name, max_size)

    def __getitem__(self, idx):
        return (
            self.df.sentence1[idx][:MAX_SEQ_LEN].lower(),
            self.df.gold_label[idx],
        )
