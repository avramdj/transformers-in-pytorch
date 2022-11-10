import pathlib

import pandas as pd
from torch.utils.data import Dataset

MAX_SEQ_LEN = 512


def parse_dataset(file_name, max_size=None, frac=None):
    ext = pathlib.Path(file_name).suffix[1:]
    actions = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "jsonl": lambda p: pd.read_json(p, lines=True),
    }
    assert ext in actions, "file type not supported"
    df = actions[ext](file_name)
    if max_size or frac:
        df = cut_shuffle_df(df, max_size=max_size, frac=frac)
    return df


def cut_shuffle_df(df, max_size=None, frac=None):
    assert bool(max_size) ^ bool(frac), "Provide either `max_size` or `frac`"
    if max_size:
        frac = min(max_size / len(df), 1)
    df = df.sample(frac=frac).reset_index(drop=True)
    return df


class _MetaWrapper(type):
    def __call__(cls, *args, **kwargs):
        dataset = super().__call__(*args, **kwargs)
        print(f"Dataset size: {len(dataset)}")
        return dataset


class WrappedDataset(Dataset, metaclass=_MetaWrapper):
    def __init__(self):
        super().__init__()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class ImdbDataset(WrappedDataset):
    def __init__(self, file_name, max_size=None):
        super().__init__()
        df = parse_dataset(file_name, max_size=max_size)
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
        df = parse_dataset(file_name, max_size=max_size)
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
        df = parse_dataset(file_name, max_size=max_size)
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
            self.df.gold_label[idx].lower(),
        )
