import pandas as pd
from torch.utils.data import Dataset

MAX_SEQ_LEN = 512


def cut_shuffle_df(df, max_size):
    if max_size:
        frac = min(max_size / len(df), 1)
        df = df.sample(frac=frac).reset_index(drop=True)
    return df


class ImdbDataset(Dataset):
    def __init__(self, file_name, max_size=None):
        super().__init__()
        df = pd.read_csv(file_name)
        cut_shuffle_df(df, max_size)
        self.df = df
        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.review[idx][-MAX_SEQ_LEN:].lower(), float(
            self.df.sentiment[idx] == "positive"
        )


class TweetDataset(Dataset):
    def __init__(self, file_name, max_size=None):
        super().__init__()
        df = pd.read_csv(file_name)
        cut_shuffle_df(df, max_size)
        df = df[df.columns[[0, -1]]]
        df.columns = ["target", "tweet"]
        self.df = df
        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # target: (0 = negative, 4 = positive)
        return self.df.tweet[idx][:MAX_SEQ_LEN].lower(), float(self.df.target[idx] == 4)
