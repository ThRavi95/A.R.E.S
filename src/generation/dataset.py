import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizer import encode

class PeptideDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.data = df["sequence"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        encoded = torch.tensor(encode(seq), dtype=torch.long)
        return encoded