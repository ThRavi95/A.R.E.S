import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizer import encode

class PeptideDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)

        if "sequence" not in df.columns:
            raise ValueError("CSV must contain a 'sequence' column.")

        sequences = (
            df["sequence"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        tokens = torch.tensor(encode(seq), dtype=torch.long)
        return tokens