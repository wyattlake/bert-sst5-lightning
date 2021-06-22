from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import torch

class SSTDataset(Dataset):
    def __init__(self, cfg, full_dataset, device, size=60, split="train"):
        tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.raw_dataset = full_dataset[split]
        self.dataset = [
            (
                self.pad([tokenizer.cls_token_id] + tokenizer.encode(item["sentence"]) +
                         [tokenizer.sep_token_id], size=size),
                self.map(item["label"]),
            )
            for item in self.raw_dataset
        ]
        self.device = device

    def pad(self, text, size=52):
        text_len = len(text)
        if text_len >= size:
            return text[:size]
        else:
            extra = size - len(text)
            return text + [0] * extra

    def map(self, val):
        if val == 0:
            return 0
        else:
            return np.ceil(val * 5) - 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text, label = self.dataset[index]
        text = torch.tensor(text, device=self.device)
        label = torch.tensor(label, device=self.device)
        return text, label
