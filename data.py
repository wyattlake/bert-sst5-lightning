from collections import namedtuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients
import numpy as np
import torch


class SSTDataset(Dataset):
    def __init__(self, cfg, full_dataset, device, size=60, split="train"):
        tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.raw_dataset = full_dataset[split]
        self.has_explanations = False
        self.dataset = []

        # Reformatting data
        for item in self.raw_dataset:
            token_ids = self.pad([tokenizer.cls_token_id] + tokenizer.encode(item["sentence"]) +
                                 [tokenizer.sep_token_id], size=size)
            new_item = [
                token_ids,
                self.map(item["label"]),
                ([1]*len(token_ids) + [0]*(size - len(token_ids))),
            ]
            self.dataset.append(new_item)

        self.device = device

    def generate_explanations(self, model, k=10):
        self.has_explanations = True
        self.attribution_fn = IntegratedGradients(model)
        for item in self.dataset:
            attribution = self.generate_float_attribution(item)
            item.append()

    def generate_float_attribution(self, item):
        token_ids = torch.tensor(
            item[0], dtype=torch.float32, device=self.device).unsqueeze(0)
        target = int(item[1])
        attention_mask = torch.tensor(
            item[3], device=self.device).unsqueeze(0)
        attributions = self.attribution_fn.attribute(inputs=token_ids, target=target,
                                                     additional_forward_args=attention_mask)
        return self.summarize_attributions(attributions)

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

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
        text = torch.tensor(self.dataset[index][0], device=self.device)
        label = torch.tensor(self.dataset[index][1], device=self.device)
        if not self.has_explanations:
            return text, label
        else:
            explanation = torch.tensor(
                self.dataset[index][3], device=self.device)
            return text, label, explanation
