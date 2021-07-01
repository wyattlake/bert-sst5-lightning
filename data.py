from collections import namedtuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
import numpy as np
import torch
import csv


class SSTDataset(Dataset):
    def __init__(self, cfg, full_dataset, device, size=60, split="train"):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.raw_dataset = full_dataset[split]
        self.dataset = []

        # Reformatting data
        for item in self.raw_dataset:
            token_ids = self.pad([self.tokenizer.cls_token_id] + self.tokenizer.encode(item["sentence"]) +
                                 [self.tokenizer.sep_token_id], size=size)
            new_item = [
                token_ids,
                self.map(item["label"]),
                ([1]*len(token_ids) + [0]*(size - len(token_ids))),
            ]
            self.dataset.append(new_item)

        self.device = device
        self.cfg = cfg
        self.has_explanations = False

    def load_explanations(self):
        self.has_explanations = True
        with open(self.cfg.data.load_csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for idx, item in enumerate(reader):
                converted = []
                for sub in item:
                    sub = float(sub)
                    converted.append(sub)
                self.dataset[idx].append(converted)

    def generate_explanations(self, model, k=10):
        self.teacher = model
        self.has_explanations = True
        self.attribution_fn = LayerIntegratedGradients(
            self.captum_forward, self.teacher.bert.embeddings.word_embeddings)
        print("Generating Explanations")
        if self.cfg.data.save_explanations:
            with open(self.cfg.data.save_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for item in tqdm(self.dataset):
                    attributions, tokens = self.generate_float_attribution(
                        item)

                    # Finding indices with float attributions in the top k percent
                    topk_indices = torch.topk(attributions, int(
                        attributions.shape[0] * k / 100), sorted=False).indices

                    # Building the binary explanation from the float attributions
                    binary_explanation = torch.ones_like(
                        torch.tensor([item[0]], device=self.device)) * (1e-8)
                    binary_explanation.index_fill_(1, topk_indices, 1)

                    writer.writerow(binary_explanation.squeeze().tolist())

                    # Appending new information to the data
                    item.append(binary_explanation)
                    item.append(tokens)
        else:
            for item in tqdm(self.dataset):
                attributions, tokens = self.generate_float_attribution(item)

                # Finding indices with float attributions in the top k percent
                topk_indices = torch.topk(attributions, int(
                    attributions.shape[0] * k / 100), sorted=False).indices

                # Building the binary explanation from the float attributions
                binary_explanation = torch.ones_like(
                    torch.tensor([item[0]], device=self.device)) * (1e-8)
                binary_explanation.index_fill_(1, topk_indices, 1)

                # Appending new information to the data
                item.append(binary_explanation)
                item.append(tokens)

    def generate_float_attribution(self, item):
        token_ids = torch.tensor(
            item[0], device=self.device).unsqueeze(0)
        target = torch.tensor(item[1], dtype=torch.int64, device=self.device)
        attention_mask = torch.tensor(
            item[2], device=self.device).unsqueeze(0)
        attributions = self.attribution_fn.attribute(inputs=token_ids, target=target,
                                                     additional_forward_args=attention_mask)
        attributions = self.summarize_attributions(attributions)
        all_tokens = self.tokenizer.convert_ids_to_tokens(item[0])
        return attributions, all_tokens

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def captum_forward(self, input_ids, attention_mask=None):
        output = self.teacher(input_ids, attention_mask=attention_mask)[0]
        return output

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
        if self.has_explanations:
            explanation = torch.tensor(
                self.dataset[index][3], device=self.device)
            return text, label, explanation
        else:
            return text, label
