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
        self.cfg = cfg

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
        self.has_explanations = False

    def load_explanations(self, path):
        self.has_explanations = True
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for idx, item in enumerate(reader):
                converted = []
                for sub in item:
                    sub = float(sub)
                    converted.append(sub)
                self.dataset[idx].append(converted)

    def generate_explanations(self, model, path, k=10):
        self.teacher = model
        self.has_explanations = True
        self.attribution_fn = LayerIntegratedGradients(
            self.captum_forward, self.teacher.bert.embeddings.word_embeddings)
        print("Generating Explanations")
        if self.cfg.data.save_explanations:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for item in tqdm(self.dataset):
                    attributions = self.generate_float_attribution(
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
        else:
            for item in tqdm(self.dataset):
                attributions = self.generate_float_attribution(item)

                # Finding indices with float attributions in the top k percent
                topk_indices = torch.topk(attributions, int(
                    attributions.shape[0] * k / 100), sorted=False).indices

                # Building the binary explanation from the float attributions
                binary_explanation = torch.ones_like(
                    torch.tensor([item[0]], device=self.device)) * (1e-8)
                binary_explanation.index_fill_(1, topk_indices, 1)

                # Appending new information to the data
                item.append(binary_explanation)

    def generate_float_attribution(self, item):
        token_ids = torch.tensor(
            item[0], device=self.device).unsqueeze(0)
        target = torch.tensor(item[1], dtype=torch.int64, device=self.device)
        attention_mask = torch.tensor(
            item[2], device=self.device).unsqueeze(0)
        all_attributions = []

        # Generating explanations for each of the 5 possible labels
        for i in range(self.cfg.label_count):
            attributions = self.attribution_fn.attribute(inputs=token_ids, target=i,
                                                         additional_forward_args=attention_mask)
            attributions = self.summarize_attributions(attributions)

            # Attributions are flipped if the label is not the target
            if i != target:
                torch.mul(attributions, -1)

            # Attributions are standardized and added to the list
            all_attributions.append(
                self.standardize_attributions(attributions))

        # Target and non target attributions are averaged
        stacked_attributions = torch.stack(all_attributions)
        averaged_attributions = stacked_attributions.mean(dim=0)

        return averaged_attributions

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def captum_forward(self, input_ids, attention_mask=None):
        output = self.teacher(input_ids, attention_mask=attention_mask)[0]
        return output

    def standardize_attributions(self, tensor):
        means = tensor.mean()
        stds = tensor.std()
        return (tensor - means) / stds

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
            return np.ceil(val * self.cfg.label_count) - 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = torch.tensor(self.dataset[index][0], device=self.device)
        label = torch.tensor(self.dataset[index][1], device=self.device)
        attention_mask = torch.tensor(
            self.dataset[index][2], device=self.device)
        if self.has_explanations:
            explanation = torch.tensor(
                self.dataset[index][3], device=self.device)
            return text, label, attention_mask, explanation
        else:
            return text, label, attention_mask
