import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import torch


class BertSST(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.checkpoint, num_labels=5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = cfg.learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x).logits
        loss = self.criterion(logits, y.long())
        return loss
