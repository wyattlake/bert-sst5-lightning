import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import torch


class BertSST(pl.LightningModule):
    def __init__(self, cfg, run):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.checkpoint, num_labels=5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.run = run

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, _batch_idx):
        x, y = batch
        logits = self.model(x).logits
        loss = self.criterion(logits, y.long())

        # Accuracy calculations
        pred_labels = torch.argmax(logits, axis=1)
        batch_acc = (pred_labels == y).sum().item()

        # Neptune logging
        self.run['train/loss'].log(loss.item() / self.batch_size)
        self.run['train/acc'].log(batch_acc / self.batch_size)

        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self.model(x).logits
        loss = self.criterion(logits, y.long())

        # Accuracy calculations
        pred_labels = torch.argmax(logits, axis=1)
        batch_acc = (pred_labels == y).sum().item()

        # Neptune logging
        self.run['eval/loss'].log(loss.item() / self.batch_size)
        self.run['eval/acc'].log(batch_acc / self.batch_size)

        return loss
