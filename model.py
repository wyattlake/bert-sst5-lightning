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
        self.run = run

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs).logits
        loss = self.criterion(logits, targets.long())

        # Accuracy calculations
        preds = torch.argmax(logits, axis=1)
        acc = self.calc_acc(preds, targets)

        # Neptune logging
        self.run['train/loss'].log(loss.item() / self.batch_size)
        self.run['train/acc'].log(acc)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def training_epoch_end(self, outputs):
        # Evaluating epoch averages
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Neptune logging
        self.run['train_epoch/loss'].log(loss.item())
        self.run['train_epoch/acc'].log(self.calc_acc(preds, targets))

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs).logits
        loss = self.criterion(logits, targets.long())

        preds = torch.argmax(logits, axis=1)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs):
        # Evaluating epoch averages
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Neptune logging
        self.run['eval_epoch/loss'].log(loss.item())
        self.run['eval_epoch/acc'].log(self.calc_acc(preds, targets))

    def calc_acc(self, preds, targets):
        return 100 * (preds == targets).float().mean()
