import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import lr_scheduler
import torch


class BertSST(pl.LightningModule):
    def __init__(self, cfg, run, train_len):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.checkpoint, num_labels=5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.run = run
        self.cfg = cfg
        self.total_steps = train_len // (self.cfg.batch_size *
                                         max(1, self.cfg.gpus)) // self.cfg.accumulation_steps * self.cfg.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        if self.cfg.lr.type == "fixed":
            return optimizer
        elif self.cfg.lr.type == "step":
            scheduler = lr_scheduler.StepLR(
                optimizer, self.total_steps, gamma=self.cfg.lr.gamma)
        elif self.cfg.lr.type == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.cfg.lr.warmup_steps, num_training_steps=self.total_steps)
        return [optimizer], [scheduler]

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
        self.run['train/loss'].log(loss.item() / self.cfg.batch_size)
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

    def test_step(self, batch, batch_idx):
        results = self.validation_step(batch, batch_idx)
        return results

    def test_epoch_end(self, outputs):
        # Evaluating epoch averages
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Neptune logging
        self.run['test_epoch/loss'].log(loss.item())
        self.run['test_epoch/acc'].log(self.calc_acc(preds, targets))

    def calc_acc(self, preds, targets):
        return 100 * (preds == targets).float().mean()
