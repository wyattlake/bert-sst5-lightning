from data import SSTDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import BertSST
import torch
import hydra


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # Cuda setup
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Dataset loading
    full_dataset = load_dataset("sst", "default")
    train_dataset = SSTDataset(cfg, full_dataset, device, split="train")
    eval_dataset = SSTDataset(full_dataset, device, split="validation")

    bert_sst5 = BertSST(cfg)

    # Data loaders
    train_loader = DataLoader(train_dataset)
    eval_loader = DataLoader(eval_dataset)

    trainer = pl.Trainer()
    trainer.fit(bert_sst5, train_loader, eval_loader)


if __name__ == "__main__":
    main()
