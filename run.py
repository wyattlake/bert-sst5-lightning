from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from data import SSTDataset
from datasets import load_dataset
from model import BertSST
import torch
import hydra
from dotenv import load_dotenv
import neptune.new as neptune
import os


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    load_dotenv()

    # Cuda setup
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Random seeding
    seed_everything(cfg.seed, workers=True)

    # Dataset loading
    full_dataset = load_dataset("sst", "default")
    train_dataset = SSTDataset(cfg, full_dataset, device, split="train")
    eval_dataset = SSTDataset(cfg, full_dataset, device, split="validation")
    test_dataset = SSTDataset(cfg, full_dataset, device, split="test")

    NEPTUNE_TOKEN = os.environ.get("NEPTUNE_TOKEN")
    run = neptune.init(
        project='wyattlake/bert-sst5-lightning',
        api_token=NEPTUNE_TOKEN,
    )

    bert_sst5 = BertSST(cfg, run, len(train_dataset))

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False)

    trainer = Trainer(deterministic=True,
                      accumulate_grad_batches=cfg.accumulation_steps, max_epochs=cfg.max_epochs, gpus=cfg.gpus)

    trainer.fit(bert_sst5, train_loader, eval_loader)
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
