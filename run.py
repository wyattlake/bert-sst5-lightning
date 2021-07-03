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

    # Neptune initialization
    if cfg.logging:
        NEPTUNE_TOKEN = os.environ.get("NEPTUNE_TOKEN")
        run = neptune.init(
            project='wyattlake/bert-sst5-lightning',
            api_token=NEPTUNE_TOKEN,
        )
    else:
        run = None

    if cfg.data.generate_explanations:
        teacher = BertSST(cfg, None, len(train_dataset), False, device, False)
        teacher.load_state_dict(torch.load(cfg.data.teacher_path))

        # Generating explanations for each dataset
        train_dataset.generate_explanations(
            teacher.model, cfg.data.train_csv_path)
        eval_dataset.generate_explanations(
            teacher.model, cfg.data.eval_csv_path)
        test_dataset.generate_explanations(
            teacher.model, cfg.data.test_csv_path)

    if cfg.data.load_explanations:
        # Loading explanations for each dataset
        train_dataset.load_explanations(cfg.data.train_dataset_path)
        eval_dataset.load_explanations(cfg.data.eval_dataset_path)
        test_dataset.load_explanations(cfg.data.test_dataset_path)

    if cfg.run.train_model:
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True)
        eval_loader = DataLoader(
            eval_dataset, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False)

        trainer = Trainer(deterministic=True,
                          accumulate_grad_batches=cfg.accumulation_steps, max_epochs=cfg.max_epochs, gpus=cfg.gpus)

        bert_sst5 = BertSST(cfg, run, len(train_dataset),
                            cfg.explanation_regularization, device, cfg.logging)

        trainer.fit(bert_sst5, train_loader, eval_loader)

        if cfg.run.test_model:
            trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
