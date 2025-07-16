# -*- coding: utf-8 -*-
"""
Uruchomienie treningu DC-GAN z Hydra + W&B
"""
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from lit_dcgan import DCGANLit
from datamodule import HistologyDataModule


@hydra.main(config_path="../configs", config_name="dcgan")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    # ---------- dane ---------- #
    dm = HistologyDataModule(**cfg.datamodule)

    # ---------- model ---------- #
    model = DCGANLit(**cfg.model)

    # ---------- logger ---------- #
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, offline=True)

    # ---------- trainer ---------- #
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        log_every_n_steps=50,
        deterministic=True,
        enable_checkpointing=False,
    )

    trainer.fit(model, dm)
    wandb.finish()


if __name__ == "__main__":
    train()