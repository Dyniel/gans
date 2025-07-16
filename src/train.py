# -*- coding: utf-8 -*-
"""
Uruchomienie treningu DC-GAN z Hydra + W&B
"""
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from lit_dcgan import DCGANLit
from datamodule import HistologyDataModule  # <-- zakładam, że istnieje tak jak wcześniej

pl.seed_everything(42, workers=True)


@hydra.main(config_path="../configs", config_name="dcgan")
def train(cfg: DictConfig):
    # ---------- dane ---------- #
    dm = HistologyDataModule(**cfg.datamodule)

    # ---------- model ---------- #
    model = DCGANLit(**cfg.model)

    # ---------- logger ---------- #
    wandb_logger = WandbLogger(project="gans-histopathology", name="dcgan_128")

    # ---------- trainer ---------- #
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="gpu",
        devices=-1,               # wszystkie dostępne karty
        precision=16,             # AMP
        logger=wandb_logger,
        log_every_n_steps=50,
        deterministic=True,
        enable_checkpointing=False,  # uproszczenie
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    train()