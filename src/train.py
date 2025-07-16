# -*- coding: utf-8 -*-
"""
Uruchomienie treningu DC-GAN z Hydra + W&B
"""
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from lit_dcgan import DCGANLit
from lit_ganformer import GANformerLit
from datamodule import HistologyDataModule  # <-- zakładam, że istnieje tak jak wcześniej

pl.seed_everything(42, workers=True)


@hydra.main(config_path="../configs", config_name="dcgan")
def train(cfg: DictConfig):
    # ---------- dane ---------- #
    dm = HistologyDataModule(**cfg.data)

    # ---------- model ---------- #
    if cfg.model_type == "dcgan":
        model = DCGANLit(**cfg.model)
        logger_name = "dcgan_128"
    elif cfg.model_type == "ganformer":
        model = GANformerLit(**cfg.model)
        logger_name = "ganformer_16"
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    # ---------- logger ---------- #
    wandb_logger = WandbLogger(project="gans-histopathology", name=logger_name, mode="offline")

    # ---------- trainer ---------- #
    wandb_logger = WandbLogger(project="gans-histopathology", name=logger_name)

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