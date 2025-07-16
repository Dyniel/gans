import hydra, torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.datamodule import HistologyDataModule
from src.lit_modules.dcgan import DCGANLit

@hydra.main(config_path="../configs", config_name="dcgan")
def train(cfg: DictConfig):
    pl.seed_everything(42, workers=True)

    dm = HistologyDataModule(
        data_dir=cfg.data_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = DCGANLit(
        img_size=cfg.img_size,
        latent_dim=cfg.latent_dim,
        base_channels=cfg.base_channels,
        lr=cfg.lr,
    )

    logger = WandbLogger(project="gans-histopathology", name=f"dcgan_{cfg.img_size}")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu",
        devices=1,
        precision=16,
        logger=logger,
        log_every_n_steps=50,
        enable_checkpointing=False,
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    train()
