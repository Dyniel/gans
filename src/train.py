import hydra
from omegaconf import DictConfig
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datamodule import HistologyDataModule
from callbacks.metrics import HistologyMetrics

@hydra.main(config_path="../configs", config_name="dcgan")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri(f"file://{hydra.utils.get_original_cwd()}/logs")
    mlflow.set_experiment(cfg.model_name)
    with mlflow.start_run():
        mlflow.log_params(cfg)

        # Data
        dm = HistologyDataModule(
            data_dir   = cfg.data_dir,
            batch_size = cfg.batch_size,
            img_size   = cfg.img_size
        )
        dm.setup()

        # Model selection
        if cfg.model_name == 'dcgan':
            from lit_modules.dcgan import DCGANLitModule as Model
        elif cfg.model_name == 'wgan_gp':
            from lit_modules.wgan_gp import WGANLitModule as Model
        else:
            raise ValueError(f"Unknown model: {cfg.model_name}")
        model = Model(cfg)

        # Callbacks
        ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.model_name}",
            monitor="val_fid",
            mode="min",
            save_top_k=1,
            filename="{epoch}-{val_fid:.2f}"
        )
        es = EarlyStopping(
            monitor="val_fid",
            patience=20,
            mode="min"
        )

        # Logger
        wandb_logger = WandbLogger(
            project="gans-histopathology",
            name=cfg.model_name
        )

        # Trainer (skip validation sanity runs)
        trainer = pl.Trainer(
            max_epochs=100,
            gpus=1,                         # or accelerator='gpu', devices=1 in PL 1.7+
            num_sanity_val_steps=0,         # <— replaces limit_sanity_steps
            callbacks=[ckpt, es, HistologyMetrics()],
            logger=wandb_logger
        )

        trainer.fit(model, dm)

if __name__ == "__main__":
    train()