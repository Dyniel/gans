import hydra
from omegaconf import DictConfig
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datamodule import HistologyDataModule
from callbacks.metrics import HistologyMetrics
# Import lit_modules here
# from lit_modules.dcgan import DCGANLitModule
# from lit_modules.wgan_gp import WGANLitModule
# ...

@hydra.main(config_path="../configs", config_name="dcgan")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/logs")
    mlflow.set_experiment(cfg.model_name)
    with mlflow.start_run():
        mlflow.log_params(cfg)

        datamodule = HistologyDataModule(data_dir=cfg.data_dir, batch_size=cfg.batch_size, img_size=cfg.img_size)

        # Initialize the model
        if cfg.model_name == 'dcgan':
            from lit_modules.dcgan import DCGANLitModule
            model = DCGANLitModule(cfg)
        elif cfg.model_name == 'wgan_gp':
            from lit_modules.wgan_gp import WGANLitModule
            model = WGANLitModule(cfg)
        elif cfg.model_name == 'stylegan2_ada':
            from lit_modules.stylegan2_ada import StyleGAN2ADALitModule
            model = StyleGAN2ADALitModule(cfg)
        # ... and so on for other models

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.model_name}",
            filename="{epoch}-{val_fid:.2f}",
            save_top_k=1,
            monitor="val_fid",
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_fid",
            patience=10,
            mode="min"
        )
        metrics_callback = HistologyMetrics()

        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project='gans-histopathology', name=cfg.model_name)


        trainer = pl.Trainer(
            max_epochs=100,
            gpus=1,
            callbacks=[checkpoint_callback, early_stopping_callback, metrics_callback],
            logger=wandb_logger

        )

        trainer.fit(model, datamodule)

if __name__ == "__main__":
    train()
