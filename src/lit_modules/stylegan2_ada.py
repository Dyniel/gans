import pytorch_lightning as pl
import torch
import dnnlib
from ..stylegan2_ada.training.loss import StyleGAN2Loss
from ..stylegan2_ada.training import training_loop
from ..stylegan2_ada.training.networks import Generator, Discriminator
from ..stylegan2_ada.training.dataset import ImageFolderDataset
from ..stylegan2_ada.training.augment import AugmentPipe

class StyleGAN2ADALitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.G = Generator(z_dim=self.hparams.latent_dim, c_dim=0, w_dim=self.hparams.latent_dim, img_resolution=self.hparams.img_size, img_channels=3)
        self.D = Discriminator(c_dim=0, img_resolution=self.hparams.img_size, img_channels=3)
        self.G_ema = dnnlib.util.copy_module(self.G, requires_grad=False)
        self.loss = StyleGAN2Loss(device=self.device, G_mapping=self.G.mapping, G_synthesis=self.G.synthesis, D=self.D)
        self.augment_pipe = AugmentPipe(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1) if self.hparams.aug == 'bgc' else None
        self.generated_imgs = None

    def forward(self, z):
        return self.G(z, None)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # This is a simplified version of the training loop.
        # The actual training will be done in the `train.py` script.
        pass

    def validation_step(self, batch, batch_idx):
        z = torch.randn(self.hparams.batch_size, self.hparams.latent_dim, device=self.device)
        self.generated_imgs = self.G_ema(z, None)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr, betas=(0.0, 0.99))
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr, betas=(0.0, 0.99))
        return [g_opt, d_opt], []

    def on_train_start(self):
        # The main training loop is handled by the `training_loop.py` script.
        # We just need to call it here.
        # This is not the standard way of using PyTorch Lightning, but it's the easiest way to integrate the StyleGAN2-ADA code.
        training_loop.training_loop(
            run_dir='.',
            training_set_kwargs=dict(class_name='training.dataset.ImageFolderDataset', path=self.trainer.datamodule.data_dir, use_labels=False, max_size=None, xflip=self.hparams.mirror, resolution=self.hparams.img_size),
            data_loader_kwargs=dict(pin_memory=True, num_workers=4, prefetch_factor=2),
            G_kwargs=dict(class_name='training.networks.Generator', z_dim=self.hparams.latent_dim, c_dim=0, w_dim=self.hparams.latent_dim, img_resolution=self.hparams.img_size, img_channels=3),
            D_kwargs=dict(class_name='training.networks.Discriminator', c_dim=0, img_resolution=self.hparams.img_size, img_channels=3),
            G_opt_kwargs=dict(class_name='torch.optim.Adam', lr=self.hparams.lr, betas=[0,0.99], eps=1e-8),
            D_opt_kwargs=dict(class_name='torch.optim.Adam', lr=self.hparams.lr, betas=[0,0.99], eps=1e-8),
            augment_kwargs=dict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1) if self.hparams.aug == 'bgc' else None,
            loss_kwargs=dict(class_name='training.loss.StyleGAN2Loss', r1_gamma=10),
            metrics=[],
            random_seed=0,
            num_gpus=1,
            rank=0,
            batch_size=self.hparams.batch_size,
            batch_gpu=self.hparams.batch_size,
            ema_kimg=10.0,
            ema_rampup=None,
            G_reg_interval=4,
            D_reg_interval=16,
            augment_p=0,
            ada_target=0.6,
            ada_interval=4,
            ada_kimg=500,
            total_kimg=25000,
            kimg_per_tick=4,
            image_snapshot_ticks=50,
            network_snapshot_ticks=50,
            resume_pkl=None,
            cudnn_benchmark=True,
            allow_tf32=False,
            abort_fn=None,
            progress_fn=None,
        )
