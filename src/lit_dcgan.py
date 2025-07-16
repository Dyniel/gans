# -*- coding: utf-8 -*-
"""
Lightning-wrapper DC-GAN + EMA + FID/KID + W&B wizualizacje
"""

import torch, torch.nn as nn, torch.optim as optim, torchvision
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import wandb
from contextlib import nullcontext
import torch.cuda.amp as amp

from dcgan_modules import Generator, Discriminator


class DCGANLit(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 128,
        base_channels: int = 64,
        lr: float = 2e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- sieci --- #
        self.G = Generator(img_size, latent_dim, base_channels)
        self.D = Discriminator(img_size, base_channels)
        self.ema = AveragedModel(self.G, avg_fn=lambda a, b, _: a * 0.999 + b * 0.001)

        # --- straty --- #
        self.bce = nn.BCEWithLogitsLoss()

        # --- metryki --- #
        self.fid = FrechetInceptionDistance(feature=64, normalize=True)
        # mały subset_size ⇒ brak błędu przy małej walidacji
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True, subsets=50)

    # --------------------------------------------------------- #
    # Forward = generowanie
    def forward(self, z):
        return self.G(z)

    # --------------------------------------------------------- #
    # TRAINING
    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch  # [B,3,H,W] ∈ (-1,1)
        b = real.size(0)
        z = torch.randn(b, self.hparams.latent_dim, 1, 1, device=self.device)
        fake = self.G(z)

        # -------- G --------
        if optimizer_idx == 0:
            pred_f = self.D(fake)
            g_loss = self.bce(pred_f, torch.ones_like(pred_f))
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # -------- D --------
        pred_r = self.D(real)
        pred_f = self.D(fake.detach())
        loss_r = self.bce(pred_r, torch.ones_like(pred_r))
        loss_f = self.bce(pred_f, torch.zeros_like(pred_f))
        d_loss = 0.5 * (loss_r + loss_f)
        self.log("d_loss", d_loss, prog_bar=True)
        return d_loss

    def on_train_batch_end(self, *_):
        self.ema.update_parameters(self.G)

    # --------------------------------------------------------- #
    # VALIDATION
    @staticmethod
    def _to_uint8(img_float: torch.Tensor) -> torch.Tensor:
        """ [-1,1] → uint8  """
        img = (img_float * 0.5 + 0.5).clamp(0, 1) * 255.0
        return img.round().to(torch.uint8)

    def validation_step(self, batch, batch_idx):
        real = batch
        b = real.size(0)
        z = torch.randn(b, self.hparams.latent_dim, 1, 1, device=self.device)
        fake = self.G(z)

        # --- uint8 & wyłączony autocast dla metryk --- #
        real_u8 = self._to_uint8(real)
        fake_u8 = self._to_uint8(fake)

        amp_off = amp.autocast(False) if amp.is_autocast_enabled() else nullcontext()
        with amp_off:
            self.fid.update(real_u8, real=True)
            self.fid.update(fake_u8, real=False)
            self.kid.update(real_u8, real=True)
            self.kid.update(fake_u8, real=False)

        # --- W&B – raz na epokę (batch_idx == 0) --- #
        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            grid_r = torchvision.utils.make_grid(real_u8[:9], nrow=3)
            grid_f = torchvision.utils.make_grid(fake_u8[:9], nrow=3)
            self.logger.log_image("real", [grid_r])
            self.logger.log_image("fake", [grid_f])

    def on_validation_epoch_end(self):
        # obliczenia poza autocast
        with amp.autocast(False):
            fid_val = self.fid.compute()
            kid_mean, kid_std = self.kid.compute()
        self.log_dict(
            {
                "val_fid": fid_val,
                "val_kid_mean": kid_mean,
                "val_kid_std": kid_std,
                "epoch": self.current_epoch,
                "step": self.global_step,
            },
            prog_bar=True,
        )
        self.fid.reset(); self.kid.reset()

    # --------------------------------------------------------- #
    # OPTIMIZERS
    def configure_optimizers(self):
        opt_g = optim.Adam(self.G.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.D.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        sch_g = StepLR(opt_g, step_size=30, gamma=0.5)
        sch_d = StepLR(opt_d, step_size=30, gamma=0.5)
        return [opt_g, opt_d], [sch_g, sch_d]