import math
import torch.nn as nn

def _g_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )

def _d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, True),
    )

class Generator(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, base_ch: int = 64, img_ch: int = 3):
        super().__init__()
        assert img_size & (img_size - 1) == 0 and img_size >= 32, "img_size must be power of 2 ≥ 32"
        n_ups = int(math.log2(img_size)) - 3  # 4→8→...→img_size
        ch = base_ch * (2 ** n_ups)
        layers = [
            nn.ConvTranspose2d(latent_dim, ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
        ]
        for _ in range(n_ups):
            layers += list(_g_block(ch, ch // 2))
            ch //= 2
        layers += [nn.ConvTranspose2d(ch, img_ch, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_size: int, base_ch: int = 64, img_ch: int = 3):
        super().__init__()
        assert img_size & (img_size - 1) == 0 and img_size >= 32, "img_size must be power of 2 ≥ 32"
        n_downs = int(math.log2(img_size)) - 3  # img→...→4
        layers = [nn.Conv2d(img_ch, base_ch, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)]
        ch = base_ch
        for _ in range(n_downs):
            layers += list(_d_block(ch, ch * 2))
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(x.size(0))

import torch, torch.nn as nn, torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel

class DCGANLit(pl.LightningModule):
    def __init__(self, img_size: int, latent_dim: int, base_channels: int = 64, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.G = Generator(img_size, latent_dim, base_channels)
        self.D = Discriminator(img_size, base_channels)
        self.ema = AveragedModel(self.G, avg_fn=lambda a, b, n: a * 0.999 + b * 0.001)
        self.bce = nn.BCEWithLogitsLoss()
        self.example_z = torch.randn(8, latent_dim, 1, 1)

    def forward(self, z):
        return self.G(z)

    # --- training step ---
    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch
        b = real.size(0)
        z = torch.randn(b, self.hparams.latent_dim, 1, 1, device=self.device)
        fake = self(z)

        if optimizer_idx == 0:  # G
            pred_f = self.D(fake)
            loss_g = self.bce(pred_f, torch.ones_like(pred_f))
            self.log("g_loss", loss_g, prog_bar=True, on_step=True)
            return loss_g

        if optimizer_idx == 1:  # D
            pred_r = self.D(real)
            pred_f = self.D(fake.detach())
            loss_r = self.bce(pred_r, torch.ones_like(pred_r))
            loss_f = self.bce(pred_f, torch.zeros_like(pred_f))
            loss_d = (loss_r + loss_f) / 2
            self.log("d_loss", loss_d, prog_bar=True, on_step=True)
            return loss_d

    def on_train_batch_end(self, *_):
        self.ema.update_parameters(self.G)

    # --- validation ---
    def validation_step(self, batch, batch_idx):
        self.fake_batch = self(self.example_z.to(self.device))
        self.real_batch = batch

    # --- optimizers ---
    def configure_optimizers(self):
        opt_g = optim.Adam(self.G.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.D.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        sch_g = StepLR(opt_g, step_size=30, gamma=0.5)
        sch_d = StepLR(opt_d, step_size=30, gamma=0.5)
        return [opt_g, opt_d], [sch_g, sch_d]