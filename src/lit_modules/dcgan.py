import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(latent_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8,  4, 2, 1),
            self._block(features_g * 8,  features_g * 4,  4, 2, 1),
            self._block(features_g * 4,  features_g * 2,  4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, features_d,    4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d,   features_d*2,   4, 2, 1)),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d*2, features_d*4,   4, 2, 1)),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d*4, features_d*8,   4, 2, 1)),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d*8, 1,              4, 2, 1)),
            # no Sigmoid
        )

    def forward(self, x):
        return self.net(x)


class DCGANLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.generator     = Generator(cfg.latent_dim, 3, 64)
        self.discriminator = Discriminator(3, 64)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch.to(self.device)
        B    = real.size(0)
        z    = torch.randn(B, self.hparams.latent_dim, 1, 1, device=self.device)
        fake = self(z)

        # Generator (hinge loss)
        if optimizer_idx == 0:
            pred_fake = self.discriminator(fake)
            g_loss = -pred_fake.mean()
            self.log('g_loss', g_loss, prog_bar=True)
            return g_loss

        # Discriminator (hinge loss)
        pred_real = self.discriminator(real)
        pred_fake = self.discriminator(fake.detach())
        loss_real = F.relu(1.0 - pred_real).mean()
        loss_fake = F.relu(1.0 + pred_fake).mean()
        d_loss = 0.5 * (loss_real + loss_fake)
        self.log('d_loss', d_loss, prog_bar=True)
        return d_loss

    def validation_step(self, batch, batch_idx):
        B = batch.size(0)
        z = torch.randn(B, self.hparams.latent_dim, 1, 1, device=self.device)
        self.generated_imgs = self(z)

    def configure_optimizers(self):
        lr   = self.hparams.lr
        g_opt = optim.Adam(self.generator.parameters(),     lr=lr,     betas=(0.0, 0.99))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=lr*0.1, betas=(0.0, 0.99))
        return [g_opt, d_opt], []