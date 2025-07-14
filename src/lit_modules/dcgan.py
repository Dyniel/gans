import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(latent_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

class DCGANLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.generator = Generator(self.hparams.latent_dim, 3, 64)
        self.discriminator = Discriminator(3, 64)
        self.criterion = nn.BCELoss()
        self.generated_imgs = None

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs = batch
        batch_size = real_imgs.size(0)

        # Train Generator
        if optimizer_idx == 0:
            z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, device=self.device)
            fake_imgs = self(z)
            g_loss = self.criterion(self.discriminator(fake_imgs), torch.ones(batch_size, 1, 1, 1, device=self.device))
            self.log('g_loss', g_loss, prog_bar=True)
            return g_loss

        # Train Discriminator
        if optimizer_idx == 1:
            z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, device=self.device)
            fake_imgs = self(z).detach()
            real_loss = self.criterion(self.discriminator(real_imgs), torch.ones(batch_size, 1, 1, 1, device=self.device))
            fake_loss = self.criterion(self.discriminator(fake_imgs), torch.zeros(batch_size, 1, 1, 1, device=self.device))
            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss, prog_bar=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, device=self.device)
        self.generated_imgs = self(z)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = 0.5
        b2 = 0.999

        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
