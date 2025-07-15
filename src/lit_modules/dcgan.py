import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(latent_dim,        features_g * 16, 4, 1, 0),
            self._block(features_g * 16,   features_g * 8,  4, 2, 1),
            self._block(features_g * 8,    features_g * 4,  4, 2, 1),
            self._block(features_g * 4,    features_g * 2,  4, 2, 1),
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
            nn.Conv2d(img_channels, features_d,    4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(features_d,   features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),

            nn.Conv2d(features_d*8, 1, 4, 2, 1),
            # no Sigmoid here!
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DCGANLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Models
        self.generator     = Generator(self.hparams.latent_dim, 3, 64)
        self.discriminator = Discriminator(3, 64)

        # Use BCEWithLogits (stronger gradients) and label smoothing
        self.criterion = nn.BCEWithLogitsLoss()
        self.generated_imgs = None

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch.to(self.device)
        B = real.size(0)

        # -----------------
        # 1) Generator step
        # -----------------
        if optimizer_idx == 0:
            z     = torch.randn(B, self.hparams.latent_dim, 1, 1, device=self.device)
            fake  = self(z)
            pred  = self.discriminator(fake)
            # smooth the real label for G
            valid = torch.empty_like(pred).uniform_(0.8, 1.0)
            g_loss = self.criterion(pred, valid)
            self.log('g_loss', g_loss, prog_bar=True)
            return g_loss

        # ---------------------
        # 2) Discriminator step
        # ---------------------
        if optimizer_idx == 1:
            # real loss (smoothed positives)
            pred_real = self.discriminator(real)
            valid     = torch.empty_like(pred_real).uniform_(0.8, 1.0)
            loss_real = self.criterion(pred_real, valid)

            # fake loss (noisy negatives)
            z         = torch.randn(B, self.hparams.latent_dim, 1, 1, device=self.device)
            fake      = self(z).detach()
            pred_fake = self.discriminator(fake)
            fake_lbl  = torch.empty_like(pred_fake).uniform_(0.0, 0.2)
            loss_fake = self.criterion(pred_fake, fake_lbl)

            d_loss = (loss_real + loss_fake) * 0.5
            self.log('d_loss', d_loss, prog_bar=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        # generate a batch for FID/KID
        B = batch.size(0)
        z = torch.randn(B, self.hparams.latent_dim, 1, 1, device=self.device)
        self.generated_imgs = self(z)

    def configure_optimizers(self):
        lr   = self.hparams.lr
        b1,b2 = 0.5, 0.999
        # generator learns a bit faster, discriminator half learning rate
        opt_g = optim.Adam(self.generator.parameters(),     lr=lr,    betas=(b1, b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr*0.5,betas=(b1, b2))
        return [opt_g, opt_d], []