import pyimport pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import wandb

class HistologyMetrics(pl.Callback):
    def __init__(self, kid_subsets: int = 32):
        super().__init__()
        # load Inception-v3 backbone
        fe = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        fe.fc = torch.nn.Identity()
        fe.eval()
        for p in fe.parameters():
            p.requires_grad = False

        self.fid = FrechetInceptionDistance(
            feature=fe,
            normalize=True
        )
        self.kid = KernelInceptionDistance(
            feature=fe,
            subset_size=kid_subsets,
            normalize=True
        )

        def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # gather real & generated images
        real = next(iter(trainer.datamodule.val_dataloader()))
        fake = pl_module.generated_imgs

        # move to correct device
        device = pl_module.device
        real = real.to(device)
        fake = fake.to(device)

        # resize to 299x299 for Inception
        real = F.interpolate(real, size=(299, 299), mode='bilinear', align_corners=False)
        fake = F.interpolate(fake, size=(299, 299), mode='bilinear', align_corners=False)

        # explicitly move inception models onto GPU
        self.fid.inception = self.fid.inception.to(device)
        self.kid.inception = self.kid.inception.to(device)

        # adjust subset_size if batch is small
        B = fake.size(0)
        if self.kid.subset_size >= B:
            self.kid.subset_size = max(1, B - 1)

        # update metrics
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        self.kid.update(real, real=True)
        self.kid.update(fake, real=False)

        # compute
        fid_val = self.fid.compute()
        kid_m, kid_s = self.kid.compute()

        # log
        pl_module.log('val_fid',      fid_val, prog_bar=True)
        pl_module.log('val_kid_mean', kid_m)
        pl_module.log('val_kid_std',  kid_s)

        # log a few samples to W&B
        if trainer.logger is not None:
            samples = [wandb.Image(img) for img in fake[:4]]
            trainer.logger.experiment.log({'fake_samples': samples})

        # reset for next epoch
        self.fid.reset()
        self.kid.reset()(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # gather real & generated images
        real = next(iter(trainer.datamodule.val_dataloader()))
        fake = pl_module.generated_imgs

        # move to correct device
        device = pl_module.device
        real = real.to(device)
        fake = fake.to(device)

        # resize to 299x299 for Inception
        real = F.interpolate(real, size=(299, 299), mode='bilinear', align_corners=False)
        fake = F.interpolate(fake, size=(299, 299), mode='bilinear', align_corners=False)

        # move the metrics modules to device as well
        self.fid = self.fid.to(device)
        self.kid = self.kid.to(device)

        # adjust subset_size if batch is small
        B = fake.size(0)
        if self.kid.subset_size >= B:
            self.kid.subset_size = max(1, B - 1)

        # update metrics
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        self.kid.update(real, real=True)
        self.kid.update(fake, real=False)

        # compute
        fid_val = self.fid.compute()
        kid_m, kid_s = self.kid.compute()

        # log
        pl_module.log('val_fid',      fid_val, prog_bar=True)
        pl_module.log('val_kid_mean', kid_m)
        pl_module.log('val_kid_std',  kid_s)

        # log a few samples to W&B
        if trainer.logger is not None:
            samples = [wandb.Image(img) for img in fake[:4]]
            trainer.logger.experiment.log({'fake_samples': samples})

        # reset for next epoch
        self.fid.reset()
        self.kid.reset()
