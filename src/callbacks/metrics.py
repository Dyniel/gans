import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class HistologyMetrics(pl.Callback):
    """
    PyTorch-Lightning callback to compute FID and KID on validation images.
    """

    def __init__(
        self,
        feature_extractor: str = 'inception_v3',
        kid_subsets: int = 50,
        use_frechet_histology_distance: bool = False
    ):
        super().__init__()
        self.use_frechet_histology_distance = use_frechet_histology_distance

        if use_frechet_histology_distance:
            # TODO: implement Frechet Histology Distance
            self.fhd = None
            return

        # --- build and freeze an Inception-v3 trunk ---
        if feature_extractor == 'inception_v3':
            feature_extractor = inception_v3(
                pretrained=True,
                aux_logits=False,
                transform_input=False
            )
            # remove final classifier
            feature_extractor.fc = nn.Identity()

        # ensure BatchNorm uses running stats (no batch of size 1 error)
        feature_extractor.eval()
        for p in feature_extractor.parameters():
            p.requires_grad = False

        # instantiate FID & KID metrics
        self.fid = FrechetInceptionDistance(
            feature=feature_extractor,
            normalize=True
        )
        self.kid = KernelInceptionDistance(
            feature=feature_extractor,
            subset_size=kid_subsets,
            normalize=True
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # grab generated & real images
        gen_imgs = pl_module.generated_imgs        # [B, C, H, W], floats in [0,1]
        real_imgs, _ = next(iter(trainer.datamodule.train_dataloader()))

        # move metrics to the same device
        device = gen_imgs.device
        self.fid.to(device)
        self.kid.to(device)

        if self.use_frechet_histology_distance:
            # fhd_score = self.fhd.compute(real=real_imgs, fake=gen_imgs)
            # pl_module.log('val_fhd', fhd_score)
            return

        # update metrics
        self.fid.update(real_imgs, real=True)
        self.fid.update(gen_imgs,  real=False)
        self.kid.update(real_imgs, real=True)
        self.kid.update(gen_imgs,  real=False)

        # compute
        fid_score = self.fid.compute()
        kid_mean, kid_std = self.kid.compute()

        # log to Lightning (and show FID in progress bar)
        pl_module.log('val_fid',      fid_score, prog_bar=True)
        pl_module.log('val_kid_mean', kid_mean)
        pl_module.log('val_kid_std',  kid_std)

        # also log images to Weights & Biases if using WandbLogger
        if trainer.logger is not None:
            trainer.logger.experiment.log({
                "generated_images": [wandb.Image(img) for img in gen_imgs]
            })

        # reset for next epoch
        self.fid.reset()
        self.kid.reset()