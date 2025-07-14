import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class HistologyMetrics(pl.Callback):
    """
    PyTorch-Lightning callback to compute FID and KID on validation images.
    """

    def __init__(
        self,
        feature_extractor: str = 'inception_v3',
        kid_subsets: int = 32,
        use_frechet_histology_distance: bool = False
    ):
        super().__init__()
        self.use_frechet_histology_distance = use_frechet_histology_distance
        self.kid_subsets = kid_subsets

        if use_frechet_histology_distance:
            # TODO: implement Frechet Histology Distance
            self.fhd = None
            return

        # build & freeze Inception-v3
        if feature_extractor == 'inception_v3':
            feature_extractor = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                progress=True
            )
            feature_extractor.fc = nn.Identity()

        # ensure BatchNorm uses running stats
        feature_extractor.eval()
        for p in feature_extractor.parameters():
            p.requires_grad = False

        # instantiate metrics with initial subset size
        self.fid = FrechetInceptionDistance(
            feature=feature_extractor,
            normalize=True
        )
        self.kid = KernelInceptionDistance(
            feature=feature_extractor,
            subset_size=self.kid_subsets,
            normalize=True
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # generated images from the validation step
        gen_imgs = pl_module.generated_imgs  # [B, C, H, W]

        # real images from validation dataloader
        real_imgs = next(iter(trainer.datamodule.val_dataloader()))  # [B, C, H, W]

        # move data & metrics to device
        device = gen_imgs.device
        gen_imgs = gen_imgs.to(device)
        real_imgs = real_imgs.to(device)
        self.fid.to(device)
        self.kid.to(device)

        # resize to Inception expected input size
        gen_resized = F.interpolate(gen_imgs, size=(299, 299), mode='bilinear', align_corners=False)
        real_resized = F.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)

        if self.use_frechet_histology_distance:
            return

        # update metrics
        self.fid.update(real_resized, real=True)
        self.fid.update(gen_resized,  real=False)
        self.kid.update(real_resized, real=True)
        self.kid.update(gen_resized,  real=False)

        # ensure subset_size < number of samples
        n_samples = gen_resized.shape[0]
        max_subset = max(1, n_samples - 1)
        if self.kid.subset_size > max_subset:
            self.kid.subset_size = max_subset

        # compute FID & KID
        fid_score = self.fid.compute()
        kid_mean, kid_std = self.kid.compute()

        # log metrics
        pl_module.log('val_fid',      fid_score,      prog_bar=True)
        pl_module.log('val_kid_mean', kid_mean)
        pl_module.log('val_kid_std',  kid_std)

        # optionally log generated images to W&B
        if trainer.logger is not None:
            trainer.logger.experiment.log({
                "generated_images": [wandb.Image(img) for img in gen_resized]
            })

        # reset for next epoch
        self.fid.reset()
        self.kid.reset()