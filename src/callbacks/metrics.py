import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class HistologyMetrics(pl.Callback):
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

        # build & freeze Inception-v3
        if feature_extractor == 'inception_v3':
            feature_extractor = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                progress=True
            )
            feature_extractor.fc = nn.Identity()

        feature_extractor.eval()               # BatchNorm uses running stats
        for p in feature_extractor.parameters():
            p.requires_grad = False

        self.fid = FrechetInceptionDistance(
            feature=feature_extractor,
            normalize=True
        )
        self.kid = KernelInceptionDistance(
            feature=feature_extractor,
            subset_size=kid_subsets,
            normalize=True
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        gen_imgs = pl_module.generated_imgs
        real_imgs, _ = next(iter(trainer.datamodule.train_dataloader()))

        device = gen_imgs.device
        self.fid.to(device)
        self.kid.to(device)

        if self.use_frechet_histology_distance:
            return

        self.fid.update(real_imgs, real=True)
        self.fid.update(gen_imgs,  real=False)
        self.kid.update(real_imgs, real=True)
        self.kid.update(gen_imgs,  real=False)

        fid_score = self.fid.compute()
        kid_mean, kid_std = self.kid.compute()

        pl_module.log('val_fid',      fid_score, prog_bar=True)
        pl_module.log('val_kid_mean', kid_mean)
        pl_module.log('val_kid_std',  kid_std)

        if trainer.logger:
            trainer.logger.experiment.log({
                "generated_images": [wandb.Image(img) for img in gen_imgs]
            })

        self.fid.reset()
        self.kid.reset()