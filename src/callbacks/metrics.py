import pytorch_lightning as pl
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class HistologyMetrics(pl.Callback):
    def __init__(self, feature_extractor='inception-v3-compat', kid_subsets=50, use_frechet_histology_distance=False):
        super().__init__()
        if use_frechet_histology_distance:
            # Placeholder for Frechet Histology Distance
            # self.fhd = ...
            pass
        else:
            self.fid = FrechetInceptionDistance(feature=feature_extractor)
            self.kid = KernelInceptionDistance(feature=feature_extractor, subset_size=kid_subsets)
        self.use_frechet_histology_distance = use_frechet_histology_distance

    def on_validation_epoch_end(self, trainer, pl_module):
        # The generated images are on the pl_module
        generated_images = pl_module.generated_imgs
        # The real images are in the dataloader
        real_images, _ = next(iter(trainer.datamodule.train_dataloader()))

        if self.use_frechet_histology_distance:
            # fhd_score = self.fhd.compute(...)
            # pl_module.log('val_fhd', fhd_score)
            pass
        else:
            self.fid.update(real_images, real=True)
            self.fid.update(generated_images, real=False)
            self.kid.update(real_images, real=True)
            self.kid.update(generated_images, real=False)

            fid_score = self.fid.compute()
            kid_mean, kid_std = self.kid.compute()

            pl_module.log('val_fid', fid_score)
            pl_module.log('val_kid_mean', kid_mean)
            pl_module.log('val_kid_std', kid_std)

            # Log generated images to wandb
            if trainer.logger:
                trainer.logger.experiment.log({
                    "generated_images": [wandb.Image(img) for img in generated_images]
                })


            # Reset metrics
            self.fid.reset()
            self.kid.reset()
