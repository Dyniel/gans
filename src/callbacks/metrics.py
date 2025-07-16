import torch
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class HistologyMetrics(pl.Callback):
    def __init__(self):
        super().__init__()
        self.fid = FrechetInceptionDistance(feature="unbiased")
        self.kid = KernelInceptionDistance(subset_size=50, feature="unbiased")

    def on_validation_epoch_end(self, trainer, pl_module):
        # real & fake batches saved in module
        real = (pl_module.real_batch * 127.5 + 127.5).clamp(0, 255).to(dtype=torch.uint8)
        fake = (pl_module.fake_batch * 127.5 + 127.5).clamp(0, 255).to(dtype=torch.uint8)
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        self.kid.update(real, real=True)
        self.kid.update(fake, real=False)
        fid = self.fid.compute(); kid_mean, kid_std = self.kid.compute()
        trainer.loggers[0].log_metrics({"val_fid": fid.item(), "val_kid_mean": kid_mean.item()})
        self.fid.reset(); self.kid.reset()
        # log first fake image
        img = (fake[0].permute(1, 2, 0).cpu().numpy()).astype('uint8')
        trainer.loggers[0].experiment.log({"sample": [trainer.loggers[0].experiment.Image(img)]})
