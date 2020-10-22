import torch
import pugh_torch as pt
from pugh_torch.callbacks.tensorboard_base import TensorBoardCallback
from dataset import rasterize_montage


class RasterMontageCallback(TensorBoardCallback):
    """"""

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.logging_batch_interval != 0:
            return

        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        coords, rgb_vals, imgs = batch

        coords = coords[0][None]
        rgb_vals = rgb_vals[0][None]
        imgs = imgs[0][None]
        pred_rgb_vals = pl_module.last_logits[0][None]

        canvas = rasterize_montage(coords, rgb_vals, pred_rgb_vals, imgs)

        trainer.logger.experiment.add_rgb(
            "train/output_gt_pred",
            canvas,
            global_step=trainer.global_step,
            **self.logging_kwargs
        )
