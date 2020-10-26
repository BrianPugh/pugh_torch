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

class LinearHistogramCallback(TensorBoardCallback):
    """ Relies on model being stored at ``pl_module.model`` and it being
    sequential
    """

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (trainer.global_step) % self.logging_batch_interval != 0:
            return

        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        for i, layer in enumerate(pl_module.model):
            if hasattr(layer, "weight") and layer.weight is not None:
                trainer.logger.experiment.add_histogram(
                        f"linear_weight/{i}",
                        layer.weight,
                        global_step=trainer.global_step,
                        **self.logging_kwargs,
                        )
            if hasattr(layer, "bias") and layer.bias is not None:
                trainer.logger.experiment.add_histogram(
                        f"linear_bias/{i}",
                        layer.bias,
                        global_step=trainer.global_step,
                        **self.logging_kwargs,
                        )
