from .tensorboard_base import TensorBoardCallback


class TensorBoardAddDepth(TensorBoardCallback):
    """Adds the rgb, ground truth depth, and the network prediction to tensorboard.

    Assumes the network's prediction is in attribute ``last_logits`` with shape
    (b, h, w).
    """

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.global_step % self.logging_batch_interval != 0:
            return

        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        # pick the last batch and labels
        x, y = batch[:2]

        pred_depth = pl_module.last_logits
        if pred_depth.ndim == 4:
            pred_depth = pred_depth[:, 0]

        trainer.logger.experiment.add_depth(
            "train/output",
            x[:, :3],
            pred_depth,
            y,
            global_step=trainer.global_step,
            **self.logging_kwargs
        )
