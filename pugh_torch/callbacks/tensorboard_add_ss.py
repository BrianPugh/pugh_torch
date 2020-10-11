from .tensorboard_base import TensorBoardCallback


class TensorBoardAddSS(TensorBoardCallback):
    """Adds the rgb, ground truth segmentation, and the network prediction
    to tensorboard.
    """

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.logging_batch_interval != 0:
            return

        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        # pick the last batch and labels
        x, y = batch[:2]

        trainer.logger.experiment.add_ss(
            "train/output",
            x,
            pl_module.last_logits,
            y,
            global_step=trainer.global_step,
            **self.logging_kwargs
        )
