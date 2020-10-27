import torch
from pugh_torch.callbacks.tensorboard_base import TensorBoardCallback


class Histogram(TensorBoardCallback):
    """Relies on model being stored at ``pl_module.model`` and it being
    sequential
    """

    def __init__(self, *, slash_names=True, names=[], **kwargs):
        """
        Parameters
        ----------
        slash_names : bool
            For each named parameter, replace "." with "/" so that
            they appear under sublabels in TensorBoard.
        names : list of str
            The "dot" named parameters to log. Defaults to all model named parameters.
        """
        super().__init__(**kwargs)

        self.slash_names = slash_names
        self.names = names

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (trainer.global_step) % self.logging_batch_interval != 0:
            return

        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        for name, param in pl_module.named_parameters():
            if self.names and name not in self.names:
                continue

            if self.slash_names:
                name = name.replace(".", "/")

            trainer.logger.experiment.add_histogram(
                name,
                param,
                global_step=trainer.global_step,
                **self.logging_kwargs,
            )
