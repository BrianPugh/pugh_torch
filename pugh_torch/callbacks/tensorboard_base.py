import torch
from pytorch_lightning import Callback
from torch import nn

from ..utils import TensorBoardLogger

class TensorBoardCallback(Callback):
    """ Base class for pytorch-lightning callbacks.

    Can only be used with ``pugh_torch.utils.TensorBoardLogger``
    """

    def __init__(self, *, logging_batch_interval=20,  **kwargs):
        """
        Parameters
        ----------
        logging_batch_interval : int
            Log image(s) every this many batches.
        kwargs : dict
            Passed along to ``SummaryWriter`` method on hook
        """

        self.logging_batch_interval = logging_batch_interval
        self.logging_kwargs = kwargs

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """  Just some validation checks.
        """

        assert isinstance(trainer.logger, TensorBoardLogger), "Can only be used with pugh_torch.utils.TensorBoardLogger"

        if not hasattr(pl_module, "last_logits"):
            m = """please track the last_logits in the training_step like so:
                def training_step(...):
                    self.last_logits = your_logits
            """
            raise AttributeError(m)


