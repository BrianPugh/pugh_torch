""" Extends pytorch-lightning's LightningModule for some 
quality of life improvements.
"""

import torch
import pytorch_lightning as pl
import logging
import hashlib
from .load_state_dict_mixin import LoadStateDictMixin
from ..optimizers import get_optimizer
from torch.optim.lr_scheduler import LambdaLR

log = logging.getLogger(__name__)


class LightningModule(LoadStateDictMixin, pl.LightningModule):
    def configure_optimizers(self):
        """Pretty good defaults, can be easily overrided"""
        optimizers = []
        schedulers = []

        optimizers.append(
            get_optimizer(getattr(self, "optimizer", "adamw"))(
                self.parameters(),
                lr=self.learning_rate,
                **getattr(self, "optimizer_kwargs", {}),
            ),
        )

        # TODO: do similar thing as optimizers
        schedulers.append(
            LambdaLR(
                optimizers[0], lambda epoch: 1
            )  # simple identity for demonstration
        )

        log.info(
            f"Using default pugh_torch optimizers {optimizers} and schedulers {schedulers}"
        )

        return optimizers, schedulers
