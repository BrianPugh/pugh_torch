""" Extends pytorch-lightning's LightningModule for some 
quality of life improvements.
"""

import torch
import pytorch_lightning as pl
import logging
import hashlib
from .load_state_dict_mixin import LoadStateDictMixin

log = logging.getLogger(__name__)


class LightningModule(LoadStateDictMixin, pl.LightningModule):
    def configure_optimizers(self):
        """Pretty good defaults, can be easily overrided"""

        optimizers = []
        schedulers = []

        optimizers.append(torch.optim.AdamW(self.parameters(), lr=self.learning_rate))
        schedulers.append(
            LambdaLR(
                optimizers[0], lambda epoch: 1
            )  # simple identity for demonstration
        )

        log.info(
            f"Using default pugh_torch optimizers {optimizers} and schedulers {schedulers}"
        )

        return optimizers, schedulers
