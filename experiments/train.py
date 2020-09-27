""" Generic training script used by all experiments.

Added functionality must always be backwards compatible.

Typical use:
    # where "my_experiment_name" is the name of the folder in `experiment/`
    python3 train.py my_experiment_name [HYDRA CLI ARGS]
"""

import sys
import logging
from pathlib import Path

from pugh_torch.helpers import working_dir
from pugh_torch.utils import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
import hydra

log = logging.getLogger(__name__)

_usage = """Pugh Torch Trainer

Before running, copy the `template/` folder to get your experiment going.

Usage:
    train.py <experiment_name> [hydra-overrides]
"""

# Hack argv so that we have stronger control over hydra invocation
# We use the first positional argument to dictate the experiment
# folder
try:
    experiment_name = sys.argv.pop(1)
except IndexError:
    print(_usage)
    sys.exit(1)

experiment_path = Path(experiment_name).resolve()
if not experiment_path.is_dir():
    raise FileNotFoundError(f"No such directory {experiment_path}")

sys.path.insert(0, str(experiment_path))


# We switch the working directory so that hydra's automatic path handling
# happens within the experiment directory.
with working_dir(experiment_path):
    @hydra.main(config_path="config", config_name="config")
    def train(cfg):
        log.info(f"Training experiment {experiment_name} with the following config:\n{OmegaConf.to_yaml(cfg)}")

        pl.seed_everything(cfg.seed)

        # Instantiate Model to Train
        exec(f"from model import {cfg.model.name} as Model", globals())
        model = Model(cfg=cfg, **cfg.model)

        # Configure callbacks from the model
        try:
            callbacks = model.configure_callbacks()
        except AttributeError:
            log.info("No additional callbacks registered.")
            callbacks = [] 
        else:
            log.info(f"Registering callbacks:")
            for callback in callbacks:
                log.info(f"    {str(callback)}")

        trainer = Trainer(
                logger=TensorBoardLogger("."),
                callbacks=callbacks,
                **cfg.trainer)
        trainer.fit(model)

    train()
