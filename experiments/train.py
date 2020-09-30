""" Generic training script used by all experiments.

Added functionality must always be backwards compatible.

Typical use:
    # where "my_experiment_name" is the name of the folder in `experiment/`
    python3 train.py my_experiment_name [HYDRA CLI ARGS]

    ################################
    # Resuming and Loading Weights #
    ################################
    # Common trainer overrides are at the root level.
    # Only one (or None!) of these resume configurations should be set.
    #
    # If a provided path is relative, it MUST be relative to the specific 
    # experiment's outputs/ directory.
    #
    # Example:
    #      # structure:
    #      #    pugh_torch/experiments/my_experiment/outputs/2020-09-27/10-43-40/default/version_0/checkpoints
    #
    #      python3 train.py my_experiment +resume_checkpoint=2020-09-27/10-43-40/default/version_0/checkpoints 

    # Load the latest checkpoint from the latest run.
    # Note: this continues that training in a new output directory.
    python3 train.py my_experiment_name +resume_last_run=true

    # Load a specific checkpoint.
    # This checkpoint path is relative
    python3 train.py my_experiment_name +resume_checkpoint=path/to/checkpoint.ckpt
"""

import os
import sys
import logging
from pathlib import Path

import pugh_torch as pt
from pugh_torch.helpers import working_dir, most_recent_checkpoint, plot_to_np
from pugh_torch.utils import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer

try:  # TODO: only import Monitor once pytorch-lightning v0.9.1 is released.
    from pytorch_lightning.callbacks import LearningRateMonitor
except ImportError:
    from pytorch_lightning.callbacks import LearningRateLogger as LearningRateMonitor


from omegaconf import OmegaConf
import hydra

log = logging.getLogger(__name__)

_usage = """Pugh Torch Trainer

Before running, copy the `template/` folder to get your experiment going.

Usage:
    train.py <experiment_name> [hydra-overrides]
"""

this_file_path = Path(__file__).resolve()
this_file_dir = this_file_path.parent

# Hack argv so that we have stronger control over hydra invocation
# We use the first positional argument to dictate the experiment
# folder
try:
    experiment_name = sys.argv.pop(1)
except IndexError:
    print(_usage)
    sys.exit(1)


experiment_name = experiment_name.strip("/")
experiment_path = this_file_dir / experiment_name
if not experiment_path.is_dir():
    raise FileNotFoundError(f"No such directory {experiment_path}")

sys.path.insert(0, str(experiment_path))

# We switch the working directory so that hydra's automatic path handling
# happens within the experiment directory.
with working_dir(experiment_path):

    # TODO: something in hydra's parsing is preventing us from calling
    # train.py from other directories.
    @hydra.main(config_path="config", config_name="config")
    def train(cfg):
        # Note: inside this function, the current working directory is:
        #    my_experiment/outputs/%Y-%m-%d/%H-%M-%S/

        log.info(
            f"Training experiment {experiment_name} with the following config:\n{OmegaConf.to_yaml(cfg)}"
        )

        pl.seed_everything(cfg.seed)

        # Currently auto_lr_rate doesn't work when fast_dev_run=True
        # TODO: Remove this if this gets fixed upstream
        if cfg.trainer.fast_dev_run:
            cfg.trainer.auto_lr_find = False

        model_kwargs = {
            "cfg": cfg,
            **cfg.model,
        }

        trainer_kwargs = {
            "logger": TensorBoardLogger("."),
            "checkpoint_callback": pt.callbacks.ModelCheckpoint(
                save_last=True, monitor="val_loss"
            ),
            **cfg.trainer,
        }

        exclusive_flags = ["resume_last_run", "resume_check"]
        assert 1 >= sum(
            [key in cfg for key in exclusive_flags]
        ), f"Only 1 of {str(exclusive_flags)} can be specified"

        # Determine the loading of any states/weights
        outputs_path = Path(
            "./../.."
        ).resolve()  # Gets the experiment's outputs directory
        if cfg.get("resume_last_run", False):
            # trainer_kwargs['resume_from_checkpoint'] = str(most_recent_run(outputs_path) / 'default/version_0/checkpoints/last.ckpt')
            trainer_kwargs["resume_from_checkpoint"] = most_recent_checkpoint(
                outputs_path
            )
        elif cfg.get("resume_checkpoint", None):
            # This handles absolute paths fine
            trainer_kwargs["resume_from_checkpoint"] = (
                outputs_path / cfg.resume_checkpoint
            )

        if "resume_from_checkpoint" in trainer_kwargs:
            if not trainer_kwargs["resume_from_checkpoint"].is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint \"{str(trainer_kwargs['resume_from_checkpoint'])}\" does not exist"
                )
            trainer_kwargs["resume_from_checkpoint"] = str(
                trainer_kwargs["resume_from_checkpoint"]
            )
            log.info(
                f"Resuming training from \"{trainer_kwargs['resume_from_checkpoint']}\""
            )

        # Instantiate Model to Train
        exec(f"from model import {cfg.model.name} as Model", globals())
        model = Model(**model_kwargs)

        # Configure Trainer callbacks from the model
        try:
            callbacks = model.configure_callbacks()
        except AttributeError:
            log.info("No additional callbacks registered.")
            callbacks = []
        else:
            log.info(f"Registering callbacks:")
            for callback in callbacks:
                log.info(f"    {str(callback)}")
        # Add common callbacks
        callbacks.append(LearningRateMonitor())
        trainer_kwargs["callbacks"] = callbacks

        # We customize the lr finder a bit after trainer creation
        auto_lr_find = trainer_kwargs.pop("auto_lr_find", False)

        # Instantiate Trainer
        trainer = Trainer(**trainer_kwargs)

        if auto_lr_find:
            lr_finder = trainer.lr_find(model, max_lr=0.01)

            new_lr = lr_finder.suggestion()
            log.info(
                f"Updating existing learning rate {model.learning_rate} to the suggested learning rate {new_lr}"
            )

            # Add the finder plot image to tensorboard
            lr_finder_fig = lr_finder.plot(suggest=True)
            trainer.logger.experiment.add_image("auto_lr_finder", lr_finder_fig, 0)

            # Actually update the model
            model.learning_rate = new_lr

        # Begin/Resume training process.
        trainer.fit(model)

    train()
