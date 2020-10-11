""" Generic training script used by all experiments.

The model to be trained is specified via ``cfg.model.name``.
If this is a classname, it is assumed to be in ``model.py``:
    # name: MyModel
    # will be interpretted as
    from model import MyModel as Model
If this contains ".", it will be interpretted as a different location:
    # name: foo.bar.MyModel
    # will be interpretted as
    from foo.bar import MyModel as Model
    # Subsequently, the following is the same as default:
    # name: model.MyModel
    from model import MyModel as Model

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
    # experiment's ``outputs/`` directory.
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

    # Fine tune from a checkpoint
    python3 train.py my_experiment_name +fine_tune=path/to/weights.ckpt
"""

import sys
import subprocess
import logging
from pathlib import Path
import importlib

import pugh_torch as pt
from pugh_torch.helpers import working_dir, most_recent_checkpoint, plot_to_np
from pugh_torch.utils import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import LearningRateMonitor

from omegaconf import OmegaConf
import hydra

import matplotlib

# Determine which matplotlib backend to use
# matplotlib.pyplot is used when generating the auto_lr_find graph
try:
    subprocess.check_output(
        ["python", "-c", "import matplotlib.pyplot as plt; plt.figure()"],
        stderr=subprocess.DEVNULL,
    )
except subprocess.CalledProcessError:
    # No display is available
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        # if cfg.trainer.fast_dev_run:
        #    cfg.trainer.auto_lr_find = False

        model_kwargs = {
            "cfg": cfg,
            **cfg.model,
        }
        model_kwargs.pop("name")  # This is just for experiment management

        if "." in cfg.model.name:
            # When the model name has "." in it, assume its from a different module.
            tokens = cfg.model.name.split(".")
            Model = getattr(importlib.import_module(".".join(tokens[:-1]), tokens[-1]))
        else:
            # When just the ClassName is given, assume its from model.py
            Model = getattr(importlib.import_module("model"), cfg.model.name)

        trainer_kwargs = {
            "logger": TensorBoardLogger("."),
            "checkpoint_callback": pt.callbacks.ModelCheckpoint(
                save_last=True, monitor="val_loss"
            ),
            **cfg.trainer,
        }
        # We customize the lr finder a bit after trainer creation
        auto_lr_find = trainer_kwargs.pop("auto_lr_find", False)

        exclusive_flags = ["resume_last_run", "resume_checkpoint"]
        assert 1 >= sum(
            [key in cfg for key in exclusive_flags]
        ), f"Only 1 of {str(exclusive_flags)} can be specified"

        #############################
        # Resume/Fine-Tune Handling #
        #############################
        # Determine the loading of any states/weights
        outputs_path = Path(
            "./../.."
        ).resolve()  # Gets the experiment's outputs directory

        if cfg.get("resume_last_run", False):
            # trainer_kwargs['resume_from_checkpoint'] = str(most_recent_run(outputs_path) / 'default/version_0/checkpoints/last.ckpt')
            trainer_kwargs["resume_from_checkpoint"] = most_recent_checkpoint(
                outputs_path
            )
        elif cfg.get("fine_tune", None):
            # Restore weights directly via the model, not via the Trainer
            Model = Model.load_from_checkpoint  # Change the default constructor
            trainer_kwargs.pop("resume_from_checkpoint", None)
            checkpoint_path = outputs_path / cfg.fine_tune
            model_kwargs["checkpoint_path"] = str(checkpoint_path)
            model_kwargs["strict"] = False

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f'Fine tune checkpoint "{str(checkpoint_path)}" does not exist'
                )
            log.info(f"Fine tuning from weights {str(checkpoint_path)}")

        elif cfg.get("resume_checkpoint", None):
            # This handles absolute paths fine
            trainer_kwargs["resume_from_checkpoint"] = (
                outputs_path / cfg.resume_checkpoint
            )

        if "resume_from_checkpoint" in trainer_kwargs:
            if not trainer_kwargs["resume_from_checkpoint"].is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint \"{str(trainer_kwargs['resume_from_checkpoint'].resolve())}\" does not exist"
                )
            trainer_kwargs["resume_from_checkpoint"] = str(
                trainer_kwargs["resume_from_checkpoint"]
            )
            log.info(
                f"Resuming training from \"{trainer_kwargs['resume_from_checkpoint']}\""
            )

        #####################
        # Instantiate Model #
        #####################
        log.info(f'Instantiating model "{cfg.model.name}"')
        model = Model(**model_kwargs)

        ##############################################
        # Configure Trainer callbacks from the model #
        ##############################################
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

        #######################
        # Instantiate Trainer #
        #######################
        trainer = Trainer(**trainer_kwargs)

        ##################
        # Auto LR Finder #
        ##################
        if auto_lr_find and not trainer_kwargs.get("resume_from_checkpoint"):
            # automatically find the learning rate if enabled and if we are
            # not resuming training from a checkpoint (NOT fine tune-tuning).
            lr_finder = trainer.tuner.lr_find(
                model, min_lr=1e-6, max_lr=1e-2, early_stop_threshold=5
            )

            new_lr = lr_finder.suggestion()
            log.info(
                f"Updating existing learning rate {model.learning_rate} to the suggested learning rate {new_lr}"
            )

            # Add the finder plot image to tensorboard
            lr_finder_fig = lr_finder.plot(suggest=True)

            trainer.logger.experiment.add_figure("auto_lr_finder", lr_finder_fig, 0)

            # Actually update the model
            model.learning_rate = new_lr

        #########################
        # Begin/Resume training #
        #########################
        trainer.fit(model)

    train()
