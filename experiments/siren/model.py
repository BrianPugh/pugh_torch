""" Based on torchvision.models.resnet

This is basically resnet50, just to get you started.
"""

import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import pugh_torch as pt
from pugh_torch.modules import conv3x3, conv1x1

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from scipy.interpolate import griddata
import numpy as np
from math import ceil
from pathlib import Path

log = logging.getLogger(__name__)

this_file_path = Path(__file__).resolve()
this_file_dir = this_file_path.parent


def unnormalize_pos(x, shape):
    """
    Parameters
    ----------
    x : numpy.ndarray
        (N, 2) Array representing (x,y) coordinates in range [-1, 1]
    shape : tuple of length 2
        (H, W) of the image
    """

    x += 1
    x /= 2
    x[:, 0] *= shape[1] - 1
    x[:, 1] *= shape[0] - 1
    x = np.round(x).astype(np.int)
    return x


class SingleImageDataset(torch.utils.data.IterableDataset):
    """Generates sampled:
        * x: (x,y) normalized coordinates in range [-1,1]
        * y: (N,3) color

    We are defining that pixels are determined by their topleft corner.
    """

    def __init__(self, path, batch_size, shape=None, mode="train", normalize=True):
        """
        path : path-like or RGB numpy.ndarray
        """

        super().__init__()

        self.mode = mode.lower()

        if isinstance(path, np.ndarray):
            self.img = path.copy()
        else:
            # Assume that ``path`` is path-like
            self.img = cv2.imread(str(path))[..., ::-1]

        # Resize the image
        if shape is None:
            shape = self.img.shape[:2]
        else:
            assert len(shape) == 2
            self.img = cv2.resize(
                self.img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
            )
        self.shape = np.array(shape)  # (H, W) i.e. (y, x)

        # Normalize the image
        if normalize:
            self.img = pt.transforms.imagenet.np_normalize(self.img / 255)
        self.img = torch.Tensor(self.img)

        self.flat_img = self.img.reshape((-1, 3))
        self.img = self.img[None]
        self.img = self.img.permute(0, 3, 1, 2)  # (B, H, W, C)

        nx, ny = self.img.shape[2], self.img.shape[3]
        # (X, Y)
        meshgrid = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
        self.src_pts = torch.Tensor((meshgrid[0].reshape(-1), meshgrid[1].reshape(-1)))
        self.src_pts_normalized = (
            2 * self.src_pts[0] / (self.shape[1] - 1) - 1,
            2 * self.src_pts[1] / (self.shape[0] - 1) - 1,
        )

        self.batch_size = batch_size
        self.n_pixels = self.shape[0] * self.shape[1]
        self.pos = 0

    def __len__(self):
        return ceil(self.n_pixels / self.batch_size)

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        if self.pos >= self.n_pixels:
            raise StopIteration

        if self.mode == "train":
            # Select random coordinates in range [-1, 1]
            entropy = torch.rand(self.batch_size, 2)  # TODO maybe device
            coord_normalized = 2 * entropy - 1

            # interpolate at those points
            rgb_values = F.grid_sample(
                self.img, coord_normalized[None, :, None, :], align_corners=True
            )
            rgb_values = rgb_values[0, ..., 0].transpose(0, 1)
            self.pos += self.batch_size
        elif self.mode == "val":
            # Deterministcally return every pixel value in order
            # Last batch size may be smaller
            upper = self.pos + self.batch_size
            if upper > self.n_pixels:
                upper = self.n_pixels

            coord_normalized = torch.cat(
                (
                    self.src_pts_normalized[0][self.pos : upper, None],
                    self.src_pts_normalized[1][self.pos : upper, None],
                ),
                dim=1,
            )
            rgb_values = self.flat_img[self.pos : upper]
            self.pos = upper
        else:
            raise NotImplementedError

        return coord_normalized, rgb_values


class SIREN(pt.LightningModule):
    def __init__(
        self,
        *,
        cfg=None,
        layers=[128, 128, 128, 128],
        activation="sine",
        learning_rate=0.002,
        loss="mse_loss",
        optimizer="adamw",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.learning_rate = learning_rate
        self.optimizer = optimizer

        model = []
        model.append(nn.Linear(2, layers[0]))
        model.append(pt.modules.Activation(activation, model[-1]))

        for cur_layer, next_layer in zip(layers, layers[1:]):
            model.append(nn.Linear(cur_layer, next_layer))
            model.append(pt.modules.Activation(activation, model[-1]))

        model.append(nn.Linear(layers[-1], 3))
        # no final activation function

        self.model = nn.Sequential(*model)

        self.loss_fn = pt.losses.get_functional_loss(loss)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, 3, H, W) Input data

        Returns
        -------
        output : torch.Tensor
            (B, C, H, W) Predicted logits
        """
        x = self.model(x)
        return x

    ###########################
    # PyTorch Lightning Stuff #
    ###########################

    def on_train_start(
        self,
    ):
        # log the ground truth image to tensorboard for comparison
        img_path = this_file_dir / self.cfg.dataset.path
        img = cv2.imread(str(img_path))[..., ::-1]
        if self.cfg.dataset.shape:
            shape = self.cfg.dataset.shape
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

        self.logger.experiment.add_image(
            f"ground_truth",
            img,
            dataformats="HWC",
        )

    def _log_common(self, split, logits, target, loss):
        self.log(f"{split}_loss", loss, prog_bar=True)

    def _log_loss(self, split, pred, target):
        # Makes it easier to directly compare techniques that have a different
        # loss function
        loss = F.mse_loss(pred, target)
        self.log(f"{split}_mse_loss", loss, prog_bar=True)

    def _compute_loss(self, pred, target):
        return self.loss_fn(pred, target)

    def training_step(self, batch, batch_nb):
        """"""

        x, y = batch
        logits = self(x)
        self.last_logits = logits  # Commonly used in callbacks
        loss = self._compute_loss(logits, y)

        self._log_common("train", logits, y, loss)
        self._log_loss("train", logits, y)

        return loss

    def on_validation_epoch_start(
        self,
    ):
        # Create an empty image to populate as we iterate over the image
        img_shape = (*self.cfg.dataset.shape, 3)
        self.val_img = np.zeros(img_shape)

    def validation_step(self, batch, batch_nb):
        """"""

        # OPTIONAL
        x, y = batch
        logits = self(x)

        # Populate the predicted image
        x_np = x.cpu().numpy()
        logits_np = logits.cpu().numpy()

        x_np = unnormalize_pos(x_np, self.cfg.dataset.shape)

        self.val_img[x_np[:, 1], x_np[:, 0]] = logits_np

        loss = self._compute_loss(logits, y)

        self._log_common("val", logits, y, loss)
        self._log_loss("val", logits, y)

        return loss

    def on_validation_epoch_end(self):
        # unnormalize and log the image to tensorboard

        self.val_img = pt.transforms.imagenet.np_unnormalize(self.val_img)
        self.logger.experiment.add_image(
            f"val/pred",
            self.val_img,
            global_step=self.trainer.current_epoch,
            dataformats="HWC",
        )

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        optimizers.append(
            pt.optimizers.get_optimizer(self.cfg.model.optimizer)(
                self.parameters(), lr=self.learning_rate
            )
        )
        schedulers.append(
            LambdaLR(
                optimizers[0], lambda epoch: 1
            )  # simple identity for demonstration
        )

        return optimizers, schedulers

    def configure_callbacks(self):
        """Moves trainer callback declaration into the model so the same
        training script can be shared across experiments.

        This is not standard pytorch-lightning

        Returns
        -------
        callbacks : list
            List of callback objects to initialize the Trainer object with.
        """

        callbacks = []
        return callbacks

    def train_dataloader(self):
        dataset = SingleImageDataset(
            this_file_dir / self.cfg.dataset.path,
            batch_size=self.cfg.dataset.batch_size,
            shape=self.cfg.dataset.shape,
            mode="train",
        )

        loader = DataLoader(
            dataset,
            pin_memory=self.cfg.dataset.pin_memory,
            num_workers=self.cfg.dataset.num_workers,
            batch_size=None,  # Disable dataloader batching
            batch_sampler=None,  # Disable dataloader batching
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2 ** 32 - 1)
            ),
        )

        return loader

    def val_dataloader(self):
        dataset = SingleImageDataset(
            this_file_dir / self.cfg.dataset.path,
            batch_size=self.cfg.dataset.batch_size,
            shape=self.cfg.dataset.shape,
            mode="val",
        )

        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=self.cfg.dataset.pin_memory,
            batch_size=None,  # Disable dataloader batching
            batch_sampler=None,  # Disable dataloader batching
        )

        return loader
