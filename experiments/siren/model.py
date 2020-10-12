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

log = logging.getLogger(__name__)

# class ImagePixelIterator:
#    def __init__(self, img, batch_size):
#        self.img = img
#        self.batch_size = batch_size
#
#        nx, ny = self.img.shape[1], self.img.shape[0]
#        # (X, Y)
#        self.meshgrid = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
#
#    def __len__(self):
#
#    def __iter__(self):
#        return self
#
#    def __next__(self):
#        pass


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
            if len(shape) == 1:
                shape = shape[0]
            if isinstance(shape, int):
                shape = (shape, shape)
            assert len(shape) == 2
            self.img = cv2.resize(
                self.img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
            )
        self.shape = np.array(shape)  # (H, W) i.e. (y, x)

        # Normalize the image
        if normalize:
            self.img = pt.transforms.imagenet.np_normalize(self.img / 255)
        self.flat_img = self.img.reshape((-1, 3))

        nx, ny = self.img.shape[1], self.img.shape[0]
        # (X, Y)
        meshgrid = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
        self.src_pts = (meshgrid[0].reshape(-1), meshgrid[1].reshape(-1))
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
            entropy = np.random.rand(self.batch_size, 2)
            coord_normalized = 2 * entropy - 1
            coord_unnormalized = entropy * np.array(
                (self.shape[1] - 1, self.shape[0] - 1)
            ).reshape((1, 2))
            coord_unnormalized = coord_unnormalized

            # interpolate at those points
            rgb_values = griddata(self.src_pts, self.flat_img, coord_unnormalized)
            self.pos += self.batch_size
        elif self.mode == "val":
            # Deterministcally return every pixel value in order
            # Last batch size may be smaller
            upper = self.pos + self.batch_size
            if upper > self.n_pixels:
                upper = self.n_pixels

            coord_normalized = np.hstack(
                (
                    self.src_pts_normalized[0][self.pos : upper, None],
                    self.src_pts_normalized[1][self.pos : upper, None],
                )
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
        activationG="sine",
        learning_rate=0.002,
    ):
        super().__init__()

        self.cfg = cfg

        self.learning_rate = learning_rate

        model = []
        model.append(nn.Linear(2, layers[0]))
        model.append(pt.modules.Activation(activation, model[-1]))

        for cur_layer, next_layer in zip(layers, layers[1:]):
            model.append(nn.Linear(cur_layer, next_layer))
            model.append(pt.modules.Activation(activation, model[-1]))

        model.append(nn.Linear(layers[-1], 3))

        self.model = nn.Sequential(model)

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

    def _log_common(self, split, logits, target, loss):
        pred = torch.argmax(logits, dim=-1)
        self.log(f"{split}/loss", loss, prog_bar=True)
        try:
            self.log(f"{split}/acc", accuracy(pred, target), prog_bar=True)
        except RuntimeError:
            # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/3006
            pass

    def _compute_loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def training_step(self, batch, batch_nb):
        """"""

        x, y = batch
        logits = self(x)
        self.last_logits = logits  # Commonly used in callbacks
        loss = self._compute_loss(logits, y)

        self._log_common("train", logits, y, loss)

        return loss

    def validation_step(self, batch, batch_nb):
        """"""

        # OPTIONAL
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)

        self._log_common("val", logits, y, loss)

        return loss

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        optimizers.append(torch.optim.AdamW(self.parameters(), lr=self.learning_rate))
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
        from pugh_torch.callbacks import TensorBoardAddClassification

        callbacks = [
            TensorBoardAddClassification(rgb_transform="imagenet"),
        ]
        return callbacks

    def train_dataloader(self):
        dataset = SingleImageDataset(self.cfg.dataset.path)
        dataset = pt.datasets.get("classification", self.cfg.dataset.name)(
            "train", transform=transform
        )
        loader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=self.cfg.dataset.pin_memory,
            num_workers=self.cfg.dataset.num_workers,
            # batch_size=self.cfg.dataset.batch_size,
            batch_size=None,  # Disable dataloader batching
            batch_sampler=None,  # Disable dataloader batching
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2 ** 32 - 1)
            ),
        )
        return loader

    def val_dataloader(self):

        dataset = pt.datasets.get("classification", self.cfg.dataset.name)(
            "val", transform=transform
        )
        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=self.cfg.dataset.pin_memory,
            num_workers=self.cfg.dataset.num_workers,
            batch_size=self.cfg.dataset.batch_size,
        )
        return loader
