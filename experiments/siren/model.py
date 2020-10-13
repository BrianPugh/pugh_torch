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
from pathlib import Path

from dataset import SingleImageDataset

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


class SIREN(pt.LightningModule):
    """ Model that can learn an RGB image
    """

    def __init__(
        self,
        *,
        cfg=None,
        layers=[128, 128, 128, 128],
        activation="sine",
        learning_rate=0.002,
        loss="mse_loss",
        optimizer="adamw",
        optimizer_kwargs={},
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

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
            pt.optimizers.get_optimizer(self.optimizer)(
                self.parameters(),
                lr=self.learning_rate,
                **self.optimizer_kwargs,
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

class ParameterizedFC(nn.Module):

    def forward(self, x, weight, bias):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, *, feat_in)
        weights : list
            Where each element is of shape (B, *, feat_in, feat_out).
            Where feat_in is the size of the previous feat_out
        biases : list
            Where each element is of shape (B, feat_out)
        """

        x = x.matmul(weight) + bias.unsqueeze(-2)
        return x

class ParameterizedFCs(nn.Module):
    def __init__(
            self,
            activation="sine",
            ):
        super().__init__()
        self.activation = pt.modules.Activation(activation)

    def forward(self, x, weights, biases):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, *, feat_in)
        weights : list
            Where each element is of shape (B, *, feat_in, feat_out).
            Where feat_in is the size of the previous feat_out
        biases : list
            Where each element is of shape (B, feat_out)
        """

        n_layers = len(weights)

        assert n_layers == len(biases)

        activations = [torch.activation] * n_layers
        activations[-1] = None

        for weight, bias, activation in zip(weights, biases, activations):
            x = x.matmul(weight) + bias.unsqueeze(-2)
            if activation is not None:
                x = activation(x)

        return x

class FC(nn.Module):
    """ Simple Sequential FC
    """

    def __init__(
            self,
            layers=[128, 128],
            activation="relu",
            ):

        super().__init__()

        assert len(layers) >= 2

        model = []
        for cur_layer, next_layer in zip(layers, layers[1:-1]):
            model.append(nn.Linear(cur_layer, next_layer))
            model.append(pt.modules.Activation(activation, model[-1]))

        model.append(nn.Linear(layers[-2], layers[-1]))
        # no final activation function

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class FastSIREN(pt.LightningModule):
    """ Experimental model where SIREN weights are initialized via a 
    learned CNN from observing the image, hopefully to decrease training
    time.
    """
    def __init__(
        self,
        *,
        cfg=None,
        encoder={},
        hyper={},
        siren={},



        learning_rate=0.002,
        loss="mse_loss",
        optimizer="adamw",
        optimizer_kwargs={},
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.encoder_cfg = encoder
        self.hyper_cfg = hyper
        self.siren_cfg = siren

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        embedding_size = encoder.get('size', 256)
        self.encoder_net = resnet50(pretrained=True, num_classes=embedding_size)

        siren_nodes = [2, *siren.get('layers', [128] * 5), 3]

        # Compute how to partition up the hyper-network output
        n_params = 0
        self.siren_weight_indices = []
        self.siren_weight_shapes = []
        self.siren_bias_indices = []
        for f_in, f_out in zip(siren_nodes[:-1], siren_nodes[1:]):
            self.siren_bias_indices.append((n_params, n_params + f_out))
            n_params += f_out
            self.siren_weight_indices.append((n_params, n_params + f_in * f_out))
            n_params += f_in * f_out

        # This doesn't actually have any internal parameters
        self.siren_net = ParameterizedFCs(activation=siren.get('activation', 'sine'))

        hyper_hidden_layers = hyper.get('layers', [256,])
        hyper_layers = [embedding_size, *hyper_hidden_layers, siren_parameters]
        self.hyper_net = FC(layers=hyper_layers, activation=hyper.get('activation', 'relu'))
        # TODO: use thier special kaming initialization modification on the self.hyper_net

        self.encoder_loss_fn = pt.losses.get_functional_loss(encoder.get("loss", "mse_loss"))
        self.siren_loss_fn = pt.losses.get_functional_loss(siren.get("loss", "mse_loss"))
        self.hyper_loss_fn = pt.losses.get_functional_loss(hyper.get("loss", "mse_loss"))


    def _reshape_siren_parameters(self, param):
        weights = [
                param[idx_pair[0]:idx_pair[1]].reshape(shape) 
                for idx_pair, shape in 
                zip(self.siren_weight_indices, self.siren_weight_shapes)
                ]
        biases = [param[idx_pair[0]:idx_pair[1]]for idx_pair in self.siren_bias_indices]

        return weights, biases

    def forward(self, coords, imgs):
        coords, imgs = x
        embedding = self.encoder_net(imgs)
        siren_parameters = self.hyper_net(embedding)
        weights, biases = self._reshape_siren_parameters(siren_parameters)
        pred = self.siren_net(coords, weights, biases)
        return embedding, siren_parameters, pred

    def _log_common(self, split, logits, target, loss):
        self.log(f"{split}_loss", loss, prog_bar=True)

    def _log_loss(self, split, pred, target):
        # Makes it easier to directly compare techniques that have a different
        # loss function
        loss = F.mse_loss(pred, target)
        self.log(f"{split}_mse_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_nb):
        coords, rgb_vals, imgs = batch

        embedding, siren_parameters, pred = self(coords, imgs)
        self._log_loss("train", pred, rgb_vals)

        loss = self.siren_loss_fn(pred, rgb_vals)

        # Regularization encourages a gaussian prior on embedding from context encoder
        loss = loss + self.context_cfg.get("loss_weight", 1e-1) * self.encoder_loss_fn(embedding, 0)

        # Regularization encourages a lower frequency representation of the iamge
        # Not sure i believe that, but its what the paper says.
        loss = loss + self.hyper_cfg.get("loss_weight", 1e2) * self.hyper_loss_fn(siren_parameters, 0)

        self._log_common("train", pred, rgb_vals, loss)

        return loss

    def validation_step(self, batch, batch_nb):
        coords, rgb_vals, imgs = batch

        embedding, siren_parameters, pred = self(coords, imgs)
        loss = self._log_loss("val", pred, rgb_vals)
        return loss

    def train_dataloader(self):
        transform = A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # this is RGB order.
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = pt.datasets.get("classification", self.cfg.dataset.name)(
            split="train", transform=transform, num_sample=self.cfg.dataset.num_sample,
        )
        loader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=self.cfg.dataset.pin_memory,
            num_workers=self.cfg.dataset.num_workers,
            batch_size=self.cfg.dataset.batch_size,
        )
        return loader

    def val_dataloader(self):
        transform = A.Compose(
            [
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # this is RGB order.
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = pt.datasets.get("classification", self.cfg.dataset.name)(
            split="val", transform=transform, 
        )
        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=self.cfg.dataset.pin_memory,
            num_workers=self.cfg.dataset.num_workers,
            batch_size=self.cfg.dataset.batch_size,
        )
        return loader
