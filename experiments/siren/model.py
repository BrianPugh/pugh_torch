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
from pugh_torch.models import resnet50
from pugh_torch.modules import conv3x3, conv1x1
from pugh_torch.modules.meta import BatchLinear

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from scipy.interpolate import griddata
import numpy as np
from pathlib import Path

from dataset import ImageNetSample, SingleImageDataset

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


class HyperHead(nn.Module):
    """For the multi-heads in a HyperNetwork"""

    def __init__(self, f_in, hypo_in, hypo_out):
        super().__init__()

        self.hypo_in = hypo_in
        self.hypo_out = hypo_out

        self.weight_linear = nn.Linear(f_in, hypo_in * hypo_out)
        self.bias_linear = nn.Linear(f_in, hypo_out)

        self._hyper_weight_init(self.weight_linear)
        self._hyper_bias_init(self.bias_linear)

    def _hyper_weight_init(self, m):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        m.weight.data = m.weight.data / 1.0e2

        with torch.no_grad():
            m.bias.uniform_(-1 / self.hypo_in, 1 / self.hypo_in)

    def _hyper_bias_init(self, m):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        m.weight.data = m.weight.data / 1.0e2

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)

    def forward(self, x):
        batch = x.shape[0]

        weight = self.weight_linear(x)
        weight = weight.reshape(batch, self.hypo_out, self.hypo_in)

        bias = self.bias_linear(x)

        return weight, bias


class HyperNetwork(nn.Module):
    def __init__(
        self,
        input,
        hidden,
        hypo_network,
        activation="relu",
    ):
        """
        input : int
            number feature in from embedding
        hidden : list
            List of ints, hidden layer nodes for hyper network
        hypo_network : nn.Sequential
            Sequential hypo_network
        """

        super().__init__()

        self.activation = pt.modules.Activation(activation)

        self.layers = nn.ModuleList()
        if hidden:
            self.layers.append(nn.Linear(input, hidden[0]))
            self.layers.append(self.activation)
            for h_in, h_out in zip(hidden, hidden[1:]):
                self.layers.append(nn.Linear(h_in, h_out))
                self.layers.append(self.activation)
            num_encoding = hidden[-1]
        else:
            num_encoding = input

        # Create all the heads to predict the hyponetwork parameters
        self.heads = nn.ModuleList()
        for module in hypo_network:
            if not isinstance(module, nn.Linear):
                continue
            f_out, f_in = module.weight.shape
            self.heads.append(HyperHead(num_encoding, f_in, f_out))

    def forward(self, x):
        # Run it through the hidden layers
        for layer in self.layers:
            x = layer(x)

        # Run it through the heads
        outputs = []
        for head in self.heads:
            outputs.append(head(x))

        # unpack the outputs
        weights = [x[0] for x in outputs]
        biases = [x[1] for x in outputs]

        return weights, biases


class SIREN(nn.Sequential):
    def __init__(self, layers):
        """
        Parameters
        ----------
        layers : list
           List of integers describing the numbers of node per layer where
           layers[0] is the number of input features and
        """

        modules = []

        modules.append(BatchLinear(layers[0], layers[1]))
        modules.append(pt.modules.Activation("sine", modules[-1], first=True))

        # hidden layers
        for f_in, f_out in zip(layers[1:-2], layers[2:-1]):
            modules.append(BatchLinear(f_in, f_out))
            modules.append(pt.modules.Activation("sine", modules[-1]))

        modules.append(BatchLinear(layers[-2], layers[-1]))

        super().__init__(*modules)

    def forward(self, input, weights=None, biases=None):
        bl_count = 0

        if weights is None or biases is None:
            assert weights is None and biases is None
            return super().forward(input)

        for module in self:
            if isinstance(module, BatchLinear):
                input = module(input, weights[bl_count], biases[bl_count])
                bl_count += 1
            else:
                input = module(input)
        return input


class HyperSIRENPTL(pt.LightningModule):
    """Trainable network that contains 3 main components:
        1. Encoder - any CNN for feature extraction. Here it's ResNet50 and
           it produces a 256 element feature vector
        2. HyperNetwork - A FC network that predicts weights and biases
           for a SIREN network.
        3. SIREN - Technically in this usecase, this has no learnable parameters
           because it uses the output of the HyperNetwork.

    End goal is to produce better SIREN initializations for learning
    coordinate->image mappings faster.
    """

    def __init__(
        self,
        *,
        cfg=None,
        encoder={},
        hyper={},
        siren={},
        learning_rate=0.002,
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

        embedding_size = encoder.get("size", 256)
        self.encoder_net = resnet50(pretrained=True, num_classes=embedding_size)
        self.encoder_activation = pt.modules.Activation(hyper.get("activation", "relu"))

        siren_nodes = [2, *siren.get("layers", [128] * 5), 3]
        self.siren_net = SIREN(siren_nodes)

        hyper_hidden = hyper.get("layers", [256])
        self.hyper_net = HyperNetwork(embedding_size, hyper_hidden, self.siren_net)

        self.siren_loss_fn = pt.losses.get_functional_loss(
            siren.get("loss", "mse_loss")
        )

    def forward(self, coords, imgs):
        embedding = self.encoder_net(imgs)  # (B, embedded_feat)
        siren_weights, siren_biases = self.hyper_net(
            self.encoder_activation(embedding)
        )  # (B, long_flattened_params)
        pred = self.siren_net(coords, siren_weights, siren_biases)
        return embedding, siren_weights, siren_biases, pred

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
        batch_size = coords.shape[0]

        embedding, siren_weights, siren_biases, pred = self(coords, imgs)
        self._log_loss("train", pred, rgb_vals)

        siren_loss = self.siren_loss_fn(pred, rgb_vals)

        # Regularization encourages a gaussian prior on embedding from context encoder
        embedding_reg = self.encoder_cfg["loss_weight"] * (embedding * embedding).mean()

        # Regularization encourages a lower frequency representation of the image
        # Not sure i believe that, but its what the paper says.
        n_params = sum([w.shape[-1] * w.shape[-2] for w in siren_weights])
        cum_mag = sum([torch.sum(w * w, dim=(-1, -2)) for w in siren_weights])
        hyper_reg = self.hyper_cfg["loss_weight"] * (cum_mag / n_params).mean()

        loss = siren_loss + embedding_reg + hyper_reg

        self._log_common("train", pred, rgb_vals, loss)

        return loss

    def validation_step(self, batch, batch_nb):
        coords, rgb_vals, imgs = batch

        embedding, siren_weights, siren_biases, pred = self(coords, imgs)
        loss = self._log_loss("val", pred, rgb_vals)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        transform = A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(*self.cfg.dataset["shape"]),
                A.HorizontalFlip(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # this is RGB order.
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = ImageNetSample(
            split="train", transform=transform, num_sample=self.cfg.dataset.num_sample
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
                A.RandomCrop(*self.cfg.dataset["shape"]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # this is RGB order.
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = ImageNetSample(
            split="val", transform=transform, num_sample=self.cfg.dataset.num_sample
        )

        n_exemplar = len(dataset)
        step = n_exemplar // self.cfg.dataset.num_val_subset
        indices = list(range(0, n_exemplar))[0 : n_exemplar * step : step]
        dataset = torch.utils.data.Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=self.cfg.dataset.pin_memory,
            num_workers=self.cfg.dataset.num_workers,
            batch_size=self.cfg.dataset.batch_size,
        )
        return loader


class SIRENCoordToImg(pt.LightningModule):
    """Model that learns coordinate->RGB image mapping"""

    def __init__(
        self,
        *,
        cfg=None,
        layers=[128, 128, 128, 128],
        learning_rate=0.002,
        loss="mse_loss",
        optimizer="adamw",
        optimizer_kwargs={},
        hyper_ckpt=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        siren_nodes = [2, *layers, 3]
        self.model = SIREN(siren_nodes)

        self.loss_fn = pt.losses.get_functional_loss(loss)

        if hyper_ckpt:
            # TODO initialize network using hypernetwork here if available
            hyper = HyperSIRENPTL.load_from_checkpoint(hyper_ckpt)

            transform = A.Compose(
                [
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],  # this is RGB order.
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ]
            )

            img = self.transform(image=self.img)["image"]

            with torch.no_grad():
                _, siren_weights, siren_biases, _ = hyper(img)
                import ipdb as pdb

                pdb.set_trace()
                for w, b in zip(siren_weights, siren_biases):
                    pass

            raise NotImplementedError

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

    @property
    def img(self):
        try:
            return self._img
        except AttributeError:
            # log the ground truth image to tensorboard for comparison
            img_path = this_file_dir / self.cfg.dataset.path
            self._img = cv2.imread(str(img_path))[..., ::-1]
            if self.cfg.dataset.shape:
                shape = self.cfg.dataset.shape
                self._img = cv2.resize(
                    self._img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
                )

        return self._img

    def on_train_start(
        self,
    ):
        # log the ground truth image to tensorboard for comparison
        self.logger.experiment.add_image(
            f"ground_truth",
            self.img,
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
