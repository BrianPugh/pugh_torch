import torch
from torch.utils import tensorboard as tb
import cv2
import numpy as np

from ..transforms import imagenet
from ..mappings.color import get_palette
from ..helpers import add_text_under_img

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None


class SummaryWriter(tb.SummaryWriter):
    """Extension of Summary Writer for convenient common uses."""

    def __init__(self, *args, rgb_transform=None, **kwargs):
        """
        Parameters
        ----------
        rgb_transform : str, torchvision.transforms.*
            Used when rgb images need to be logged. Transform applied to
            rgb images when logged.
        """
        super().__init__(*args, **kwargs)

        self.rgb_transform = self._parse_rgb_transform(rgb_transform)

    def _parse_rgb_transform(self, rgb_transform):
        if rgb_transform is None:
            try:
                return self.rgb_transform
            except AttributeError:
                return lambda x: x  # callable identity function

        if callable(rgb_transform):
            return rgb_transform

        if isinstance(rgb_transform, str):
            rgb_transform = rgb_transform.lower()
            if rgb_transform == "imagenet":
                return imagenet.Unnormalize()

        raise NotImplementedError(f"Cannot parse rgb_transform {rgb_transform}")

    def _parse_labels(self, labels, n):
        """Converts labels into a standardized list of strings."""

        if isinstance(labels, str) or labels is None:
            labels = [
                labels,
            ] * n

        assert isinstance(labels, list)

        return labels

    def add_rgb(
        self,
        tag,
        rgbs,
        global_step=None,
        walltime=None,
        dataformats="CHW",
        *,
        rgb_transform=None,
        n_images=1,
        labels=None,
    ):
        """Applies a transform and adds image to log

        A common scenario is when you only have a normalized image (say, by
        ImageNet's mean and stddev) and you want to log it to tensorboard.

        In this case, it may be more convenient to set:
            rgb_transform="imagenet"
        in the constructor.

        Paramters
        ---------
        rgbs : torch.Tensor
            (B, 3, H, W) Image data.  See ``rgb_transform`` argument.
        rgb_transform : str or callable
            Transform to apply to the rgb data. If not provided, defaults to
            the transform provided in __init__
        n_images : int
            Maximum number of images to add.
        labels : list or str
            Some string to rasterize to text and display under image.
            If str, the same str will be appied under all images.
        """

        assert isinstance(rgbs, torch.Tensor)

        rgb_transform = self._parse_rgb_transform(rgb_transform)
        labels = self._parse_labels(labels, n_images)

        rgbs = rgb_transform(rgbs)
        rgbs = rgbs.cpu().numpy()
        rgbs = np.clip(rgbs, 0, 1)
        rgbs = np.transpose(rgbs, (0, 2, 3, 1))  # (B, H, W, 3)
        rgbs = (rgbs * 255).astype(np.uint8)

        for i, rgb, label in zip(range(n_images), rgbs, labels):
            if label is not None:
                rgb = add_text_under_img(rgb, label)

            self.add_image(
                f"{tag}/{i}",
                rgb,
                global_step=global_step,
                walltime=walltime,
                dataformats="HWC",
            )

    def add_ss(
        self,
        tag,
        rgbs,
        preds,
        targets,
        global_step=None,
        walltime=None,
        dataformats="CHW",
        *,
        rgb_transform=None,
        n_images=1,
        palette="ade20k",
        offset=0,
        labels=None,
    ):
        """Add a semantic segmentation image and it's pairing input montage.

        ``self.add_rgb``'s documentation applies to the ``rgbs`` input here.

        TODO: more control over which image to show

        Parameters
        ----------
        tag : str
            Data identifier
        rgbs : torch.Tensor
            (B, 3, H, W) Image data.  See ``rgb_transform`` argument.
        preds : torch.Tensor
            (B, C, H, W) Predicted semantic segmentation data. This method will
            argmax over the C dimension.
        targets : torch.Tensor
            (B, H, W) Indexed ground truth data.
        rgb_transform : str or callable
            Transform to apply to the rgb data. If not provided, defaults to
            the transform provided in __init__. Expects data to be in range
            [0, 1] after transform.
        n_images : int
            Maximum number of images to add.
        offset : int
            Add this to the pred and target index into the colormap.
            A common value might be 1 if your network isn't using a
            background class.
        labels : list or str
            Some string to rasterize to text and display under image.
            If str, the same str will be appied under all images.
        """

        # Input validation
        if dataformats != "CHW":
            raise NotImplementedError("TODO: allow other dataformats")
        if isinstance(palette, str):
            palette = get_palette(palette)
        elif isinstance(palette, np.ndarray):
            # Do nothing, TODO: maybe add some assertions here
            pass
        else:
            raise NotImplementedError(
                f"Don't know how to handle palette type {type(palette)}"
            )
        n_colors = len(palette)
        rgb_transform = self._parse_rgb_transform(rgb_transform)
        labels = self._parse_labels(labels, n_images)

        for i, rgb in enumerate(rgbs):
            rgbs[i] = rgb_transform(rgb)

        # Get the most likely class from the network's logits/softmax/whatever.
        preds = torch.argmax(preds, dim=1)  # (B, H, W)

        # Move all the data to cpu
        rgbs = rgbs.cpu().numpy()
        preds = preds.cpu().numpy().astype(np.int)
        targets = targets.cpu().numpy().astype(np.int)

        # General cleanup operations
        rgbs = np.clip(rgbs, 0, 1)
        rgbs = np.transpose(rgbs, (0, 2, 3, 1))  # (B, H, W, 3)
        rgbs = (rgbs * 255).astype(np.uint8)

        preds += offset
        targets += offset

        _, h, w, _ = rgbs.shape

        # Iterate over exemplars and log
        # Note: ``zip`` limits iterations to the shortest input iterator.
        for i, rgb, pred, target, label in zip(
            range(n_images), rgbs, preds, targets, labels
        ):
            # Resize pred and target
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            target = cv2.resize(target, (w, h), interpolation=cv2.INTER_NEAREST)

            pred[pred >= n_colors] = 0
            pred[pred < 0] = 0

            target[target >= n_colors] = 0
            target[target < 0] = 0

            # Apply a color palette
            color_pred = palette[pred]
            color_target = palette[target]
            color_target[target < 0] = 0

            # Horizontally combine all three into a single image
            montage = np.concatenate((color_target, rgb, color_pred), axis=1).astype(
                np.uint8
            )

            if label is not None:
                montage = add_text_under_img(montage, label)

            # Log the montage to tensorboard
            self.add_image(
                f"{tag}/{i}",
                montage,
                global_step=global_step,
                walltime=walltime,
                dataformats="HWC",
            )


if pl is not None:

    class TensorBoardLogger(pl.loggers.TensorBoardLogger):
        """Same as default PyTorch Lightning TensorBoard Logger, but uses
        the extended SummaryWriter defined in this file.
        """

        @property
        @pl.loggers.base.rank_zero_experiment
        def experiment(self) -> SummaryWriter:
            r"""
            Actual tensorboard object. To use TensorBoard features in your
            :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
            Example::
                self.logger.experiment.some_tensorboard_function()
            """
            if self._experiment is not None:
                return self._experiment

            assert (
                rank_zero_only.rank == 0
            ), "tried to init log dirs in non global_rank=0"
            if self.root_dir:
                self._fs.makedirs(self.root_dir, exist_ok=True)
            self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
            return self._experiment
