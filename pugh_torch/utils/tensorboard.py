import torch
from torch.utils import tensorboard as tb
import cv2
import numpy as np
from ..transforms import imagenet


class SummaryWriter(tb.SummaryWriter):
    """ Extension of Summary Writer for convenient common uses.
    """

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
                return None

        if callable(rgb_transform):
            return rgb_transform

        if isinstance(rgb_transform, str):
            rgb_transform = rgb_transform.lower()
            if rgb_transform == "imagenet":
                return imagenet.Unnormalize()

        raise NotImplementedError(f"Cannot parse rgb_transform {rgb_transform}")


    def add_rgb(self, tag, rgb, global_step=None, walltime=None, dataformats='CHW', *,
            rgb_transform=None,
            ):
        """
        Paramters
        ---------
        rgb_transform : str or callable
            Transform to apply to the rgb data. If not provided, defaults to 
            the transform provided in __init__
        """

        rgb_transform = self._parse_rgb_transform(rgb_transform)


    def add_ss(self, tag, rgb, ss, global_step=None, walltime=None, dataformats='CHW', *,
            rgb_transform=None,
            ):
        """ Add a semantic segmentation image and it's pairing input montage

        Parameters
        ----------
        tag : str
            Data identifier
        rgb : torch.Tensor
            (B, 3, H, W) Image data
        ss : torch.Tensor
            (B, C, H, W) semantic segmentation data
        rgb_transform : str or callable
            Transform to apply to the rgb data. If not provided, defaults to 
            the transform provided in __init__
        """

        rgb_transform = self._parse_rgb_transform(rgb_transform)

