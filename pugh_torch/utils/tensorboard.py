import torch
from torch.utils import tensorboard as tb
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

        self._init_rgb_transform(rgb_transform)

    def _init_rgb_transform(self, rgb_transform):
        if rgb_transform is None:
            self.rgb_transform = None
            return

        if callable(rgb_transform):
            self.rgb_transform = rgb_transform
            return

        if isinstance(rgb_transform, str):
            rgb_transform = rgb_transform.lower()
            if rgb_transform == "imagenet":
                self.rgb_transform = imagenet.Unnormalize()
            else:
                raise NotImplementedError(f"Cannot parse rgb_transform {rgb_transform}")
            return

        raise NotImplementedError(f"Cannot parse rgb_transform {rgb_transform}")


    def add_ss(self, tag, rgb, ss, global_step=None, walltime=None, dataformats='CHW'):
        """ Add a semantic segmentation image and it's pairing input

        Parameters
        ----------
        tag : str
            Data identifier
        rgb : torch.Tensor
            Image data
        """
        pass
