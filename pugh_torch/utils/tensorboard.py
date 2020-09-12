import torch
from torch.utils import tensorboard as tb
import cv2
import numpy as np
from ..transforms import imagenet
from ..mappings.color import get_palette


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
                return lambda x: x  # callable identity function

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


    def add_ss(self, tag, rgbs, preds, targets,
            global_step=None,
            walltime=None,
            dataformats='CHW', *,
            rgb_transform=None,
            n_images=1,
            palette='ade20k',
            offset=0,
            ):
        """ Add a semantic segmentation image and it's pairing input montage

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
        """

        # Input validation
        if dataformats != 'CHW':
            raise NotImplementedError("TODO: allow other dataformats")
        if isinstance(palette, str):
            palette = get_palette(palette)
        elif isinstance(palette, np.ndarray):
            # Do nothing, TODO: maybe add some assertions here
            pass
        else:
            raise NotImplementedError(f"Don't know how to handle palette type {type(palette)}")
        rgb_transform = self._parse_rgb_transform(rgb_transform)
        rgbs = rgb_transform(rgbs)

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
        for i, rgb, pred, target in zip(range(n_images), rgbs, preds, targets):
            # Resize pred and target
            pred = cv2.resize(pred, (w,h), interpolation=cv2.INTER_NEAREST)
            target = cv2.resize(target, (w,h), interpolation=cv2.INTER_NEAREST)

            # Apply a color palette
            color_pred = palette[pred]
            color_target = palette[target]

            # Horizontally combine all three into a single image
            montage = np.concatenate((color_target, rgb, color_pred), axis=1).astype(np.uint8)

            # Log the montage to tensorboard
            self.add_image(f"{tag}/{i}", montage, global_step=global_step, walltime=walltime, dataformats="HWC")
