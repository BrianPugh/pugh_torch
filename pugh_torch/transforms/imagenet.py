import torch
from torchvision import transforms
from PIL import Image
import numpy as np

__all__ = [
    "Normalize",
    "Unnormalize",
]


class Normalize:
    """Applies standard ImageNet normalization to an RGB image"""

    def __init__(
        self,
    ):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img):
        """
        Parameters
        ----------
        img : numpy.ndarray or PIL.Image or torch.Tensor
            If numpy.ndarray or PIL.Image, expects (H,W,3).
            If torch.Tensor, expects (3, H, W)
            Expects values to be in range [0, 1], NOT [0,255]
            Channels are in RGB order.

        Returns
        -------
        torch.Tensor
            (C, H, W)
        """

        if not isinstance(img, torch.Tensor):
            # Need to convert to torch.Tensor first
            if isinstance(img, np.ndarray):
                if img.max() > 1 or np.issubdtype(img.dtype, np.integer):
                    raise ValueError(
                        "Image must be a float in range [0,1] and in RGB order prior to transform"
                    )
            img = self.to_tensor(img)  # Converts to tensor, and permutes to (C, H, W)
        return self.normalize(img)  # Expects (C, H, W)


class Unnormalize:
    """Reverses standard ImageNet normalization"""

    def __init__(
        self,
    ):
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

    def __call__(self, img):
        """
        Parameters
        ----------
        img : torch.Tensor
            (3, H, W) normalized tensor (should be apprixmately in range [-1, 1])

        Returns
        -------
        img : torch.Tensor
            (3, H, W) unnormalized tensor (should be approximately in range [0, 1])
        """

        return self.unnormalize(img)
