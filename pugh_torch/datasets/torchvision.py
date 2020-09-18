""" Lightly wraps torchvision datasets.

This just allows us greater customization without modifying another repo.

Most notably, this:
    * Automatically gets the torchvision dataset constructor based on name
    * Moves the transform responsibility to us
    * Applies our automatic opinionated pathing rules.
"""

import torchvision
from .base import Dataset


class TorchVisionDataset(Dataset):
    def __init_subclass__(cls, **kwargs):
        """Automatically gets the constructor of the torchvision dataset
        of the same name
        """

        cls.__torchvision_constructor = getattr(torchvision.datasets, cls.__name__)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kwargs["root"] = self.path
        kwargs["transform"] = None  # We handle our own transforms.
        self.dataset = self.__torchvision_constructor(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def download(self):
        """ Handled by the torchvision dataset """
        return

    def unpack(self):
        """ Handled by the torchvision dataset """
        return

    def __getitem__(self, index):
        """This is a typical implementation, you can override this with
        your own methods.

        Remember: this is the only location ``self.transform`` is used,
        so its exact type/usage/interpretation is up to you!

        This exact implementation won't work for every dataset.
        """

        img, label = self.dataset[index]

        import ipdb as pdb

        pdb.set_trace()
        img = self.transform(img)

        return img, label
