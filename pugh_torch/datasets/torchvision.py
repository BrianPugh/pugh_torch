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
    auto_construct = True  # Invoke the torchvision dataset constructor in __init__

    def __init_subclass__(cls, **kwargs):
        """Automatically gets the constructor of the torchvision dataset
        of the same name
        """
        super().__init_subclass__(**kwargs)
        cls.torchvision_constructor = getattr(torchvision.datasets, cls.__name__)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.auto_construct:
            kwargs["root"] = self.path
            kwargs["transform"] = None  # We handle our own transforms.
            self.dataset = self.torchvision_constructor(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def download(self):
        """ Handled by the torchvision dataset """
        return

    def unpack(self):
        """ Handled by the torchvision dataset """
        return

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __getitem__(self, index):
        """This is a typical implementation, you can override this with
        your own methods.

        Remember: this is the only location ``self.transform`` is used,
        so its exact type/usage/interpretation is up to you!

        This exact implementation won't work for every dataset.
        """

        img, label = self.dataset[index]
        img = self.transform(img)
        return img, label
