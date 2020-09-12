import torch
from torchvision import transforms


class Normalize:
    """ Applies standard ImageNet normalization
    """

    def __init__(self,):
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)


class Unnormalize:
    """ Reverses standard ImageNet normalization
    """

    def __init__(self,):
        self.transforms = transforms.Compose([
                transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                        std=[1/0.229, 1/0.224, 1/0.255],
                ),
        ])


    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)
