from .. import TorchVisionDataset


class CIFAR10(TorchVisionDataset):
    auto_construct = False

    def __init__(self, **kwargs):
        """
        Note both "val" and "test" splits map to "test"
        """

        super().__init__(**kwargs)

        self.dataset = self.torchvision_constructor(
            root=self.path,
            train=self.split,
            transform=None,
            target_transform=None,
            download=True,
        )


class CIFAR100(CIFAR10):
    pass
