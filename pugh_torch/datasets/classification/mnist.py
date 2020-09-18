from .. import TorchVisionDataset


class MNIST(TorchVisionDataset):
    auto_construct = False

    def __init__(self, **kwargs):
        """
        Note both "val" and "test" splits map to "test"
        """

        super().__init__(**kwargs)

        self.dataset = self.torchvision_constructor(
            root=self.path,
            train=self.split == "train",
            transform=None,
            target_transform=None,
            download=True,
        )
