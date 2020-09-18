""" Example of a lightly wrapped torchvision dataset.

This is mostly done for greater flexibility of the data processing pipeline.


"""

import torchvision
from torchvision import transforms


from .. import TorchVisionDataset
from ...exceptions import DataUnavailableError
import tarfile


class ImageNet(TorchVisionDataset):
    __download_train_payload = "ILSVRC2012_img_train.tar"
    __download_val_payload = "ILSVRC2012_img_val.tar"
    __download_devkit_payload = "ILSVRC2012_devkit_t12.tar.gz"

    __download_train_link = (
        "https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2"
    )
    __download_val_link = (
        "https://academictorrents.com/details/dfa9ab2528ce76b907047aa8cf8fc792852facb9"
    )
    __download_devkit_link = (
        "https://drive.google.com/file/d/1ina38-4xDAlWVcPcuMZMIx4filKtFqBL"
    )

    def download(self):
        if (self.path / self.__download_train_payload).exists() and (
            self.path / self.__download_val_payload
        ).exists():
            # Definitely don't download if the download payloads already exist.
            return

        raise DataUnavailableError(
            "\n\n"
            "ImageNet 2012 is sort of hard to download; the official website has a bunch of broken links."
            " The easiest way to obtain the data seems to be via:\n"
            f"   train:  {self.__download_train_link}\n"
            f"   val:    {self.__download_val_link}\n"
            f"   devkit: {self.__download_devkit_link}\n"
            "\n"
            "Once downloaded, move the tar files to:\n"
            f"     {self.path / self.__download_train_payload}\n"
            f"     {self.path / self.__download_val_payload}\n"
            f"     {self.path / self.__download_devkit_payload}\n"
        )
