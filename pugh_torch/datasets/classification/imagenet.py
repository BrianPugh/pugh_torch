import torchvision

from .. import Dataset
from ...exceptions import DataUnavailableError
import tarfile


class ImageNet(Dataset, torchvision.datasets.ImageNet):
    __download_train_payload = "ILSVRC2012_img_train.tar"
    __download_val_payload = "ILSVRC2012_img_val.tar"

    __download_train_link = "https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2"
    __download_val_link = "https://academictorrents.com/details/dfa9ab2528ce76b907047aa8cf8fc792852facb9"

    def __init__(self, split='train'):
        Datset.__init__(split=split)
        torchvision.datasets.ImageNet(root=self.path, split=split)


    def download(self):
        if (self.path / self.__download_train_payload).exists() \
                and (self.path / self.__download_val_payload).exists():
            # Definitely don't download if the download payloads already exist.
            return

        raise DataUnavailableError(
                "ImageNet 2012 is sort of hard to download; the official website has a bunch of broken links."
                " The easiest way to obtain the data seems to be via:\n"
                f"   train: {self.__download_train_link}\n"
                f"   val:   {self.__download_val_link}\n"
                "\n"
                "Once downloaded, move the tar files to:\n"
                f"     {self.path / self.__download_train_payload}\n"
                f"     {self.path / self.__download_val_payload}\n"
                )


    def unpack(self):
        with tarfile.open(fname) as tar:
            tar.extractall(path=targetd_dir)
