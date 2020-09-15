from .. import ROOT_DATASET_PATH, Dataset
from ...exceptions import DataUnavailableError
import tarfile


class ImageNet(Dataset):
    __download_train_payload = "ILSVRC2012_img_train.tar"
    __download_train_sha1 = "43eda4fe35c1705d6606a6a7a633bc965d194284"
    __download_val_payload = "ILSVRC2012_img_val.tar"
    __download_val_sha1 = "5f3f73da3395154b60528b2b2a2caf2374f5f178"

    __download_train_link = "https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2"
    __download_val_link = "https://academictorrents.com/details/dfa9ab2528ce76b907047aa8cf8fc792852facb9"

    def download(self):
        raise DataUnavailableError(
                "ImageNet 2012 is sort of hard to download; the official website has a bunch of broken links."
                " The easiest way to obtain the data seems to be via:\n"
                f"   train: {self.__download_train_link}\n"
                f"   val:   {self.__download_val_link}\n"
                )


    def unpack(self):
        with tarfile.open(fname) as tar:
            tar.extractall(path=targetd_dir)
