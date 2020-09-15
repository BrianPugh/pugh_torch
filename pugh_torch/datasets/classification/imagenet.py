from .. import ROOT_DATASET_PATH, Dataset
import tarfile


class ImageNet(Dataset):
    __download_train_payload = "ILSVRC2012_img_train.tar"
    __download_train_sha1 = "43eda4fe35c1705d6606a6a7a633bc965d194284"
    __download_val_payload = "ILSVRC2012_img_val.tar"
    __download_val_sha1 = "5f3f73da3395154b60528b2b2a2caf2374f5f178"

    def download(self):
        raise NotImplementedError

    def unpack(self):
        with tarfile.open(fname) as tar:
            tar.extractall(path=targetd_dir)
