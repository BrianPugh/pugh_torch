from .. import ROOT_DATASET_PATH, Dataset


class ImageNet(Dataset):
    def download(self):
        raise NotImplementedError
