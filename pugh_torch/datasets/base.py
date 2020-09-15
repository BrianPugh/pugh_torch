"""
"""

import torch
from . import ROOT_DATASET_PATH


class Dataset(torch.utils.data.Dataset):
    """
    """

    def __init_subclass__(cls, **kwargs):
        """ 
        """
        pass


    @property
    def path(self):
        try:
            return self.__path
        except AttributeError:
            try:
                dataset_type = self.__class__.__module__.split('.')[-2]  # e.x. "classification", "segmentation"
            except IndexError:
                dataset_type = "unknown"
            self.__path = ROOT_DATASET_PATH / dataset_type / self.__class__.__name__
            return self.__path


    @property
    def downloaded(self):
        """ We detect if the data has been fully downloaded by a "downloaded"
        file in the root of the data directory.
        """
        import ipdb as pdb; pdb.set_trace()

        downloaded_path = self.path / downloaded
        return downloaded_path.exists()
