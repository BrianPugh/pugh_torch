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


    def __init__(self, *args, **kwargs):
        """
        Attempts to download data.
        """

        self._download_dataset_if_not_downloaded()


    def _download_dataset_if_not_downloaded(self):
        if not self.downloaded:
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()
            self.downloaded = True


    @property
    def path(self):
        """ pathlib.Path to the root of the stored data
        """

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
    def downloaded_file(self):
        return self.path / "downloaded"


    @property
    def downloaded(self):
        """ We detect if the data has been fully downloaded by a "downloaded"
        file in the root of the data directory.
        """

        return self.downloaded_file.exists()


    @downloaded.setter
    def downloaded(self, val):
        """ Touch/Delete sentinel downloaded data file
        """

        if val:
            try:
                self.downloaded_file.touch(exist_ok=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not create sentinel \"{self.downloaded_file}\" file, which indicates the directory doesn't exist, and so the data most certainly has NOT been downloaded!") from e
        else:
            self.downloaded_file.unlink(missing_ok=True)


    def download(self):
        """ Function to download data to ``self.path``.

        The directories up to ``self.path`` have already been created.

        Will only be called if data has not been downloaded. 
        """

        raise NotImplementedError
