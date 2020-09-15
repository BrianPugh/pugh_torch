"""
"""

import torch
from . import ROOT_DATASET_PATH


class Dataset(torch.utils.data.Dataset):
    """"""

    def __init_subclass__(cls, **kwargs):
        """"""
        pass

    def __init__(self, *args, split="train", **kwargs):
        """
        Attempts to download data.
        """

        assert split in ("train", "val", "test")
        self.split = split

        self._download_dataset_if_not_downloaded()

    def _download_dataset_if_not_downloaded(self):
        if not self.downloaded:
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()
            self.downloaded = True


    def _unpack_dataset_if_not_unpacked(self):
        if not self.unpacked:
            self.path.mkdir(parents=True, exist_ok=True)
            self.unpack()
            self.unpacked = True


    @property
    def path(self):
        """pathlib.Path to the root of the stored data"""

        try:
            return self.__path
        except AttributeError:
            try:
                dataset_type = self.__class__.__module__.split(".")[
                    -2
                ]  # e.x. "classification", "segmentation"
            except IndexError:
                dataset_type = "unknown"
            self.__path = ROOT_DATASET_PATH / dataset_type / self.__class__.__name__
            return self.__path

    @property
    def downloaded_file(self):
        return self.path / "downloaded"

    @property
    def downloaded(self):
        """We detect if the data has been fully downloaded by a "downloaded"
        file in the root of the data directory.
        """

        return self.downloaded_file.exists()

    @downloaded.setter
    def downloaded(self, val):
        """Touch/Delete sentinel downloaded data file"""

        if val:
            try:
                self.downloaded_file.touch(exist_ok=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Could not create sentinel "{self.downloaded_file}" file, which indicates the directory doesn\'t exist, and so the data most certainly has NOT been downloaded!'
                ) from e
        else:
            self.downloaded_file.unlink(missing_ok=True)

    def download(self):
        """Function to download data to ``self.path``.

        The directories up to ``self.path`` have already been created.

        Will only be called if data has not been downloaded.
        """

        raise NotImplementedError


    @property
    def unpacked_file(self):
        return self.path / "unpacked"

    @property
    def unpacked(self):
        """We detect if the data has been fully unpacked by a "unpacked"
        file in the root of the data directory.
        """

        return self.unpacked_file.exists()

    @unpacked.setter
    def unpacked(self, val):
        """Touch/Delete sentinel unpacked data file"""

        if val:
            try:
                self.unpacked_file.touch(exist_ok=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Could not create sentinel "{self.unpacked_file}" file, which indicates the directory doesn\'t exist, and so the data most certainly has NOT been unpacked!'
                ) from e
        else:
            self.unpacked_file.unlink(missing_ok=True)

    def unpack(self):
        """Post-process the downloaded payload.

        Typically this will be something like unpacking a tar file, or possibly
        re-arranging files.
        """

        raise NotImplementedError
