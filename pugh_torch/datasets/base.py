"""
Design philosophies/rules:
    * All datasets in this repo are a child of ``Dataset``.
    * All paths are pathlib.Path objects.
        * If something cannot handle it as a Path object, cast it to a string
          as late as possible.
    * Whenever possible, require the least amount of effort on the dev's part
      to get a dataset downloaded and properly formatted.
    * Dataset directories are automatically parsed/derived, so no need to
      prompt the developer on where they want their dataset files.
    * ``self.transform`` is ONLY ever used in the dev's implementation of
      ``self.__getitem__``


To implement your own dataset:
    1. Subclass the ``pugh_torch.datasets.Dataset`` class.
       This class itself is a subclass of ``torch.utils.data.Dataset``.
    2. Implement the download method:
            def download(self):
                # the local folder (guarenteed to exist) is ``self.path``
       This will only be called if the downloaded data isn't available.
       The download being available is determined by a sentinel "downloaded"
       file.
    3. Implement the unpack method:
            def unpack(self):
                # the local folder (guarenteed to exist) is ``self.path``
       This will only be called if the data hasn't been unpacked yet.
       The unpacked being available is determined by a sentinel "unpacked"
       file.

    4. Follow the other remaining instructions at:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""


import torch
from torchvision import transforms
from . import ROOT_DATASET_PATH, DATASETS


class Dataset(torch.utils.data.Dataset):
    """"""

    def __init_subclass__(cls, **kwargs):
        """Automatic registration stuff"""
        super().__init_subclass__(**kwargs)

        # Register in DATASETS
        modules = cls.__module__.split(".")
        if len(modules) > 3 and modules[0] == "pugh_torch" and modules[1] == "datasets":
            d = DATASETS
            for module in modules[2:-1]:
                if module not in d:
                    d[module] = {}
                d = d[module]
            d[cls.__name__.lower()] = cls

    def __init__(self, split="train", *, transform=None, **kwargs):
        """
        Attempts to download data.

        Parameters
        ----------
        split : str
            One of {"train", "val", "test"}.
            Which data partition to use. Case insensitive.
        transform : obj
            Whatever format you want. Depends on dataset __getitem__ implementation.
            Defaults to just a ``ToTensor`` transform.
            This attribute is NOT used anywhere except in the dataset-specific
            __get__ implementation, or other parent classes of the dataset..
        """

        split = split.lower()
        assert split in ("train", "val", "test")
        self.split = split

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        self.path.mkdir(parents=True, exist_ok=True)
        self._download_dataset_if_not_downloaded()
        self._unpack_dataset_if_not_unpacked()

    def _download_dataset_if_not_downloaded(self):
        self.path.mkdir(parents=True, exist_ok=True)

        if self.unpacked or self.downloaded:
            return

        self.download()
        self.downloaded = True

    def _unpack_dataset_if_not_unpacked(self):
        if self.unpacked:
            return

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
