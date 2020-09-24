import os
import shutil
import argparse
import zipfile
from ...helpers import download

from .. import Dataset


class ADE20K(Dataset):
    DOWNLOAD_URLS = [
        (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            "219e1696abb36c8ba3a3afe7fb2f4b4606a897c7",
        ),
        (
            "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip",
            "e05747892219d10e9243933371a497e905a4860c",
        ),
    ]

    PAYLOAD_NAMES = ["ADEChallengeData2016.zip", "release_test.zip"]

    def __init__(self, split="train", *, transform=None, **kwargs):
        super().__init__(split=split, transform=transform)
        self._populate_ade20k_pairs()

    def download(
        self,
    ):
        for url, checksum in self.DOWNLOAD_URLS:
            filename = download(url, path=self.path, overwrite=True, sha1_hash=checksum)

    def unpack(
        self,
    ):
        """Extract the downloaded zip files.
        No furth processing required.
        """

        for payload in self.PAYLOAD_NAMES:
            filename = self.path / payload
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(path=self.path)

    def __getitem__(
        self,
    ):
        raise NotImplementedError

    def _populate_ade20k_pairs():
        """Populates the attributes:
        * images - list of Paths to images
        * masks - list of Paths to semantic segmentation masks
        """

        self.images, self.masks = [], []

        path = self.path / "ADEChallengeData2016"

        if split == "train":
            img_folder = path / "images/training"
            mask_folder = path / "annotations/training"
            expected_len = 20210
        elif split == "val":
            img_folder = path / "images/validation"
            mask_folder = path / "annotations/validation"
            expected_len = 2000
        else:
            raise ValueError("split must be train or val")

        potential_images = img_folder.glob("*.jpg")
        for potential_image in potential_images:
            mask_path = mask_folder / (potential_image.stem + ".png")
            if mask_path.isfile():
                self.images.append(potential_image)
                self.masks.append(mask_path)
            else:
                print(f"Cannot find mask for {potential_image}")

        assert (
            len(self.images) == expected_len
        ), f"Expected {expected_len} exemplars, only found {len(self.images)}"
