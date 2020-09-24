import cv2
import numpy as np
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
            with zipfile.ZipFile(filename, "r") as f:
                f.extractall(path=self.path)

    def __getitem__(
        self,
        index,
    ):
        img = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)[..., ::-1]  # Result should be RGB
        img = img.astype(np.float32) / 255  # Images are supposed to be float in range [0, 1]
        mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=img, mask=mask)

        return transformed['image'], transformed['mask']

    def __len__(self):
        return len(self.images)

    def _populate_ade20k_pairs(self):
        """Populates the attributes:
        * images - list of Paths to images
        * masks - list of Paths to semantic segmentation masks
        """

        self.images, self.masks = [], []

        path = self.path / "ADEChallengeData2016"

        if self.split == "train":
            img_folder = path / "images/training"
            mask_folder = path / "annotations/training"
            expected_len = 20210
        elif self.split == "val":
            img_folder = path / "images/validation"
            mask_folder = path / "annotations/validation"
            expected_len = 2000
        else:
            raise ValueError(f"split must be train or val; got \"{self.split}\"")

        potential_images = img_folder.glob("*.jpg")
        for potential_image in potential_images:
            mask_path = mask_folder / (potential_image.stem + ".png")
            if mask_path.is_file():
                self.images.append(potential_image)
                self.masks.append(mask_path)
            else:
                print(f"Cannot find mask for {potential_image}")

        assert (
            len(self.images) == expected_len
        ), f"Expected {expected_len} exemplars, only found {len(self.images)}"
