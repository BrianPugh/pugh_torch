import os
import shutil
import argparse
import zipfile
from ...helpers import download

from .. import Dataset


class ADE20k(Dataset):
    DOWNLOAD_URLS = [
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),
    ]

    PAYLOAD_NAMES = ["ADEChallengeData2016.zip", "release_test.zip"]

    def download(self,):
        for url, checksum in self.DOWNLOAD_URLS:
            filename = download(url, path=self.path, overwrite=True, sha1_hash=checksum)

    def unpack(self,):
        """ Extract the downloaded zip files.
        No furth processing required.
        """

        for payload in self.PAYLOAD_NAMES:
            filename = self.path / payload
            with zipfile.ZipFile(filename,"r") as zip_ref:
                zip_ref.extractall(path=self.path)

    def __getitem__(self,):
        # TODO
        pass
