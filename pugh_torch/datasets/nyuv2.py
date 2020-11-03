import h5py
import numpy as np
from . import Dataset
from ..helpers import download


class NYUv2(Dataset):
    DOWNLOAD_URL = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    PAYLOAD_NAME = "nyu_depth_v2_labeled.mat"

    available_types = set(["rgb", "depth", "instances", "labels"])

    fx = 5.1885790117450188e02
    fy = 5.1946961112127485e02
    cx = 3.2558244941119034e02
    cy = 2.5373616633400465e02

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    K4 = np.array([fx, fy, cx, cy])

    def __init__(
        self, *args, raw_depth=False, types=["rgb", "depth"], transform=None, **kwargs
    ):
        """
        Data Description
        ----------------
        rgb : np.array uint8
            Images in RGB order
        depth : np.array float32
            Depth in meters

        Parameters
        ----------
        raw_depth : bool
            Return the depth data before invalid areas were infilled.
            Defaults to ``False``.
        types : list of str
            Data types to return.
        """

        assert (
            transform is None
        ), f"Transforms aren't currently supported for nyuv2, it is recommend to subclass NYUv2 and add custom transform handling"

        super().__init__(*args, **kwargs)
        self.transform = None

        assert set(types).issubset(self.available_types)
        self.types = types

        self.raw_depth = raw_depth
        self.data = data = h5py.File(self.path / self.PAYLOAD_NAME)

        if self.split == "train":
            min_idx = 0
            max_idx = 1400
        else:
            min_idx = 1401
            max_idx = None

        # All data is in format (b, c, w, h) or (b, w, h)

        self.rgb = data["images"][min_idx:max_idx]
        if raw_depth:
            self.depth = data["rawDepths"][min_idx:max_idx]
        else:
            self.depth = data["depths"][min_idx:max_idx]
        self.instances = data["instances"][min_idx:max_idx]
        self.labels = data["labels"][min_idx:max_idx]

        self.classes = data["names"]
        self.class_to_idx = data["namesToIds"]

    def download(self):
        download(self.DOWNLOAD_URL, path=self.path / self.PAYLOAD_NAME, overwrite=True)

    def unpack(
        self,
    ):
        """ No unpacking necessary """

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        output = []

        for t in self.types:
            datum = getattr(self, t)[index]

            # convert (..., W, H) to (..., H, W)
            datum = datum.swapaxes(-1, -2)
            output.append(datum)

        return tuple(output)
