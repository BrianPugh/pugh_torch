""" Demo application to run data through your network.
"""

import argparse
from pathlib import Path
import cv2
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import MyModel as Model
from time import time


def parse_args():
    """ Parse CLI arguments into an object and a dictionary """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=Path, default="data/cat.jpg", help="Input image to run"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="output.png",
        help="Where to save the output prediction.",
    )
    parser.add_argument(
        "--model", "-m", type=Path, default=None, help="Path to model to load."  # TODO
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Don't run the model on without GPU."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load network and weights
    model = Model.load_from_checkpoint(args.model)
    if not args.no_gpu:
        model.cuda()
    model.eval()

    # TODO load data
    img = cv2.imread(str(args.input))[..., ::-1]  # Network expects RGB data

    albumentations_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    # TODO input transform
    # TODO: make sure the input is (1, 3, H, W)

    # Feed data through network
    pred = model(img)

    # TODO: post-processing on prediction.

    # Save inference result
    cv2.imwrite(str(args.output), pred)


if __name__ == "__main__":
    t_start = time()

    main()

    t_end = time()
    print(f"Loaded and ran network in {t_end - t_start} seconds.")
