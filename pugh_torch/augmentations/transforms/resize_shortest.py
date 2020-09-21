from albumentations.core.transforms_interface import DualTransform
import albumentations.augmentations.functional as F

import cv2

__all__ = ["ResizeShortest"]


def _compute_new_shape(length, height, width, trunc=True):
    """Computes the new target shape that results in the shortest side length.

    Parameters
    ----------
    length : float
        Length of the resulting shortest side
    height : float
        Image height
    width : float
        Image width

    Returns
    -------
    int or float
        New height
    int or float
        New width
    """

    if height < width:
        new_height = length
        new_width = width * length / height
    elif width < height:
        new_width = length
        new_height = height * length / width
    else:
        new_width = length
        new_height = length

    if trunc:
        new_height = int(new_height)
        new_width = int(new_width)

    return new_height, new_width


class ResizeShortest(DualTransform):
    """Resize the input so that the shortest side has the given pixel dimension."""

    def __init__(self, length, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        """
        Parameters
        ----------
        length : int
            Length in pixels of the resulting shortest side of the image.
        """

        super().__init__(always_apply, p)
        self.length = length
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        new_height, new_width = _compute_new_shape(
            self.length, params["rows"], params["cols"]
        )
        return F.resize(
            img, height=new_height, width=new_width, interpolation=interpolation
        )

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant since coordinates are normalized.
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        """
        Keypoints are not normalized and need their xy rescaled.
        """

        new_height, new_width = _compute_new_shape(
            self.length, params["rows"], params["cols"]
        )
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        # TODO
        return ("length", "interpolation")


# Alias with similar naming to albumentations.augmentations.transforms.LongestMaxSize
ShortestMaxSize = ResizeShortest
