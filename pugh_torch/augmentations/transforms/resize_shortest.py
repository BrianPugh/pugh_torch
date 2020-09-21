from albumentations.core.transforms_interface import DualTransform
import albumentations.augmentations.functional as F

import cv2


class ResizeShortest(DualTransform):
    """ Resize the input so that the shortest side has the given pixel dimension.
    """

    def __init__(self, length, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        """
        Parameters
        ----------
        length : int
            Length in pixels of the resulting shortest side of the image.
        """

        super(Resize, self).__init__(always_apply, p)
        import ipdb as pdb; pdb.set_trace()
        self.length = length
        # TODO

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        # TODO
        return F.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # TODO
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        # TODO
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        # TODO
        return ("height", "width", "interpolation")
