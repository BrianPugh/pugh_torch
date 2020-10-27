import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pugh_torch as pt
from math import sqrt, ceil, floor


def unnormalize_coords(x, shape):
    """
    Parameters
    ----------
    x : numpy.ndarray
        (N, 2) Array representing (x,y) coordinates in range [-1, 1]
    shape : tuple of length 2
        (H, W) of the image
    """
    if isinstance(x, np.ndarray):
        x = x.copy()
        x += 1
        x /= 2
        x[..., 0] *= shape[1] - 1
        x[..., 1] *= shape[0] - 1
        x = np.round(x).astype(np.int)
    else:
        x = x.detach().clone()
        x += 1
        x /= 2
        x[..., 0] *= shape[1] - 1
        x[..., 1] *= shape[0] - 1
        x = torch.round(x).long()
    return x


def rasterize_image(coords, rgb_vals, shape, normalized=True):
    """
    Parameters
    ----------
    coords : torch.Tensor
        (B, num_sample, 2)
    rgb_vals : torch.Tensor
        (B, num_sample, 2) If normalized==True, then imagenet-normalized
        black pixels will be used as the fill.
    shape : tuple
        Tuple of length 2, the (h,w) of the output canvas
    normalized : bool
        If rgb_vals is imagenet normalized, set this to ``True`` so that the
        fill value is normalized-black instead of (0,0,0).

    Returns
    -------
    torch.Tensor
         (B, 3, H, W)
    """
    bsize, num_sample = coords.shape[:2]
    device = rgb_vals.device

    fill_value = 0
    canvas = torch.zeros(bsize, 3, shape[0], shape[1], device=device)
    if normalized:
        fill_value = torch.Tensor(
            np.array([-2.11790393, -2.03571429, -1.80444444]).reshape(1, 3, 1, 1)
        ).to(device)
        canvas += fill_value

    bselect = torch.arange(canvas.size(0), dtype=torch.long, device=device)
    coords_unnormalized = unnormalize_coords(coords, shape)
    canvas[
        bselect[..., None], :, coords_unnormalized[..., 1], coords_unnormalized[..., 0]
    ] = rgb_vals

    return canvas


def rasterize_montage(coords, gt_rgb_vals, pred_rgb_vals, gt_imgs, normalized=True):
    h, w = gt_imgs.shape[2:]
    gt_rasterize = rasterize_image(coords, gt_rgb_vals, (h, w), normalized=normalized)
    pred_rasterize = rasterize_image(
        coords, pred_rgb_vals, (h, w), normalized=normalized
    )
    montage = torch.cat((gt_imgs.cpu(), gt_rasterize.cpu(), pred_rasterize.cpu()), -1)
    return montage


class ImageNetSample(torch.utils.data.Dataset):
    def __init__(self, *args, num_sample=4096, **kwargs):
        self.imagenet = pt.datasets.classification.ImageNet(*args, **kwargs)
        self.num_sample = num_sample

    def __len__(self):
        return len(self.imagenet)

    @property
    def val_src_pts(self):
        if hasattr(self, "_val_src_pts"):
            return self._val_src_pts

        # May result in a little smaller bathc size
        n_grid = int(floor(sqrt(self.num_sample)))

        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, n_grid, 1), torch.arange(0, n_grid, 1)
        )
        src_pts = torch.stack(
            (grid_x.reshape(-1), grid_y.reshape(-1)),
            dim=-1,
        )

        self._val_src_pts = src_pts / float(n_grid - 1)

        return self._val_src_pts

    def __getitem__(self, index):
        # This img has already be transformed, just need to sample
        # img has shape (3, H, W)
        img, _ = self.imagenet[index]

        if self.imagenet.split == "train":
            coord_normalized, rgb_values = _sample_img(img[None], self.num_sample)
        elif self.imagenet.split == "val":
            coord_normalized, rgb_values = _sample_img(
                img[None], self.num_sample, entropy=self.val_src_pts
            )
        else:
            raise NotImplementedError

        return coord_normalized, rgb_values, img


def _sample_img(img, num_sample=None, entropy=None):
    """
    Parameters
    ----------
    img : torch.Tensor
        (B, C, H, W) image to sample from
    num_sample : int
        Number of locations to sample. Ignored if ``entropy`` is provided.
    entropy : torch.Tensor
        (num_sample, 2) If provided, in range [0,1]. Sampling coordinates are derived from here.

    Returns
    -------
    coord_normalized : torch.Tensor
        (num_sample, 2) where each row is (x, y)
    rgb_values : torch.Tensor
        (num_sample, 3) where each row is (r, g, b)
    """

    if entropy is None:
        # Select random coordinates in range [-1, 1]
        entropy = torch.rand(num_sample, 2)
    coord_normalized = 2 * entropy - 1

    # interpolate at those points
    rgb_values = F.grid_sample(
        img, coord_normalized[None, :, None, :], align_corners=True
    )
    rgb_values = rgb_values[0, ..., 0].transpose(0, 1)
    return coord_normalized, rgb_values


class SingleImageDataset(torch.utils.data.IterableDataset):
    """Generates sampled:
        * x: Shape (N, 2) where each row is (x,y) normalized coordinates in range [-1,1]
        * y: Shape (N,3) where each row is RGB color at that sampled point

    We are defining that pixels are determined by their topleft corner.
    """

    def __init__(self, path, batch_size, shape=None, mode="train", normalize=True):
        """
        path : path-like or RGB numpy.ndarray
        """

        super().__init__()

        self.mode = mode.lower()

        if isinstance(path, np.ndarray):
            self.img = path.copy()
        else:
            # Assume that ``path`` is path-like
            self.img = cv2.imread(str(path))[..., ::-1]

        # Resize the image
        if shape is None:
            shape = self.img.shape[:2]
        else:
            assert len(shape) == 2
            self.img = cv2.resize(
                self.img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
            )
        self.shape = np.array(shape)  # (H, W) i.e. (y, x)

        # Normalize the image
        if normalize:
            self.img = pt.transforms.imagenet.np_normalize(self.img / 255)
        self.img = torch.Tensor(self.img)

        self.flat_img = self.img.reshape((-1, 3))
        self.img = self.img[None]
        self.img = self.img.permute(0, 3, 1, 2)  # (B, C, H, W)

        nx, ny = self.img.shape[3], self.img.shape[2]
        # (X, Y)
        meshgrid = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
        self.src_pts = torch.Tensor((meshgrid[0].reshape(-1), meshgrid[1].reshape(-1)))
        self.src_pts_normalized = (
            2 * self.src_pts[0] / (self.shape[1] - 1) - 1,
            2 * self.src_pts[1] / (self.shape[0] - 1) - 1,
        )

        self.batch_size = batch_size
        self.n_pixels = self.shape[0] * self.shape[1]
        self.pos = 0

    def __len__(self):
        return ceil(self.n_pixels / self.batch_size)

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        if self.pos >= self.n_pixels:
            raise StopIteration

        if self.mode == "train":
            coord_normalized, rgb_values = _sample_img(self.img, self.batch_size)
            self.pos += self.batch_size
        elif self.mode == "val":
            # Deterministcally return every pixel value in order
            # Last batch size may be smaller
            upper = self.pos + self.batch_size
            if upper > self.n_pixels:
                upper = self.n_pixels

            coord_normalized = torch.cat(
                (
                    self.src_pts_normalized[0][self.pos : upper, None],
                    self.src_pts_normalized[1][self.pos : upper, None],
                ),
                dim=1,
            )
            rgb_values = self.flat_img[self.pos : upper]
            self.pos = upper
        else:
            raise NotImplementedError

        return coord_normalized, rgb_values
