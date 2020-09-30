from PIL import Image, ImageDraw, ImageFont
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import hashlib
import inspect
import numpy as np
import os
import re
import requests
import logging

from .exceptions import ShouldNeverHappenError


log = logging.getLogger(__name__)


def camel_to_snake(s):
    """Converts a camelCase or pascalCase string to snake_case

    Parameters
    ----------
    s : str
        Some camel/pascal case string to convert to snake case

    Returns
    -------
    str
        camel_case equivalent of the provided string.
    """

    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def compute_sha1(path, chunk_size=2 ** 20):
    """Computes the SHA1 hash of a file

    Parameters
    ----------
    path : str-like
        Path to file to compute the hash of.
    """

    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def compare_hash(expected, actual):
    """
    Raises
    ------
    pugh_torch.HashMismatchError
        If the hashes do not match.
    """

    if expected == actual:
        return

    raise HashMismatchError(f"{actual} doesn't match expected hash {expected}")


def add_text_under_img(
    img,
    text,
    font_size=None,
    min_font_size=10,
    font="DejaVuSansMono.ttf",
    bg="white",
    fg="black",
):
    """Rasterize and add text under an image.

    Based on:
        https://stackoverflow.com/a/4902713/13376237

    Parameters
    ----------
    img : numpy.ndarray or PIL.Image.Image
        (H,W,3) Image
    text : str
        Text to display under the image
    min_font_size : int
        Minimum font size to render. If the resulting text is wider than the
        passed in img, then the img will be resized.
        Ignored if ``font_size`` is provided.
    font_size : int
        If provided, uses this font size and doesn't auto-search for a font size.
    bg: str or tuple
        Background color of annotation. Defaults to white.
    fg: str or tuple
        Text color. Defaults to black.

    Returns
    -------
    numpy.ndarray
        Resulting annotated image. The image may be rescaled depending on
        ``min_font_size``.
    """

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if font_size is None:
        # Figure out biggest font that will fit.
        font_size = min_font_size
        font_obj = ImageFont.truetype(font, font_size)
        jump_size = 75
        while True:
            if font_obj.getsize_multiline(text)[0] < img.size[0]:
                font_size += jump_size
            else:
                jump_size = int(jump_size / 2)
                if jump_size == 0 or font_size == min_font_size:
                    break
                font_size -= jump_size
            font_obj = ImageFont.truetype(font, font_size)
            if jump_size <= 1:
                break

        if font_size < min_font_size:
            font_size = min_font_size
            font_obj = ImageFont.truetype(font, font_size)
    else:
        font_obj = ImageFont.truetype(font, font_size)

    # Check and resize img if the font doesn't fit
    font_dim = font_obj.getsize_multiline(text)
    if font_dim[0] > img.size[0]:
        new_w = font_dim[0]
        new_h = int(new_w / img.size[0] * img.size[1])
        img = img.resize((new_w, new_h))

    output = Image.new("RGB", (img.size[0], img.size[1] + font_dim[1]), color=bg)
    output.paste(img)

    draw = ImageDraw.Draw(output)
    draw.text((0, img.size[1]), text, font=font_obj, fill=fg)

    # Convert to a numpy object
    return np.array(output)


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str or Path-like, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will delete and re-download
        existing file when hash is specified but doesn't match.

    Returns
    -------
    pathlib.Path
        The file path of the downloaded file.
    """

    if path is None:
        path = Path.cwd()
    else:
        path = Path(path).expanduser()

    if path.is_dir():
        file_path = path / url.split("/")[-1]
    else:
        file_path = path
        path = path.parent
        path.mkdir(parents=True, exist_ok=True)

    if sha1_hash and file_path.exists():
        # Compute and compare the existing file to the hash
        actual_hash = compute_sha1(file_path)
        try:
            compare_hash(sha1_hash, actual_hash)
            return
        except HashMismatchError:
            print(f"Local file {file_path} hash mismatch. Re-downloading...")
            file_path.unlink()

    if overwrite or not file_path.exists():
        print(f"Downloading {file_path} from {url}...")
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get("content-length")
        with open(file_path, "wb") as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit="KB",
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)

        # Verify newly downloaded file's hash
        if sha1_hash:
            actual_hash = compute_sha1(file_path)
            compare_hash(sha1_hash, actual_hash)

    return file_path


def calling_scope(name, index=1, strict=True):
    """Gets an object from the calling scope.

    This uses a bunch of hacky stuff and may be fragile.

    Parameters
    ----------
    name : str
        Object in the calling scope to get.
    index : int
        How many frames to go up in the stack. Defaults to 1 (direct caller).
    strict: bool
        If ``True``, only search the specified frame's local and global scope.
        Otherwise, iterate up the stack until the object is found.

    Returns
    -------
    obj
       Object from caller scope.
    """

    name = str(name)
    frame = inspect.stack()[index][0]
    while True:
        if name in frame.f_locals:
            return frame.f_locals[name]
        elif name in frame.f_globals:
            return frame.f_globals[name]
        elif strict:
            raise KeyError(f'"{name}" not in calling scope')
        else:
            frame = frame.f_back
            if frame is None:
                raise KeyError(f'"{name}" not in calling scope')

    raise ShouldNeverHappenError


def to_obj(s, index=0):
    """Converts str to its respective object in caller's scope.

    This can be thought of converting the string into the object available
    in the caller's scope.

    Useful for specifying programmatic objects in Hydra configs.

    Example:
        # Assuming we are in a method where this is available
        assert self.foo.bar == to_obj("self.foo.bar")

    Parameters
    ----------
    s : obj or str
        Object to convert into it's callable equivalent.
        If this is already an object, it justs passes it through.
    index : int
        Scope to search, where 0 means the caller's scope, 1 is the caller's caller
        scope, etc.

    Returns
    -------
    callable
        Callable equivalent represented by the input.
    """

    if isinstance(s, str):
        components = s.split(".")
        root_str = components[0]

        output = calling_scope(root_str, index=index + 2)

        for component in components[1:]:
            output = getattr(output, component)

        return output
    else:
        # Identity pass-thru
        return s


@contextmanager
def working_dir(newdir):
    """
    Changes working directory and returns to previous on exit.

    Usage:
        with working_dir("my_experiment_path"):
            # do stuff within `my_experiment_path/`
        # Now we are back in the original working directory

    Parameters
    ----------
    new
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def most_recent_run(outputs_path, fmts=["%Y-%m-%d", "%H-%M-%S"]):
    """Get the most recent Hydra run folder.

    Parameters
    ----------
    outputs_path : pathlib.Path or str
        The root Hydra outputs directory
    fmts : list
        Optional list of string formats of how to interpret the folders.

    Returns
    -------
    pathlib.Path
        Most recent output path.
    """

    cur_path = Path(outputs_path).resolve()

    for fmt in fmts:
        fmt = str(fmt)
        times = [
            datetime.strptime(str(f.name), fmt)
            for f in cur_path.iterdir()
            if f.is_dir()
        ]
        cur_path = cur_path / max(times).strftime(fmt)

    return cur_path


def most_recent_checkpoint(outputs_path):
    """Get the most recent valid checkpoint path.

    Searches over all the subdirectories in `outputs_paths/` and returns
    the most recent found checkpoint path.

    Relies on ModelCheckpoint(save_last=True)

    Parameters
    ----------
    outputs_path : pathlib.Path or str
        The root Hydra outputs directory

    Raises
    ------
    FileNotFoundError
        If the most recent checkpoint can not be found.

    Returns
    -------
    pathlib.Path
        Most recent output path.
    """

    day_fmt = "%Y-%m-%d"
    time_fmt = "%H-%M-%S"

    outputs_path = Path(outputs_path).resolve()

    days = [
        datetime.strptime(str(f.name), day_fmt)
        for f in outputs_path.iterdir()
        if f.is_dir()
    ]
    days = sorted(days, reverse=True)
    for day in days:
        day_str = day.strftime(day_fmt)
        day_path = outputs_path / day_str
        times = [
            datetime.strptime(str(f.name), time_fmt)
            for f in day_path.iterdir()
            if f.is_dir()
        ]
        times = sorted(times, reverse=True)
        for time in times:
            time_str = time.strftime(time_fmt)
            experiment_id = f"{day_str}/{time_str}"
            experiment_path = outputs_path / experiment_id
            ckpt_path = experiment_path / "default/version_0/checkpoints/last.ckpt"
            if ckpt_path.is_file():
                return ckpt_path
            else:
                log.warn(
                    f'Recent experiment "{experiment_id}" did not have a checkpoint. Searching next run.'
                )

    raise FileNotFoundError("Could not find most recent checkpoint")


def plot_to_np(fig):
    """Converts a matplotlib.pyplot figure into a numpy array.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that you would like converted

    Returns
    -------
    numpy.ndarray
        Rasterized figure as an RGB-ordered numpy array
    """

    # Force a draw so we can grab the pixel buffer
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    output = np.array(fig.canvas.renderer.buffer_rgba())

    # Most stuff expects RGB, so we'll chop of the alpha channel
    output = output[..., :3]

    return output
