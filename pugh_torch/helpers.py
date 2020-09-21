import re
import hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np


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
