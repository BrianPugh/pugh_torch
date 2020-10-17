import gdown
from pathlib import Path


class GDriveDownloadError(IOError):
    """Failed to download file from google drive"""


def gdrive_download(url, path):
    """Download a file from google drive.

    Parameters
    ----------
    url : str-like
        Public google drive link.
    path : str-like
        Path to where the file should be downloaded.
        If a directory (determined if extensionless), it will be created if
        necessary and use the name from google drive.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.
    """

    url = str(url)
    path = Path(path)

    assert url.startswith("https://drive.google.com")

    if "https://drive.google.com/uc?id=" not in url:
        # Handles new google drive link format. Can be removed after:
        #    https://github.com/wkentaro/gdown/pull/76
        # is merged.
        url = (
            "https://drive.google.com/uc?id="
            + url.split("https://drive.google.com/file/d/")[-1]
        )

    # Remove the "view?usp=sharing" part if it exists
    suffix = "/view?usp=sharing"
    if url.endswith(suffix):
        url = url[: -len(suffix)]

    # Handle local path parsing
    is_dir = not path.suffix
    gdown_path = str(path)
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
        gdown_path += "/"

    # TODO maybe use cached_download
    local_path = gdown.download(url, gdown_path)

    if local_path is None:
        raise GDriveDownloadError

    return Path(local_path)
