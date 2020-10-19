import torch
from pathlib import Path
from . import ROOT_MODELS_PATH
from ..utils.io import gdrive_download


def load_state_dict_from_url(url, local, map_location=None, progress=True, force=False):
    """

    Parameters
    ----------
    url : str
    local : str-like
        Local path of where to save or check for cached file.
    force : bool
        Force the redownload, even if the local file exists.
    """

    url = str(url)
    local = Path(local)

    if url.startswith("https://drive.google.com"):
        # Custom solution
        if not local.is_file() or force:
            downloaded_path = gdrive_download(url, local)
            assert downloaded_path == local
        state_dict = torch.load(local)
    else:
        # Let torch.hub handle everything
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            model_dir=None,
            map_location=None,
            progress=True,
            check_hash=False,
            file_name=None,
        )

    return state_dict
