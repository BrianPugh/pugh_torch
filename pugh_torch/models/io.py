import torch
from pathlib import Path
from . import ROOT_MODELS_PATH
from ..utils.io import gdrive_download
import logging

log = logging.getLogger(__name__)


def load_state_dict_from_url(
    url, local=None, map_location=None, progress=True, force=False, **kwargs
):
    """

    Parameters
    ----------
    url : str
    local : str-like
        Local path of where to save or check for cached file.
        If relative, is relative to the torch.hub directory.
    force : bool
        Force the redownload, even if the local file exists.
    """

    url = str(url)
    local = Path(local)

    gdrive_prefix = "https://drive.google.com"
    if url.startswith(gdrive_prefix):
        assert local is not None, f"Must specify path for gdrive files."

        local = Path(torch.hub.get_dir()) / local
        # Custom solution
        if not local.is_file() or force:
            downloaded_path = gdrive_download(url, local)
            assert downloaded_path == local
        state_dict = torch.load(local, map_location=map_location)
    else:
        if "drive.google.com" in url:
            log.warning(
                f'URL {url} contaings "drive.google.com", suggesting a google drive link. Google drive links MUST start with "{gdrive_prefix}"'
            )
        # Let torch.hub handle everything
        if local is None:
            model_dir = None
            file_name = None
        else:
            if local.suffix:  # Has extension
                model_dir = Path(torch.hub.get_dir()) / local.parent
                file_name = local.name
            else:
                # Assume it was meant to be a directory
                model_dir = Path(torch.hub.get_dir()) / local
                model_dir.mkdir(parents=True, exist_ok=True)
                file_name = None  # use whatever name torch.hub gives us

        if force and file_name is not None:
            # Delete the local file if it exists to force redownload.
            local_file = model_dir / file_name
            try:
                local_file.unlink()
            except FileNotFoundError:
                pass

        # TODO: force only works if an explicit local file path is provided.
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            model_dir=str(model_dir),
            map_location=map_location,
            progress=progress,
            check_hash=False,
            file_name=file_name,
        )

    return state_dict
