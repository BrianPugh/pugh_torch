import torch
import pytorch_lightning as pl
from pathlib import Path
from shutil import copyfile

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    """
    def _do_check_save(
        self,
        filepath: str,
        current: torch.Tensor,
        epoch: int,
        trainer,
        pl_module,
    ):
        """ Additionally copies the best checkpoint to "best.ckpt" in the checkpoint
        directory.
        """

        res = super()._do_check_save(filepath, current, epoch, trainer, pl_module)

        best_model_path = Path(self.best_model_path).resolve()
        dst = best_model_path.parent / 'best.ckpt'
        copyfile(best_model_path, dst)

        return res
