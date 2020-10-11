import torch
from .tensorboard_base import TensorBoardCallback


class TensorBoardAddClassification(TensorBoardCallback):
    """Adds the rgb, ground truth label, and the network prediction
    to tensorboard.
    """

    def __init__(self, *, classes=None, **kwargs):
        """
        Parameters
        ----------
        classes : list
            If provided, the predictions will contain the appropriate annotation.
            `len(classes)`` should equal the number of networ output channels.
            If this is available under ``trainer``.
        """
        super().__init__(**kwargs)
        self.classes = classes

    def on_train_start(self, trainer, pl_module):
        if self.classes is None:
            # Attempt to initialize the ``classes`` attributes from ``dataset.classes``
            try:
                self.classes = trainer.train_dataloader.dataset.classes
            except AttributeError:
                pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % self.logging_batch_interval != 0:
            return

        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        # pick the last batch and labels
        x, y = batch[:2]
        y = y.cpu().numpy()

        logits = pl_module.last_logits
        preds = torch.argmax(logits, dim=-1)
        preds = preds.cpu().numpy()

        assert preds.ndim == 1

        labels = []
        for pred, gt in zip(preds, y):
            pred_class = "N/A"
            gt_class = "N/A"

            try:
                pred_class = self.classes[pred]
            except:
                pass

            try:
                gt_class = self.classes[gt]
            except:
                pass

            labels.append(f"Truth: {gt} ({gt_class})\nPred: {pred} ({pred_class})")

        trainer.logger.experiment.add_rgb(
            "train/output",
            x,
            global_step=trainer.global_step,
            labels=labels,
            **self.logging_kwargs,
        )
