import torch
import torch.nn.functional as F
import numpy as np


def _format_weight(weight, shape, device):
    if isinstance(weight, np.ndarray):
        return torch.Tensor(weight).to(device)
    elif isinstance(weight, torch.Tensor):
        return weight.to(device)
    elif isinstance(weight, (int, float)):
        return torch.full(shape, weight, device=device, dtype=torch.float)
    else:
        raise ValueError(f"unknown weight type {type(weight)}")


def hetero_cross_entropy(
    preds,
    targets,
    availables,
    *,
    super_index=0,
    ignore_index=-100,
    alpha=0,
    weight=1,
):
    """Cross Entropy Loss for Heterogenous Datasets.

    Basically a cross-entropy-loss for when you are training on multiple
    datasets where some of the datasets may have some of the classes labeled
    as a single superset (for example, background).

    Wordy logic:
        1. Lets say Dataset A has classes ["cat", "dog", "bird"].
        2. Lets say Dataset B has classes ["cat",              , "other"].
           i.e. Dataset B is missing class "dog" and "bird", and those pixels
           fall into the "other" category.
        3. For Dataset A, ``hetero_cross_entropy`` is exactly the same as
           normal cross entropy loss.
        4. For Dataset B, a "dog" will be labeled into the "other" category,
           but we don't want to penalize the network for predicting dog,
           because it could be correct.  However, we absolutely know that
           the network should not be predicting "cat".
        5. To solve this on dataset B, we apply cross entropy loss over the
           pixels not in the "other" class in the target. In addition to this,
           we want to maximize the probability that the network guessed either
           "dog" or "bird" for the pixels labeled as "other" in the target.

    Described in "Combining Heterogeneously Labeled Datasets For Training
    Segmentation Networks"
        https://arxiv.org/pdf/1807.08935.pdf

    TODO: this could be extended to multiple supersets, but the API might
    get complicated, and I personally don't have a use yet.

    Parameters
    ----------
    preds : torch.Tensor
        (N, C, ...) where C is the number of classes.
        Typically this is the **logits** coming out of your network.
    targets : torch.Tensor
        (N, ...) T
    availables : torch.Tensor
        (N, C) One-hot vector of which classes are availble from the dataset
        for this exemplar.
    ignore_index : int
        Pixels with this target value will be not contribute to the loss what
        so ever. Typically this value might be what you pad the target
        segmentation image with if you are performing random crops.
        May be negative.
    super_index : int
        Pixels with this target value will be encouraged to be of classes not
        in the dataset it came from. Typically this is a "background", "unknown",
        or "other" class.
        May be negative.
    alpha : float
        If nonzero, enables label smoothing. This will divide up
        the smoothing weight amongst the superclass labels such that the total
        weight is equivalent to a single class not in the superclass.
    weight : float or numpy.ndarray or torch.Tensor
        (N, C, ...)
        If C==1, will broadcast across all classes. Applied AFTER the ``log``
        operation, but before the reduction.

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    # Make sure all the batch sizes are the same
    assert len(preds) == len(targets) == len(availables)

    weights = _format_weight(weight, preds.shape, preds.device)
    del weight

    super_hots = (
        ~availables.bool()
    )  # (B, C) where 1 means that class is a part of the superclass

    # Initialize the two components of our loss on the correct device
    ce_loss = preds.new([0]).float()
    ce_loss.requires_grad_()  # Just in case we don't actually add to this loss, ``backward()`` will still work
    super_loss = preds.new([0]).float()

    # Iterate over the batch dimension because the shapes are going to get jagged.
    for i, (pred, target, super_hot) in enumerate(zip(preds, targets, super_hots)):
        # pred - (C, H, W) float
        # target - (H, W) long
        # super_hot - (C,) bool
        inbound_mask = target != ignore_index  # (H, W)
        super_mask = target == super_index  # (H, W)
        ce_mask = inbound_mask * ~super_mask  # (H, W)

        if weights.shape[0] == 1:
            weight = weights[0]
        else:
            weight = weights[i]

        num_super_classes = super_hot.sum().float()
        num_non_super_classes = (~super_hot).sum().float()

        # Superclass loss
        if torch.any(super_mask) and num_super_classes > 0:

            def compute_super_loss():
                super_weight = weight[:, super_mask]  # (1 or C, n_pix)
                super_pred = pred[:, super_mask]  # (C, n_pix)

                super_pred_exp = torch.exp(super_pred)

                super_numerator = torch.log(super_pred_exp[super_hot].sum(dim=0))
                denominator = torch.log(super_pred_exp.sum(dim=0))

                super_nll = -(super_weight * (super_numerator - denominator)).mean()

                if alpha == 0:
                    exemplar_super_loss = super_nll
                else:
                    # Handles the case where a superclass label should really
                    # be not in the superclass
                    super_smooth_weight = (1 - alpha) + (alpha / 2)

                    non_super_numerator = torch.log(
                        super_pred_exp[~super_hot].sum(dim=0)
                    )
                    non_super_nll = -(
                        super_weight * (non_super_numerator - denominator)
                    ).mean()

                    exemplar_super_loss = (super_smooth_weight * super_nll) + (
                        (alpha / 2) * non_super_nll
                    )
                return exemplar_super_loss

            super_loss = super_loss + compute_super_loss()

        # Apply CE to inbound mask, excluding the super_index
        if torch.any(ce_mask):

            def compute_ce_loss():
                ce_weight = weight[:, ce_mask]  # (1 or C, n_pix)
                ce_pred = pred[:, ce_mask]  # (C, n_valid)
                ce_target = target[ce_mask]  # (n_valid)

                if alpha == 0 or num_super_classes == 0:
                    ce_log_softmax = F.log_softmax(ce_pred, 0)
                    ce_nll_loss = F.nll_loss(
                        ce_log_softmax.transpose(0, 1), ce_target, reduction="none"
                    )
                    return (ce_weight * ce_nll_loss).mean()
                else:
                    ce_pred_exp = torch.exp(ce_pred)  # (C, n_valid)

                    # Handles the case where a nonsuperclass label should really
                    # be in the superclass
                    uniform_weight = alpha / (num_non_super_classes + 1)

                    # This is going to be a lil numerically unstable
                    super_numerator = torch.log(
                        ce_pred_exp[super_hot].sum(dim=0)
                    )  # (n_valid)
                    denominator = torch.log(ce_pred_exp.sum(dim=0))  # (n_valid,)

                    if ce_weight.shape[0] == 1:
                        ce_weight_super = ce_weight
                    else:
                        ce_weight_super = ce_weight[super_hot]

                    super_nll = -(
                        uniform_weight
                        * ce_weight_super
                        * (super_numerator - denominator)
                    ).mean()

                    # Construct the smoothed target label
                    one_hot = F.one_hot(
                        ce_target, num_classes=ce_pred.shape[0]
                    ).transpose(
                        0, 1
                    )  # (C, n_valid)
                    one_hot = one_hot[~super_hot]  # (C_nonsuper, n_valid)
                    uniform = torch.full_like(
                        one_hot, uniform_weight, dtype=torch.float
                    )  # (C_nonsuper, )
                    smooth_target = (
                        1 - alpha
                    ) * one_hot + uniform  # (C_nonsuper, n_valid)

                    non_super_log_probs = (
                        ce_pred[~super_hot] - denominator
                    )  # (C_nonsuper, n_valid)

                    if ce_weight.shape[0] == 1:
                        ce_weight_non_super = ce_weight
                    else:
                        ce_weight_non_super = ce_weight[~super_hot]

                    # non_super_log_probs = F.log_softmax(ce_pred[~super_hot], dim=0)  # (C_nonsuper, n_valid)
                    non_super_nll = (
                        -(ce_weight_non_super * smooth_target * non_super_log_probs)
                        .sum(dim=0)
                        .mean()
                    )  # Average over all pixels
                    return non_super_nll + super_nll

            ce_loss = ce_loss + compute_ce_loss()

    return ce_loss + super_loss
