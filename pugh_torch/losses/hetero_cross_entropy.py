import torch
import torch.nn.functional as F


def hetero_cross_entropy(
    preds,
    targets,
    availables,
    *,
    super_index=0,
    ignore_index=-100,
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

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    # Make sure all the batch sizes are the same
    assert len(preds) == len(targets) == len(availables)

    super_hots = (
        ~availables.bool()
    )  # (B, C) where 1 means that class is a part of the superclass

    # Initialize the two components of our loss on the correct device
    ce_loss = preds.new([0]).float()
    ce_loss.requires_grad_()  # Just in case we don't actually add to this loss, ``backward()`` will still work
    super_loss = preds.new([0]).float()

    # Iterate over the batch dimension because the shapes are going to get jagged.
    for pred, target, super_hot in zip(preds, targets, super_hots):
        # pred - (C, H, W) float
        # target - (H, W) long
        # super_hot - (C,) bool
        inbound_mask = target != ignore_index  # (H, W)
        super_mask = target == super_index
        ce_mask = inbound_mask * ~super_mask

        # Apply CE to inbound mask, excluding the super_index
        if torch.any(ce_mask):
            ce_pred = pred[:, ce_mask]  # (C, n_valid)
            ce_target = target[ce_mask]  # (n_valid)
            ce_pred = ce_pred.transpose(0, 1)  # (n_valid, C)
            ce_loss = ce_loss + F.cross_entropy(ce_pred, ce_target)

        if torch.any(super_mask):
            super_pred = pred[:, super_mask]  # (C, n_pix)

            # Compute the softmax over all classes
            super_pred_exp = torch.exp(super_pred)
            super_loss = super_loss - torch.mean(
                torch.log(super_pred_exp[super_hot].sum(dim=0))
                - torch.log(super_pred_exp.sum(dim=0))
            )

    return ce_loss + super_loss
