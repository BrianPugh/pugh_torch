"""
Based on: 
        https://discuss.pytorch.org/t/batch-index-select/9115/11
"""

import torch


def batch_index_select(input, dim, index):
    """batch version of ``torch.index_select``.

    Returns a new tensor which indexes the input tensor along dimension ``dim``
    using the corresponding entries in ``index`` which is a ``LongTensor``.

    The returned tensor has the same number of dimensions as the original tensor (input).
    The ``dim``th dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.

    Parameters
    ----------
    input : torch.Tensor
        (B, ..)the input tensor.
    dim : int
        the dimension in which we index. Must be ``>0`` since we use the ``0``th
        index as the batch.
        May be negative.
    index : torch.LongTensor
        (B, N) the 1-D tensor containing the indices to index per batch

    Returns
    -------
    torch.Tensor
        (B, ...) tensor that matches the input dimensions, except the ``dim``th
        dimension now has length ``N``.

        NOTE: does NOT use the same storage as ``input`` Tensor
    """

    if dim < 0:
        dim = input.ndim + dim

    assert dim > 0, "Cannot index along batch dimension."
    assert (
        input.shape[0] == index.shape[0]
    ), "input and index must have same batch dimension."

    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
