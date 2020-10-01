import torch
import torch.nn.functional as F
from pugh_torch.losses import hetero_cross_entropy
import pytest
import numpy as np


@pytest.fixture
def pred():
    """
    (1, 4, 2, 3) logits
    """

    data = torch.FloatTensor(
        [
            [  # Batch 0
                [
                    [1.1, 0.5, 0.2],
                    [1.3, 0.1, 0.6],
                ],  # Class 0
                [
                    [0.7, 0.4, 1.5],
                    [6, 0.1, -2],
                ],  # Class 1
                [
                    [1.7, 0.25, -3],
                    [4, 0.9, 0.8],
                ],  # Class 2
                [
                    [1, 2, 3],
                    [3, 2, 1],
                ],  # Class 3
            ]
        ],
    )
    data.requires_grad_()
    return data


def test_hetero_cross_entropy_ce_only(pred):
    """Should behave as normal cross entropy when no superclass index is
    specified.
    """
    # (1,2,3)
    target = torch.LongTensor(
        [
            [
                [-2, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, True, True],
        ]
    )

    actual_loss = hetero_cross_entropy(pred, target, available, ignore_index=-2)
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()

    # Compute what the expected value should be
    pred_valid = torch.FloatTensor([[[0.5, 0.4, 0.25, 2], [0.6, -2, 0.8, 1],]]).permute(
        0, 2, 1
    )  # (1, 4, 2)
    target_valid = torch.LongTensor([[1, 1]])
    expected = F.cross_entropy(pred_valid, target_valid)  # This should be 3.0005

    assert np.isclose(actual_loss, expected.detach().numpy())


def test_hetero_cross_entropy_super_only_simple():
    pred = torch.Tensor([-1, 0, 1, 2]).reshape(1, 4, 1)  # logits
    target = torch.full((1, 1), -1, dtype=torch.long)
    available = torch.BoolTensor([True, True, False, False]).reshape(1, -1)

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()
    actual_loss = actual_loss.detach().numpy()

    assert np.isclose(0.9401897, actual_loss)


def test_hetero_cross_entropy_super_only(pred):
    target = torch.LongTensor(
        [
            [
                [-1, -1, -1],
                [-1, -1, -1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, False, False, False],
        ]  # Only class 0 is available.
    )

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 5.015609)


def test_hetero_cross_entropy_all_invalid(pred):
    """This primarily tests that when all invalid data is provided (i.e.
    the returned loss is 0) that the ``backward()`` method still works.
    """

    target = torch.LongTensor(
        [
            [
                [-2, -2, -2],
                [-2, -2, -2],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, True, True],
        ]
    )

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()
    assert actual_loss == 0


def test_hetero_cross_entropy_complete(pred):
    """Test both parts (ce_loss + super_loss) combined"""

    target = torch.LongTensor(
        [
            [
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, False, False],
        ]
    )

    pred_softmax = F.softmax(pred, dim=1)  # For inspecting/debugging purposes
    """
    Predicted classes (where X means doesn't matter AT ALL):
    tensor([[[2, 3, x],
             [x, x, 3]]])

    [0,0] -> 2 is the interesting one here.
        If the loss is working, this prediction should increase 2/3 class prob
        # pred - learning_rate * grad  should get us closer.
    """

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()  # This should always work

    pred_grad = pred.grad.detach().numpy()
    assert not pred_grad[:, :, 0, 2].any()
    assert not pred_grad[:, :, 1, 0].any()
    assert not pred_grad[:, :, 1, 1].any()

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 4.2314653)

    updated_pred = pred - (0.1 * pred.grad)
    updated_pred_softmax = F.softmax(updated_pred, dim=1)

    assert (
        updated_pred_softmax[0, 1, 0, 1] > pred_softmax[0, 1, 0, 1]
    )  # prediction should be more sure of 1 target
    assert (
        updated_pred_softmax[0, 1, 1, 2] > pred_softmax[0, 1, 1, 2]
    )  # prediction should be more sure of 1 target

    # The classes that are in the dataset should now have lower probabilities
    # for the pixel that's marked as unlabeled.
    assert updated_pred_softmax[0, 0, 0, 0] < pred_softmax[0, 0, 0, 0]
    assert updated_pred_softmax[0, 1, 0, 0] < pred_softmax[0, 1, 0, 0]


def test_hetero_cross_entropy_super_only_simple_alpha_near_zero():
    pred = torch.Tensor([-1, 0, 1, 2]).reshape(1, 4, 1)  # logits
    target = torch.full((1, 1), -1, dtype=torch.long)
    available = torch.BoolTensor([True, True, False, False]).reshape(1, -1)

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1, alpha=0.0001
    )
    actual_loss.backward()
    actual_loss = actual_loss.detach().numpy()

    assert np.isclose(0.9401897, actual_loss, rtol=0.01)


def test_hetero_cross_entropy_smoothing_complete_alpha_near_zero(pred):
    """Test both parts (ce_loss + super_loss) combined + label smoothing"""

    target = torch.LongTensor(
        [
            [
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, False, False],
        ]
    )

    pred_softmax = F.softmax(pred, dim=1)  # For inspecting/debugging purposes
    """
    Predicted classes (where X means doesn't matter AT ALL):
    tensor([[[2, 3, x],
             [x, x, 3]]])

    [0,0] -> 2 is the interesting one here.
        If the loss is working, this prediction should increase 2/3 class prob
        # pred - learning_rate * grad  should get us closer.
    """

    actual_loss = hetero_cross_entropy(
        pred,
        target,
        available,
        ignore_index=-2,
        super_index=-1,
        alpha=0.00001,
    )

    actual_loss.backward()  # This should always work

    pred_grad = pred.grad.detach().numpy()
    assert not pred_grad[:, :, 0, 2].any()
    assert not pred_grad[:, :, 1, 0].any()
    assert not pred_grad[:, :, 1, 1].any()

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 4.2314653, rtol=0.01)

    updated_pred = pred - (0.1 * pred.grad)
    updated_pred_softmax = F.softmax(updated_pred, dim=1)

    assert (
        updated_pred_softmax[0, 1, 0, 1] > pred_softmax[0, 1, 0, 1]
    )  # prediction should be more sure of 1 target
    assert (
        updated_pred_softmax[0, 1, 1, 2] > pred_softmax[0, 1, 1, 2]
    )  # prediction should be more sure of 1 target

    # The classes that are in the dataset should now have lower probabilities
    # for the pixel that's marked as unlabeled.
    assert updated_pred_softmax[0, 0, 0, 0] < pred_softmax[0, 0, 0, 0]
    assert updated_pred_softmax[0, 1, 0, 0] < pred_softmax[0, 1, 0, 0]
