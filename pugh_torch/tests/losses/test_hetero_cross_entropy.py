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

    return torch.FloatTensor([
            [ # Batch 0
                [ # Class 0
                    [1.1, 0.5, 0.2],
                    [1.3, 0.1, 0.6],
                ],
                [ # Class 1
                    [0.7, 0.4, 1.5],
                    [6, 0.1, -2],
                ],
                [ # Class 2
                    [1.7, 0.25, -3],
                    [4, 0.9, 0.8],
                ],
                [ # Class 3
                    [1, 2, 3],
                    [3, 2, 1],
                ],
            ]
        ])


def test_hetero_cross_entropy_ce_only(pred):
    """ Should behave as normal cross entropy when no superclass index is
    specified.
    """
    # (1,2,3)
    target = torch.LongTensor([
            [ # Height
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ])
    available = torch.BoolTensor([
        [True, True, True, True],
        ])

    actual_loss = hetero_cross_entropy(pred, target, available, ignore_index=-2, super_index=-1)
    actual_loss = actual_loss.numpy()

    # Compute what the expected value should be
    pred_valid = torch.FloatTensor([
            [
                [0.5, 0.4, 0.25, 2],
                [0.6, -2, 0.8, 1],
            ]
        ]).permute(0, 2, 1)  # (1, 4, 2)
    target_valid = torch.LongTensor(
            [[1, 1]]
            )
    expected = F.cross_entropy(pred_valid, target_valid)  # This should be 3.0005

    assert np.isclose(actual_loss, expected.numpy())
