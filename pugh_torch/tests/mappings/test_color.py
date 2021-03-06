import pytest
import numpy as np
import pugh_torch as pt


def test_turbo_auto_range():
    x = np.arange(20).reshape(4, 5)

    actual = pt.mappings.turbo(x)

    assert actual.shape == (4, 5, 3)

    expected = np.array(
        [
            [
                [0.18995, 0.07176, 0.23217],
                [0.24234, 0.21941, 0.56942],
                [0.27103, 0.35926, 0.81156],
                [0.27543, 0.50115, 0.96594],
                [0.23288, 0.62923, 0.99202],
            ],
            [
                [0.13886, 0.76279, 0.8955],
                [0.09267, 0.86554, 0.7623],
                [0.17377, 0.94053, 0.61938],
                [0.35043, 0.98477, 0.45002],
                [0.56026, 0.99873, 0.28623],
            ],
            [
                [0.70553, 0.97255, 0.21032],
                [0.84133, 0.89986, 0.20926],
                [0.93909, 0.80439, 0.22744],
                [0.99163, 0.68408, 0.20706],
                [0.99153, 0.54036, 0.1491],
            ],
            [
                [0.94977, 0.37729, 0.07905],
                [0.88066, 0.25334, 0.03521],
                [0.77377, 0.15028, 0.01148],
                [0.64223, 0.0738, 0.00401],
                [0.4796, 0.01583, 0.01055],
            ],
        ]
    )

    assert np.allclose(actual, expected)


def test_turbo_many_dim():
    shape = (5, 5, 5, 5, 5)
    x = np.random.rand(*shape)
    actual = pt.mappings.turbo(x)
    assert actual.shape == (*shape, 3)
