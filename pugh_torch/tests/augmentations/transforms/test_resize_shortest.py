import pytest
import numpy as np
import pugh_torch as pt
import albumentations as A


def test_compute_new_shape():
    actual = pt.A.resize_shortest._compute_new_shape(100, 200, 400)
    assert actual == (100, 200)

    actual = pt.A.resize_shortest._compute_new_shape(100, 201, 400)
    assert actual == (100, 199)

    actual = pt.A.resize_shortest._compute_new_shape(100, 201, 400, trunc=False)
    assert np.isclose(actual, np.array((100, 199.004975))).all()


def test_resize_shortest_image(chelsea):
    assert chelsea.shape == (300, 451, 3)

    transform = pt.A.ResizeShortest(100)
    result = transform(image=chelsea)

    assert result["image"].shape == (100, 150, 3)


def test_resize_shortest_image_keypoint(chelsea):
    keypoints = [
        (264, 203),
        (86, 88),
        (254, 160),
        (193, 103),
    ]

    transform = A.Compose(
        [
            pt.A.ResizeShortest(100),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    result = transform(image=chelsea, keypoints=keypoints)
    expected_keypoints = np.array(
        [
            (87.80487804878048, 67.66666666666666),
            (28.60310421286031, 29.333333333333332),
            (84.47893569844788, 53.33333333333333),
            (64.19068736141907, 34.33333333333333),
        ]
    )
    actual_keypoints = np.array(result["keypoints"])
    assert np.isclose(expected_keypoints, actual_keypoints).all()
