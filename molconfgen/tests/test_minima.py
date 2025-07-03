import numpy as np
import pytest
from molconfgen.minima import wrap_angles

@pytest.mark.parametrize(
    "phi, lower, upper, expected",
    [
        # Standard interval [-180, 180)
        (90, -180, 180, 90),
        (190, -180, 180, -170),
        (-190, -180, 180, 170),
        (180, -180, 180, -180),
        (-180, -180, 180, -180),
        (0, -180, 180, 0),
        # Interval [0, 360)
        (370, 0, 360, 10),
        (-10, 0, 360, 350),
        (360, 0, 360, 0),
        (0, 0, 360, 0),
        # Interval [10, 20)
        (25, 10, 20, 15),
        (9, 10, 20, 19),
        (10, 10, 20, 10),
        (20, 10, 20, 10),
        # Float input
        (370.5, 0, 360, 10.5),
        (-10.5, 0, 360, 349.5),
        # Vectorized input
        (np.array([90, 190, -190, 180, -180, 0]), -180, 180, np.array([90, -170, 170, -180, -180, 0])),
    ]
)
def test_wrap_angles_parametrized(phi, lower, upper, expected):
    result = wrap_angles(phi, lower, upper)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert np.isclose(result, expected) 