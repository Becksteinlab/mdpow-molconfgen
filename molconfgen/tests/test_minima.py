import numpy as np
import pytest
from molconfgen.minima import wrap_angles, wrap_tensor, regular_grid

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


class TestWrapTensor:
    """Test suite for wrap_tensor function"""
    
    def test_wrap_tensor_1d(self):
        """Test wrap_tensor with 1D data (M=1, N=6)"""
        N = 6
        # Create original grid from 0 to 360 (exclusive)
        grid_vectors, E = regular_grid([N], lower=0, upper=360)
        
        # Fill with test data where E[i] = i for easy verification
        E = np.arange(N, dtype=float)
        
        # Original angles: [0, 60, 120, 180, 240, 300]
        # Wrapped angles: [0, 60, 120, -180, -120, -60]
        # Sorted wrapped: [-180, -120, -60, 0, 60, 120]
        # Expected indices: [3, 4, 5, 0, 1, 2]
        
        E_wrapped, grid_vectors_wrapped = wrap_tensor(E, grid_vectors)
        
        # Check grid vectors are correctly wrapped and sorted
        expected_angles = np.array([-180, -120, -60, 0, 60, 120])
        np.testing.assert_array_almost_equal(grid_vectors_wrapped[0], expected_angles)
        
        # Check tensor is correctly reordered
        expected_E = np.array([3, 4, 5, 0, 1, 2], dtype=float)
        np.testing.assert_array_equal(E_wrapped, expected_E)
    
    def test_wrap_tensor_2d(self):
        """Test wrap_tensor with 2D data (M=2, N=6)"""
        N = 6
        grid_vectors, E = regular_grid([N, N], lower=0, upper=360)
        
        # Fill with test data where E[i,j] = i*N + j for easy verification
        for i in range(N):
            for j in range(N):
                E[i, j] = i * N + j
        
        E_wrapped, grid_vectors_wrapped = wrap_tensor(E, grid_vectors)
        
        # Check both grid vectors are correctly wrapped
        expected_angles = np.array([-180, -120, -60, 0, 60, 120])
        np.testing.assert_array_almost_equal(grid_vectors_wrapped[0], expected_angles)
        np.testing.assert_array_almost_equal(grid_vectors_wrapped[1], expected_angles)
        
        # Check tensor shape is preserved
        assert E_wrapped.shape == E.shape
        
        # Check a few specific values
        # Original E[0,0] = 0 should map to E_wrapped[3,3] (indices [3,4,5,0,1,2])
        assert E_wrapped[3, 3] == 0
        # Original E[3,3] = 3*6 + 3 = 21 should map to E_wrapped[0,0]
        assert E_wrapped[0, 0] == 21

    def test_wrap_tensor_4d(self):
        """Test wrap_tensor with 4D data (M=4, N=6)"""
        N = 6
        grid_vectors, E = regular_grid([N, N, N, N], lower=0, upper=360)
        
        # Fill with test data using a unique pattern
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        E[i, j, k, l] = i * N**3 + j * N**2 + k * N + l
        
        E_wrapped, grid_vectors_wrapped = wrap_tensor(E, grid_vectors)
        
        # Check all grid vectors are correctly wrapped
        expected_angles = np.array([-180, -120, -60, 0, 60, 120])
        for gv in grid_vectors_wrapped:
            np.testing.assert_array_almost_equal(gv, expected_angles)
        
        # Check tensor shape is preserved
        assert E_wrapped.shape == E.shape
        
        # Check a specific value
        # Original E[0,0,0,0] = 0 should map to E_wrapped[3,3,3,3]
        assert E_wrapped[3, 3, 3, 3] == 0
        
        # Original E[3,3,3,3] should map to E_wrapped[0,0,0,0]
        expected_val = 3 * N**3 + 3 * N**2 + 3 * N + 3
        assert E_wrapped[0, 0, 0, 0] == expected_val

    def test_wrap_tensor_custom_bounds(self):
        """Test wrap_tensor with custom lower and upper bounds"""
        N = 6
        grid_vectors, E = regular_grid([N], lower=0, upper=360)
        E = np.arange(N, dtype=float)
        
        # Test with custom bounds [-90, 270)
        E_wrapped, grid_vectors_wrapped = wrap_tensor(E, grid_vectors, lower=-90, upper=270)
        
        # Original angles: [0, 60, 120, 180, 240, 300]
        # Wrapped to [-90, 270): [0, 60, 120, 180, 240, -60]
        # Sorted: [-60, 0, 60, 120, 180, 240]
        # Expected indices: [5, 0, 1, 2, 3, 4]
        
        expected_angles = np.array([-60, 0, 60, 120, 180, 240])
        np.testing.assert_array_almost_equal(grid_vectors_wrapped[0], expected_angles)
        
        expected_E = np.array([5, 0, 1, 2, 3, 4], dtype=float)
        np.testing.assert_array_equal(E_wrapped, expected_E)

    def test_wrap_tensor_preserves_periodicity(self):
        """Test that wrap_tensor preserves periodic relationships"""
        N = 6
        grid_vectors, E = regular_grid([N], lower=0, upper=360)
        
        # Create a simple periodic function: cos(angle)
        angles_rad = np.radians(grid_vectors[0])
        E = np.cos(angles_rad)
        
        E_wrapped, grid_vectors_wrapped = wrap_tensor(E, grid_vectors)
        
        # Check that the wrapped function is still periodic
        wrapped_angles_rad = np.radians(grid_vectors_wrapped[0])
        expected_wrapped = np.cos(wrapped_angles_rad)
        
        np.testing.assert_array_almost_equal(E_wrapped, expected_wrapped)

    def test_wrap_tensor_with_nan_values(self):
        """Test wrap_tensor handles NaN values correctly"""
        N = 6
        grid_vectors, E = regular_grid([N], lower=0, upper=360)
        
        # Fill with test data first
        E = np.arange(N, dtype=float)
        
        # Set some values to NaN
        E[1] = np.nan
        E[4] = np.nan
        
        E_wrapped, grid_vectors_wrapped = wrap_tensor(E, grid_vectors)
        
        # Check that NaN values are preserved in the correct positions
        # Original NaN at index 1 should move to index 4 (since reordering is [3,4,5,0,1,2])
        # Original NaN at index 4 should move to index 1
        assert np.isnan(E_wrapped[4])  # Original index 1
        assert np.isnan(E_wrapped[1])  # Original index 4
        
        # Non-NaN values should be preserved
        assert E_wrapped[0] == 3  # Original index 3
        assert E_wrapped[2] == 5  # Original index 5
        assert E_wrapped[3] == 0  # Original index 0
        assert E_wrapped[5] == 2  # Original index 2 