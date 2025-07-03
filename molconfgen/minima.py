import numpy as np
from scipy import ndimage
from typing import List, Tuple, Sequence, Union
from itertools import product
from scipy.interpolate import griddata

__all__ = [
    "wrap_angles",
    "regular_grid",
    "griddata_interpolate",
    "find_local_minima",
    "merge_near_degenerate",
]

AngleArray = Union[np.ndarray, Sequence[float]]


# -----------------------------------------------------------------------------
#  General utilities
# -----------------------------------------------------------------------------

def wrap_angles(phi: AngleArray, lower: float = -180.0, upper: float = 180.0) -> np.ndarray:
    """Wrap dihedral angles to the canonical interval [lower, upper).

    Parameters
    ----------
    phi
        Array-like of angles in degrees.
    lower
        Lower bound of the interval.
    upper
        Upper bound of the interval.

    Returns
    -------
    np.ndarray
        Wrapped angles of same shape as *phi*.
    """
    phi = np.asarray(phi, dtype=float)
    phi_range = upper - lower
    return (phi - lower) % phi_range + lower


# -----------------------------------------------------------------------------
#  Interpolation using scipy.interpolate.griddata (preferred)
# -----------------------------------------------------------------------------

import scipy.spatial

def augment_periodic_samples(points: np.ndarray, values: np.ndarray, 
                             cutoff : float = 30.0,
                             lower : float = -180.0,
                             upper : float = 180.0,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment periodic samples with images within `cutoff` to handle periodicity.

    Points are assumed to be periodic in the interval [lower, upper) in all dimensions. 
    We find the nearest neighbors within `cutoff` of the convex hull of the points. 
    We then create new points by shifting the surface points by ±360° in the direction of the shift vector. 
    We then return the original points concatenated with the new points and the corresponding values.

    Parameters
    ----------
    points : array-like
        (N, M) samples in *degrees*. Will be wrapped.
    values : array-like
        Values V_i, shape (N,) (e.g., energies)
    cutoff : float
        Threshold for determining significant shifts (should be the same as the cutoff used to find the neighbors)
    lower : float
        Lower bound of the interval (e.g., -180°)
    upper : float
        Upper bound of the interval (e.g., 180°)

    Returns
    -------
    tuple
        (all_points, all_values) where:
        - all_points: numpy.ndarray of all points
        - all_values: numpy.ndarray of all values
    """
    if not np.allclose(upper - lower, 360.0):
        raise ValueError(f"upper - lower must be 360, but is {upper - lower}")
    
    # We need to rewrap from [-180, 180) to [0, 360) because the PBC-aware KDTree only works with positive values
    points360 = wrap_angles(points, lower=0, upper=360)

    ptree = scipy.spatial.KDTree(points360, boxsize=points.shape[1] * [360])

    hull = scipy.spatial.ConvexHull(points360)
    surface = points360[hull.vertices]

    # neighbors WITH PBC taken into account
    neighbors = ptree.query_ball_point(surface, r=cutoff)

    # distances WITHOUT PBC
    shifts = [points360[neighbors[i]] - surface[i] for i in range(len(neighbors))]
    d = [np.linalg.norm(group, axis=1) for group in shifts]

    points_outside = [group_distances > cutoff for group_distances in d]
    vertices_outside = [np.asarray(group)[outside] for group, outside in zip(neighbors, points_outside)]
    shifts_outside = [shift_vectors[outside] for shift_vectors, outside in zip(shifts, points_outside)]

    # go from ragged *_outside lists of arrays to flat arrays of all points outside the convex hull
    points_outside_flat = np.concatenate([v for v in vertices_outside if len(v) > 0])
    shifts_outside_flat = np.concatenate([s for s in shifts_outside if len(s) > 0])

    augmented_points, indices = _create_augmented_points(points360, points_outside_flat, shifts_outside_flat, cutoff)

    # rewrap from [0, 360) to [lower, upper)
    augmented_points = wrap_angles(augmented_points, lower=lower, upper=upper)

    all_points = np.concatenate([points, augmented_points])
    all_values = np.concatenate([values, values[indices]])

    return all_points, all_values

def _create_augmented_points(points, points_outside, shifts_outside, cutoff):
    """
    Fully vectorized approach to create augmented points.
    
    Parameters
    ----------
    points : array-like
        Array of points
    points_outside : array-like
        Indices of points that are outside the boundary.
        These are the indices into points.
    shifts_outside : array-like
        Shift vectors for points outside the boundary
        These are the shift vectors for the points outside the boundary.
    cutoff : float
        Threshold for determining significant shifts (should be the same as the cutoff used to find the neighbors)
        
    Returns
    -------
   tuple
        (augmented_points, vertex_indices) where:
        - augmented_points: numpy.ndarray of augmented points
        - vertex_indices: numpy.ndarray of actual vertex indices from the original data (points)
    """ 
    surface_points = points[points_outside]
    
    # shifts >= cutoff indicate that we need to make one or more periodic images
    # and generate augmentation points
    shift_mask = np.abs(shifts_outside) > cutoff
    
    # Get indices of points and dimensions that need shifting
    point_indices, dim_indices = np.where(shift_mask)
    
    # Create a new point for each (point_idx, dim_idx) pair
    base_points = surface_points[point_indices]
    
    # Get the shift directions
    shift_directions = np.sign(shifts_outside[point_indices, dim_indices])
    
    # Create output array and apply shifts
    augmented_points = base_points.copy()
    augmented_points[np.arange(len(dim_indices)), dim_indices] += 360.0 * shift_directions
    
    # Create array of original point indices with duplicates
    # These are the indices into points_outside, not the original vertices
    augmented_indices = point_indices
    
    # Get the actual vertex indices by mapping through points_outside_flat
    vertex_indices = points_outside[point_indices]
    
    # Assert that the length of augmented_indices equals the number of augmented points
    assert len(augmented_indices) == len(augmented_points), \
        f"Length mismatch: {len(augmented_indices)} indices vs {len(augmented_points)} points"
    
    return augmented_points, vertex_indices






def griddata_interpolate(
    points: np.ndarray,
    energies: np.ndarray,
    grid_shape: Sequence[int],
    method: str = "linear",
    fill_value: float = np.nan,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Interpolate scattered periodic dihedral samples onto a regular grid.

    Parameters
    ----------
    points
        (N, M) dihedral samples in **degrees**. Will be wrapped.
    energies
        Energies E_i, shape (N,).
    grid_shape
        Desired grid resolution as tuple/list of ints.
    method
        Interpolation method passed to :pyfunc:`scipy.interpolate.griddata`.
        One of "linear", "nearest" or "cubic".
    fill_value
        Value assigned outside the convex hull (should not occur with adequate
        periodic augmentation).
    """
    points = wrap_angles(points)
    energies = np.asarray(energies, float)

    # Augment points with ±360° images to handle periodicity
    aug_points, aug_energy = _augment_periodic_samples(points, energies)

    # Build target regular grid (same as regular_grid)
    grid_vectors, E = regular_grid(grid_shape, dtype=float)
    mesh = np.meshgrid(*grid_vectors, indexing="ij")
    xi = np.stack([m.ravel() for m in mesh], axis=-1)

    # Interpolate
    interp_vals = griddata(
        aug_points,
        aug_energy,
        xi,
        method=method,
        fill_value=fill_value,
    )

    E[:, :] = interp_vals.reshape(grid_shape)

    return grid_vectors, E


# -----------------------------------------------------------------------------
#  Minima detection
# -----------------------------------------------------------------------------

def find_local_minima(E: np.ndarray) -> np.ndarray:
    """Return indices of local minima in periodic array *E*.

    Uses a 3^M neighbourhood and `mode='wrap'` to respect periodicity.

    Returns
    -------
    np.ndarray
        Array of shape (K, M) with integer grid indices of minima.
    """
    if E.ndim < 1:
        raise ValueError("E must be at least 1-D")

    mask_finite = np.isfinite(E)
    E_work = E.copy()
    E_work[~mask_finite] = np.inf  # ignore NaNs

    local_min = E_work <= ndimage.minimum_filter(E_work, size=3, mode="wrap")
    local_min &= mask_finite

    return np.argwhere(local_min)


def _wrapped_distance(idx_a: np.ndarray, idx_b: np.ndarray, shape: Sequence[int]):
    diff = np.abs(idx_a - idx_b)
    wrapped = np.minimum(diff, np.array(shape) - diff)
    return np.max(wrapped)


def merge_near_degenerate(
    minima_idx: np.ndarray,
    E: np.ndarray,
    energy_tol: float = 0.1,
) -> List[Tuple[np.ndarray, float]]:
    """Cluster minima that are equivalent up to index distance ≤1 and ΔE < tol."""
    shape = np.array(E.shape)
    minima_idx = [idx for idx in minima_idx]  # list of ndarray
    clusters: List[List[np.ndarray]] = []

    for idx in minima_idx:
        placed = False
        for cl in clusters:
            ref = cl[0]
            if _wrapped_distance(idx, ref, shape) <= 1 and np.abs(E[tuple(idx)] - E[tuple(ref)]) <= energy_tol:
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    result: List[Tuple[np.ndarray, float]] = []
    for cl in clusters:
        cl = sorted(cl, key=lambda i: E[tuple(i)])
        best = cl[0]
        result.append((best, E[tuple(best)]))
    return result


# -----------------------------------------------------------------------------
#  Regular grid helper 
# -----------------------------------------------------------------------------

def regular_grid(shape: Sequence[int], lower: float = -180.0, upper: float = 180.0, dtype=float) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return (grid_vectors, empty_nan_array) for a periodic regular grid.

    Each axis covers the half-open interval (lower, upper] using *n_k* equally
    spaced points (endpoint excluded) so that spacing = (upper - lower) / n_k.
    """
    shape = tuple(int(n) for n in shape)
    grid_vectors = [np.linspace(lower, upper, n, endpoint=False) for n in shape]
    E = np.full(shape, np.nan, dtype=dtype)
    return grid_vectors, E

