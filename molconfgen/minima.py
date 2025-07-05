from typing import List, Tuple, Sequence, Union

import numpy as np
import scipy.interpolate
import scipy.ndimage

__all__ = [
    "wrap_angles",
    "wrap_tensor",    
    "regular_grid",
    "nearest_neighbor_interpolate",
    "find_local_minima",
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

def wrap_tensor(E: np.ndarray, grid_vectors: List[np.ndarray], lower: float = -180.0, upper: float = 180.0) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Wrap tensor E and grid vectors from [0, 360) to [lower, upper).
    
    This function transforms a tensor E with periodic data originally defined
    on grid vectors spanning [0, 360) to the equivalent representation on 
    [lower, upper), reordering the data to maintain correspondence between
    tensor elements and angle values.
    
    Parameters
    ----------
    E : np.ndarray
        M-dimensional tensor with periodic data, shape (N, N, ..., N)
    grid_vectors : List[np.ndarray]
        List of M arrays, each of length N, containing angle values in degrees
        for each axis, typically spanning [0, 360)
    lower : float, optional
        Lower bound for wrapped angles
    upper : float, optional
        Upper bound for wrapped angles
        
    Returns
    -------
    E_wrapped : np.ndarray
        Reordered tensor with same shape as E
    grid_vectors_wrapped : List[np.ndarray]
        List of M wrapped and sorted grid vectors
    """
    # Wrap each grid vector and find the reordering
    new_grid_vectors = []
    reorder_indices = []
    
    for gv in grid_vectors:
        wrapped = wrap_angles(gv, lower=lower, upper=upper)
        # Find the indices that would sort the wrapped angles
        sort_idx = np.argsort(wrapped)
        new_grid_vectors.append(wrapped[sort_idx])
        reorder_indices.append(sort_idx)
    
    # Reorder the tensor along each axis
    E_new = E.copy()
    for axis, idx in enumerate(reorder_indices):
        E_new = np.take(E_new, idx, axis=axis)
    
    return E_new, new_grid_vectors


# -----------------------------------------------------------------------------
#  Interpolation using periodic nearest neighbor interpolation
# -----------------------------------------------------------------------------


def nearest_neighbor_interpolate(
    points: np.ndarray,
    energies: np.ndarray,
    num_conformers: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Interpolate scattered periodic dihedral samples onto a regular grid.

    The interpolation is done using a fast periodic nearest neighbor interpolator
    to take periodicity of dihedral angles over 360º into account.

    The energies are shifted so that the minimum energy is 0.

    The purpose of this function is to regularize the dihedral angles and energies.
    In principle, molconfgen workflow already samples on a regular grid, but there is
    the possibility of small numerical differences and this procedure ensure that we
    can work with data on guaranteed regular grid.

    Parameters
    ----------
    points
        (N, M) dihedral samples in **degrees**. Will be wrapped.
    energies
        Energies E_i, shape (N,).
    num_conformers
        Number of conformers that were used in the sampling. The M-dimensional 
        resulting grid will have shape `(num_conformers, num_conformers, ...)`.
        The nearest neighbor interpolation will simply associate the calculated 
        energy with the nearest dihedral angle. Therefore, it makes no sense to
        choose a larger number of conformers here than the original one.

    Returns
    -------
    grid_vectors
        List of M arrays (one for each dihedral), each of length `num_conformers`, indicating
        the grid points for each dihedral.
    E
        Interpolated (and shifted)energies, shape `(num_conformers, num_conformers, ...)`
    """
    # We need to rewrap from [-180, 180) to [0, 360) because the PBC-aware KDTree only works with positive values
    points = wrap_angles(points, lower=0, upper=360)    
    energies = np.asarray(energies, float)
    energies = energies - energies.min()

    # Build target regular grid and all points to interpolate on (xi)
    grid_shape = points.shape[1] * [num_conformers]
    grid_vectors, E = regular_grid(grid_shape, dtype=float, lower=0, upper=360)
    mesh = np.meshgrid(*grid_vectors, indexing="ij")
    xi = np.stack([m.ravel() for m in mesh], axis=-1)

    # Periodic NN interpolation
    ndi = scipy.interpolate.NearestNDInterpolator(points, energies, tree_options={"boxsize": 360.})
    interp_vals = ndi(xi, workers=-1)

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

