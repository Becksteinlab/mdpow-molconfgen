## Finding Local Minima on a Discrete Energy Landscape

Assume we have a scalar energy function
\(E(\boldsymbol{\phi})\) defined over an M-dimensional space of dihedral angles
\(\boldsymbol{\phi} = (\phi_1,\, \phi_2,\, \dots,\, \phi_M)\).
Our task is to locate every **local minimum**—a point whose energy is lower
than all of its immediate neighbours on the (periodic) torus.

### 0. Periodicity Assumption  
All dihedral dimensions are **periodic** with period 360° (or 2π rad).  The
energy surface therefore lives on an *M-torus*: **the angles −180° and +180°
represent the same physical orientation** (they are identical, not merely
adjacent).  In practice you should either:
* store grid angles in the half-open range (−180°, 180°] or [−180°, 180°), so
  that each orientation appears exactly once, **or**
* if the grid contains both endpoints, treat them as duplicates and ensure they
  are not double-counted when analysing minima, neighbours, or clustering.

Key practical consequences:
* Neighbourhood queries must use *wrapped indexing* (e.g. `np.roll`) with care
  to avoid comparing a point to its duplicate copy at the opposite boundary.
* Morphological filters run with `mode='wrap'` automatically respect the torus
  but will still operate correctly if the endpoint is omitted.
* Distance metrics used for clustering minima must be **angular** (modulo 360°)
  and take the identity of −180° ≡ +180° into account.
* When reporting minima, map every angle to a chosen canonical interval such as
  (−180°, 180°].

---

### 1. Collect & Remap Scattered Samples  
Input: an arbitrary set of samples \(\{(\boldsymbol{\phi}^{(i)}, E_i)\}_{i=1}^{N}\).

1. **Remap each dihedral** to the canonical half-open interval
   \(-180^\circ \le \phi_k < 180^\circ\) using
   `phi = ((phi + 180) % 360) - 180`.
2. **Optionally filter** obvious outliers / duplicates (keep lowest energy at
   identical \(\boldsymbol{\phi}\)).
3. Pass the cleaned, wrapped dataset to the interpolation stage.

---

### 2. Periodic Interpolation / Resampling
*Interpolation is **always** performed before minima detection.*

1. **Choose Target Grid Resolution**  
   Decide bins \(n_k\) for each \(\phi_k\).


2. **Augment Periodic Images**  
   Duplicate samples near the \(\pm180^\circ\) boundaries (±360° shifts) to
   avoid edge artefacts.

3. **Interpolation Options (torus-aware)**
   | Method | Library | Notes |
   |--------|---------|-------|
   | *Periodic RBF* | `scipy.interpolate.RBFInterpolator` (with custom
   distance \(d=\min(|\Delta\theta|,360-|\Delta\theta|)\)) | Handles scattered → scattered; easy to extend to M-D |
   | *RegularGrid + bin-average* | build empty grid & accumulate samples with
   periodic bin indices, then divide by counts | Fast, simple smoothing |
   | *N-dim B-spline* | `rectBivariateSpline` in 2-D; generalise with
   `ndimage.map_coordinates` & duplicate endpoints | Good smoothness |
   | *Delaunay / barycentric* | `scipy.spatial.Delaunay` in the unwrapped
   angular space with duplicated boundary points | Exact within simplices |

4. **Fill Gaps & NaNs**  
   After interpolation some voxels may still be empty.  Fill using nearest
   neighbour or Gaussian smoothing (with `mode='wrap'`).

5. **Result**: periodic regular grid `E` on shape `(n_1,…,n_M)` covering
   \(-180^\circ,180^\circ)^{\otimes M}`.

---

### 3. Identify Candidate Minima
(Formerly Step 4)

Approach A — morphological filtering (fast, vectorised):
1. Compute `E_min = ndimage.minimum_filter(E, size=3, mode='wrap')` (3^M neighbourhood).
2. A grid point `i` is a local minimum iff `E[i] <= E_min[i]` *and* it is finite.
3. Collect the indices where this condition holds.

Approach B — explicit neighbour comparison (memory-lean):
Use `np.roll` to compare each voxel to its ±1 wrapped neighbours along every axis.

---

### 4. Merge Near-Degenerate Minima
Clustering must respect periodicity.  Compute the wrapped distance between two
index vectors *i* and *j* as:
\[d_k = \min(|i_k-j_k|,\; n_k-|i_k-j_k|)\]
where \(n_k\) is the grid size along axis *k*.  Two minima are neighbours if
`max(d_k) ≤ 1`.

---

### 5. Rank & Analyse Minima
1. Sort minima by ascending energy.
2. For each minimum record:
   • grid indices & dihedral angles
   • energy value
   • basin size (see §6)

---

### 6. Basin Assignment / Gradient Descent (updated)
When following the steepest-descent path use wrapped indexing to move to
neighbours.  Alternatively apply **periodic padding**:
```python
E_padded = np.pad(E, 1, mode='wrap')
```
simplifying neighbour lookup.

---

### 7. Algorithms & Libraries
Below are practical algorithms well-suited for this task:

1. **SciPy ndimage minimum_filter** (Approach A) – fast, pure-Python wrapper to
   C loops; scales roughly linearly with number of grid points.
2. **Flood-fill / Steepest Descent Mapping** – deterministic, finds basins and
   saddles; parallelisable; used in protein-folding energy landscape studies.
3. **Watershed Segmentation** – treat \(-E\) as a height map; the watershed
   lines correspond to ridge lines; minima are catchment basins.
4. **Topological Data Analysis (Morse–Smale complex)** – more elaborate,
   captures connectivity of minima & saddles; libraries: `gudhi`, `giotto-tda`.
5. **Basin-hopping / Monte-Carlo minimisation** – use the grid energies to
   accelerate off-grid optimisation for higher resolution, but still requires
   initial discrete minima.

---

### 8. Complexity Considerations
• Let \(N=\prod_i n_i\) be total grid points.
• Minimum-filter and neighbour comparisons are \(\mathcal{O}(N)\).
• Memory scales with \(N\); for large M, store energies in chunks or use
  Dask/Numba to process blocks.

---

### 9. Pseudocode snippet (periodic explicit method)
```python
# Explicit neighbour comparison with periodic boundaries
is_min = np.ones_like(E, bool)
for axis in range(E.ndim):
    is_min &= E <= np.roll(E,  1, axis)
    is_min &= E <  np.roll(E, -1, axis)
minima_idx = np.argwhere(is_min)
```

All subsequent analysis (clustering, basin identification) should use
wrapped/torus-aware distance and connectivity rules.

---

### 10. Next Steps
1. Implement Approach A; benchmark vs Approach B on V39 data.
2. Validate minima by running short MM optimisations starting from each
   identified grid minimum.
3. Visualise minima on 2-D and 1-D projections of the energy landscape.


