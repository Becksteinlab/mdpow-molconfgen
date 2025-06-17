# Richard Richardson wrote this ;)
# Function to add a box to the output of
# sampler.generate_conformers

import MDAnalysis as mda
import MDAnalysis.transformations
import numpy as np
from typing import List, Optional, Union, Any


def largest_r(
    ag: Union[MDAnalysis.core.groups.AtomGroup, MDAnalysis.Universe],
) -> float:
    """Calculate the largest radius that encloses the molecule.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup or MDAnalysis.Universe
        The AtomGroup or Universe containing the molecule.

    Returns
    -------
    float
        The largest radius that encloses the molecule.
    """
    u = ag.universe
    r = np.max([ag.atoms.bsphere()[0] for ts in u.trajectory])
    return r


def write_pbc_trajectory(
    ag: Union[MDAnalysis.core.groups.AtomGroup, MDAnalysis.Universe],
    filename: str,
    box: Optional[Union[float, List[float], str]] = None,
    rcoulomb: Optional[float] = None,
    buffer: float = 10.0,
    frames: Optional[Union[int, List[int], str]] = None,
) -> MDAnalysis.core.groups.AtomGroup:
    """Define the box for a trajectory and write to a file.

    The function defines a box for the trajectory associated with 'ag' and
    writes it to a file. The default option is to write the trajectory to
    a file without a box.

    This is intended to be used with the output of molconfgen's
    :func:`sampler.generate_conformers`, but it is general enough to use with any
    universe that contains a molecule and trajectory.

    Arguments
    ---------
    ag : MDAnalysis.core.groups.AtomGroup or MDAnalysis.Universe
        Contains the molecule of interest and a trajectory
    filename : str
        Name of the trajectory file to be written
    box : float, array_like, 'auto', or ``None``, optional
        There are four different options here to allow for customization
        of the box:
        - ``None: leaves the trajectory unmodified
        - 'auto': calls :func:`largest_r and adds the value of `rcoulomb` and `buffer`
        - float: assumes the box is a cube with side lengths equal to the input
        - array_like: must be a 1x6 array with the first three entries
          representing the sides of the box and the last three entries
          representing the angles between them
    rcoulomb : float, optional
        The cutoff radius for the Coulomb interaction to be used with `box="auto"`.
        If None, the Coulomb radius will not be taken into account (and set to 0).
        (in Angstrom)
    buffer : float, optional
        The buffer to be added to the box size with `box="auto"` (in Angstrom).
    frames : int or list of ints or str, optional
        The frames to write to the output file. If None, all frames are written
        (same as `frames="all"`).
        If an integer, this frame is written, with the first frame being 0 and the
        last frame -1. If a list of integers, only the specified frames are written.


    Returns
    -------
    MDAnalysis.core.groups.AtomGroup
        The AtomGroup with the transformed (or not transformed) trajectory

    Notes
    -----
    For orthorhombic boxes, the smallest box that would completely enclose the
    molecule should be 2*r where r is the largest radius to enclose the
    molecule. For triclinic boxes this is just a rule of thumb.
    """
    u = ag.universe.copy()
    r = largest_r(ag)
    ag_new = u.atoms[ag.atoms.ix]

    frames = "all" if frames is None else frames
    if isinstance(frames, int):
        frames = [frames]

    if box is None:
        ag_new.atoms.write(filename, frames="all")
        return ag_new

    rcoulomb = rcoulomb or 0.0
    if box == "auto":
        L = 2 * (r + rcoulomb + buffer)
        dim = np.array([L, L, L, 90, 90, 90])
    elif isinstance(box, (float, int)):
        if box <= 2 * (r + rcoulomb):
            raise ValueError(
                "Sides of box must be greater than 2*(r + rcoulomb) where r is the largest radius enclosing the molecule"
            )
        dim = np.array([box, box, box, 90, 90, 90])
    elif len(box) == 6:
        dim = np.array(box, dtype=np.float32)
        if np.any(dim[:2] <= 2 * (r + rcoulomb)):
            raise ValueError(
                "Sides of box must be greater than 2*(r + rcoulomb) where r is the largest radius enclosing the molecule"
            )
    else:
        raise ValueError(
            "box must be None, 'auto', a float, or a 6-element array"
        )

    transform = MDAnalysis.transformations.boxdimensions.set_dimensions(dim)
    u.trajectory.add_transformations(transform)
    ag_new.atoms.write(filename, frames=frames)
    return ag_new
