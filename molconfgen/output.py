# Richard Richardson wrote this ;)
# Function to add a box to the output of
# sampler.generate_conformers

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm


def largest_r(ag):
    """Find the largest radius to enclose 'ag'.

    The function iterates over the whole trajectory associated with 'ag'
    and returns the maximum radius.

    Arguments
    ---------
    ag : MDAnalysis.core.groups.AtomGroup or MDAnalysis.Universe
        Contains the molecule of interest

    Returns
    -------
    float
        The maximum radius needed to enclose the molecule across all frames
    """
    u = ag.universe
    r = np.max([ag.bsphere()[0] for ts in u.trajectory])
    return r


def write_pbc_trajectory(ag, filename, scale=10.0, box=None):
    """Define the box for a trajectory and write to a file.

    The function defines a box for the trajectory associated with 'ag' and
    writes it to a file. The default option is to write the trajectory to
    a file without a box.

    This is intended to be used with the output of molconfgen's
    sampler.generate_conformers, but it is general enough to use with any
    universe that contains a molecule and trajectory.

    Arguments
    ---------
    ag : MDAnalysis.core.groups.AtomGroup or MDAnalysis.Universe
        Contains the molecule of interest and a trajectory
    filename : str
        Name of the trajectory file to be written
    scale : float, optional
        Default is 10.0. This is the number that multiplies the largest_r in
        the box = 'auto' option.
    box : float, array_like, 'auto', or None, optional
        There are four different options here to allow for customization
        of the box:
        - None: leaves the trajectory unmodified
        - 'auto': calls largest_r and multiplies the result by scale
        - float: assumes the box is a cube with side lengths equal to the input
        - array_like: must be a 1x6 array with the first three entries
          representing the sides of the box and the last three entries
          representing the angles between them

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
    ag_new = u.atoms[ag.ix]

    if box is None:
        ag_new.atoms.write(filename, frames="all")
        return ag_new

    if box == "auto":
        dim = np.array([scale * r, scale * r, scale * r, 90, 90, 90])
    elif isinstance(box, (float, int)):
        if box <= 2 * r:
            raise ValueError(
                "Sides of box must be greater than 2*r where r is the largest radius enclosing the molecule"
            )
        dim = np.array([box, box, box, 90, 90, 90])
    elif len(box) == 6:
        dim = np.array(box, dtype=np.float32)
        if np.any(dim[:2] <= 2 * r):
            raise ValueError(
                "Sides of box must be greater than 2*r where r is the largest radius enclosing the molecule"
            )
    else:
        raise ValueError(
            "box must be None, 'auto', a float, or a 6-element array"
        )

    transform = mda.transformations.boxdimensions.set_dimensions(dim)
    u.trajectory.add_transformations(transform)
    ag_new.atoms.write(filename, frames="all")
    return ag_new
