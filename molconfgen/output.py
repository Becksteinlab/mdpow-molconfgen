# Richard Richardson wrote this ;)
# Function to add a box to the output of
# sampler.generate_conformers

import MDAnalysis as mda
import MDAnalysis.transformations
import numpy as np
from tqdm import tqdm


def largest_r(ag):
    """Find the largest radius to enclose 'ag'.

    The function iterates over the whole trajectory associated with 'ag'
    and returns the maximum radius.

    Arguments
    ---------
    ag : AtomGroup
        'ag' contains the molecule of interest

    Returns
    -------
    r : float
    """
    u = ag.universe
    r = np.max([ag.bsphere()[0] for ts in u.trajectory])
    return r


def write_pbc_trajectory(u, filename, l=10, box=None):
    """Define the box for a trajectory and write to a file.

    The function defines a box for the trajectory associated with 'u' and
    writes it to a file. The default option is to write the trajectory to
    a file without a box.

    This is intended to be used with the output of molconfgen's
    sampler.generate_conformers, but it is general enough to use with any
    universe that contains a molecule and trajectory

    Arguments
    ---------
    u : MDAnalysis universe
        contains molecule of interest and a trajectory
    l : int or float
        Default is 10. This is the number that multplies the largest_r in
        the box = 'auto' option.
    filename : str
        name of the trajectory file to be written
    box : int or float, array_like, 'auto',  None
        There are four different options here to allow for customization
        of the box.
        None is the default and leaves the trajectory unmodified.
        'auto' will call the largest_r function and multiply the result
        by l. The box is then a cube with side lengths equal to l*r.
        int or float will assume the box is a cube with side lengths equal
        to the input argument.
        array_like must be a 1x6 array with the first three entries
        representing the sides of the box and the last three entries
        representing the angles between them.

    Returns
    -------
    filename : str
        a file containing the transformed (or not transformed) trajectory

    Notes
    -----
    Hypothetically the smallest box that would completely enclose the
    molecule should be 2*r where r is the largest radius to enclose the
    molecule; this is why I check if box <= 2*r for box = int or float.
    In my experience I have found that Gromacs prefers the box to be
    quite large or else it will complain about box dimensions when
    using mdrun -rerun. For this reason, I made it so that the 'auto'
    argument draws a very large box."""

    if box == None:
        u.atoms.write(filename, frames="all")
        return filename

    if box == "auto":
        # call largest_r to find the largest r
        r = largest_r(u)
        dim = np.array([l * r, l * r, l * r, 90, 90, 90])
        transform = mda.transformations.boxdimensions.set_dimensions(dim)
        u.trajectory.add_transformations(transform)
        u.atoms.write(filename, frames="all")
        return filename

    if isinstance(box, (float, int)):
        r = largest_r(u)
        if box <= 2 * r:
            raise ValueError(
                "Sides of box must be greater than 2*r where r is the largest radius enclosing the molecule"
            )
        dim = np.array([box, box, box, 90, 90, 90])
        transform = mda.transformations.boxdimensions.set_dimensions(dim)
        u.trajectory.add_transformations(transform)
        u.atoms.write(filename, frames="all")
        return filename

    if len(box) == 6:
        dim = np.asarry(box, dtype=np.float32)
        r = largest_r(u)
        for x in dim[0:2]:
            if x <= 2 * r:
                raise ValueError(
                    "Sides of box must be greater than 2*r where r is the largest radius enclosing the molecule"
                )
        transform = mda.transformations.boxdimensions.set_dimensions(dim)
        u.trajectory.add_transformations(transform)
        u.atoms.write(filename, frames="all")
        return filename
