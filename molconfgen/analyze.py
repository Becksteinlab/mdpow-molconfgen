"""Analysis module for molecular conformers.

This module provides functions to analyze dihedral angles and energies from
molecular dynamics trajectories.
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from typing import List, Tuple, Optional

from . import chem


def get_energies(edr_file: str, energy_term: str = "Potential") -> np.ndarray:
    """Extract potential energies from a GROMACS energy file.

    Parameters
    ----------
    edr_file : str
        Path to the GROMACS energy file (.edr)
    energy_term : str, optional
        The energy term to extract from the energy file.
        Default is "Potential".
        Other options include "Total", "Kinetic", "Temperature", "Pressure", etc.

    Returns
    -------
    numpy.ndarray
        Array of energies in kJ/mol (or other units, depending on `energy_term`)
    """
    aux = mda.auxiliary.EDR.EDRReader(edr_file, convert_units=False)
    energies = aux.get_data(energy_term)
    return energies[energy_term]


def get_dihedral_angles(
    universe: mda.Universe, **kwargs
) -> Tuple[np.ndarray, List[mda.AtomGroup]]:
    """Get dihedral angles for all frames in a trajectory.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe containing the trajectory
    **kwargs : dict
        Additional keyword arguments to pass to the Dihedral.run() method

    Returns
    -------
    tuple
        - angles : np.ndarray
            Array of dihedral angles for each frame
        - dihedral_groups : List[mda.AtomGroup]
            List of atom groups defining each dihedral

    See Also
    --------
    :class:`MDAnalysis.analysis.dihedrals.Dihedral`
        The Dihedral analysis class that is used to calculate the dihedral angles.
    """

    # Get molecule and find dihedral indices
    mol = chem.load_mol(universe)
    dihedral_indices = chem.find_dihedral_indices(mol)

    # Create atom groups for each dihedral
    dihedral_groups = []
    for indices in dihedral_indices:
        dihedral_groups.append(universe.atoms[indices])

    # Calculate dihedral angles
    dihedral_analysis = Dihedral(dihedral_groups).run(**kwargs)

    return dihedral_analysis.results.angles, dihedral_groups


def analyze_conformers(
    topology_file: str,
    trajectory_file: str,
    energy_file: str,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Analyze conformers from a trajectory file.

    Note that the system should only contain the molecule of interest.

    Parameters
    ----------
    topology_file : str
        Path to the topology file (.pdb or .itp or .tpr)
    trajectory_file : str
        Path to the trajectory file (.trr)
    energy_file : str
        Path to the energy file (.edr)
    **kwargs : dict
        Additional keyword arguments to pass to the Dihedral.run() method

    Returns
    -------
    tuple
        - angles : np.ndarray
            Array of dihedral angles for each frame
        - energies : np.ndarray
            Array of energies for each frame
    """
    u = mda.Universe(topology_file, trajectory_file)
    angles, _ = get_dihedral_angles(u, **kwargs)
    energies = get_energies(energy_file)

    return angles, energies
