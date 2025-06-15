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


