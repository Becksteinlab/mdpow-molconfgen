"""Tests for the chem module."""

import pytest
import MDAnalysis as mda
import rdkit.Chem as rdkit
import numpy as np

from .. import chem
from ..data.files import V46_PDB, V46_ITP, V46_MOL2


@pytest.fixture
def universe():
    """Create a test universe with V46 molecule."""
    return mda.Universe(V46_ITP, V46_PDB)


@pytest.fixture
def mol(universe):
    """Create a test molecule from the universe."""
    return chem.load_mol(universe)


def test_load_mol(universe):
    """Test loading a molecule from a universe."""
    mol = chem.load_mol(universe)
    assert isinstance(mol, rdkit.Mol)
    assert mol.GetNumAtoms() == universe.atoms.n_atoms


def test_find_dihedral_indices(mol, universe):
    """Test finding dihedral indices in V46.

    The rotatable bond of interest in V46 (2-methyl-1-nitrobenzene) is O-N-C-C,
    where the dihedral is defined by the heavy atoms only.
    """
    dihedral_indices = chem.find_dihedral_indices(mol)

    # V46 has one rotatable bond (O-N-C-C)
    assert len(dihedral_indices) == 1

    # Check that the indices form a proper dihedral
    indices = dihedral_indices[0]
    assert len(indices) == 4

    # The dihedral should be O-N-C-C
    assert indices[0] == 2  # O9
    assert indices[1] == 1  # N8
    assert indices[2] == 0  # C1
    assert indices[3] == 4  # C2

    # Verify atom names
    atoms = [universe.atoms[i].name for i in indices]
    assert atoms == ["O9", "N8", "C1", "C2"]


def test_find_dihedral_indices_empty():
    """Test finding dihedral indices in a molecule with no rotatable bonds."""
    # Create a simple molecule with no rotatable bonds (e.g., methane)
    mol = rdkit.MolFromSmiles("C")
    dihedral_indices = chem.find_dihedral_indices(mol)
    assert len(dihedral_indices) == 0
