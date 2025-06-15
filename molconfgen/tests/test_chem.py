"""Tests for the chem module."""

import pytest
import MDAnalysis as mda
import rdkit.Chem as rdkit
import numpy as np

from .. import chem
from ..data.files import (
    V46_PDB,
    V46_ITP,
    V46_MOL2,
    V36_PDB,
    V36_ITP,
    V39_PDB,
    V39_ITP,
)


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


@pytest.fixture(
    params=[
        # V46: O-N-C-C (indices, atom names)
        (V46_ITP, V46_PDB, [[2, 1, 0, 4]], [["O9", "N8", "C1", "C2"]]),
        # V36: O-C-O-C (indices, atom names)
        (V36_ITP, V36_PDB, [[2, 0, 1, 3]], [["O4", "C1", "O2", "C3"]]),
        # V39: C-C-C-C (indices, atom names)
        (
            V39_ITP,
            V39_PDB,
            [[2, 0, 1, 3], [0, 1, 3, 14], [1, 3, 14, 17], [3, 14, 17, 9]],
            [
                ["O7", "C1", "O2", "C3"],
                ["C1", "O2", "C3", "C4"],
                ["O2", "C3", "C4", "C5"],
                ["C3", "C4", "C5", "C6"],
            ],
        ),
    ]
)
def mol_universe_and_expected(request):
    itp, pdb, expected_indices, expected_names = request.param
    universe = mda.Universe(itp, pdb)
    mol = chem.load_mol(universe)
    return mol, universe, expected_indices, expected_names


def test_find_dihedral_indices(mol_universe_and_expected):
    """Test finding dihedral indices for V46 and V36."""
    mol, universe, expected_indices, expected_names = mol_universe_and_expected
    dihedral_indices = chem.find_dihedral_indices(mol)
    assert len(dihedral_indices) == len(expected_indices)
    for idx, indices in enumerate(dihedral_indices):
        assert len(indices) == 4
        assert list(indices) == expected_indices[idx]
        atoms = [universe.atoms[i].name for i in indices]
        assert atoms == expected_names[idx]


def test_find_dihedral_indices_empty():
    """Test finding dihedral indices in a molecule with no rotatable bonds."""
    # Create a simple molecule with no rotatable bonds (e.g., methane)
    mol = rdkit.MolFromSmiles("C")
    dihedral_indices = chem.find_dihedral_indices(mol)
    assert len(dihedral_indices) == 0
