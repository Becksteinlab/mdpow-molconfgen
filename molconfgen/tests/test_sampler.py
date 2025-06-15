"""Tests for the sampler module."""

import pytest
import MDAnalysis as mda
import numpy as np

from .. import sampler
from .. import chem
from ..data.files import V46_PDB, V46_ITP, V36_ITP, V36_PDB, V39_ITP, V39_PDB


@pytest.fixture(
    params=[
        ("V46", V46_ITP, V46_PDB, [[2, 1, 0, 4]]),  # V46 dihedral indices
        (
            "V36",
            V36_ITP,
            V36_PDB,
            [[7, 10, 0, 1], [10, 0, 1, 3]],
        ),  # V36 dihedral indices
        (
            "V39",
            V39_ITP,
            V39_PDB,
            [[2, 0, 1, 3], [0, 1, 3, 14], [1, 3, 14, 17], [3, 14, 17, 9]],
        ),  # V39 dihedral indices
    ]
)
def molecule_data(request):
    """Create test universes for V46, V36, and V39 molecules."""
    name, itp_file, pdb_file, dihedral_indices = request.param
    universe = mda.Universe(itp_file, pdb_file)
    return name, universe, dihedral_indices


def test_generate_conformers(molecule_data, num_conformers=5):
    """Test generating conformers for V46, V36, and V39 molecules."""
    name, universe, dihedral_indices = molecule_data

    # Create a list of dihedrals using a list comprehension
    dihedrals = [
        universe.atoms[indices].dihedral for indices in dihedral_indices
    ]

    # Convert Universe to RDKit molecule
    mol = chem.load_mol(universe)

    # Generate conformers
    u = sampler.generate_conformers(mol, dihedrals, num=num_conformers)

    # Check that we got the expected number of conformers
    assert len(u.trajectory) == num_conformers ** len(dihedral_indices)

    # Check that the molecule is an RDKit molecule
    assert hasattr(mol, "GetNumAtoms")
    assert mol.GetNumAtoms() == universe.atoms.n_atoms
