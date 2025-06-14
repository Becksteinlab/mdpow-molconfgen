"""Tests for the sampler module."""

import pytest
import MDAnalysis as mda
import numpy as np

from .. import sampler
from .. import chem
from ..data.files import V46_PDB, V46_ITP


@pytest.fixture
def universe():
    """Create a test universe with V46 molecule."""
    return mda.Universe(V46_ITP, V46_PDB)


def test_generate_conformers(universe):
    """Test generating conformers for V46."""
    # Create a list of dihedrals for V46
    dih = universe.atoms[[2, 1, 0, 4]].dihedral
    dihedrals = [dih]
    
    # Convert Universe to RDKit molecule
    mol = chem.load_mol(universe)
    
    # Generate conformers
    u = sampler.generate_conformers(mol, dihedrals, num=5)
    
    # Check that we got the expected number of conformers
    assert len(u.trajectory) == 5
    
    # Check that the molecule is an RDKit molecule
    assert hasattr(mol, 'GetNumAtoms')
    assert mol.GetNumAtoms() == universe.atoms.n_atoms





