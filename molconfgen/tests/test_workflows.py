"""Tests for the workflows module."""

import pytest
import MDAnalysis as mda
import numpy as np
import os
import tempfile

from .. import workflows
from .. import analyze
from ..data.files import V46_PDB, V46_ITP, V46_TOP


@pytest.fixture
def universe():
    """Create a test universe with V46 molecule."""
    return mda.Universe(V46_ITP, V46_PDB)


def test_conformers_to_energies(universe):
    """Test the conformers_to_energies workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run the workflow
        ener = workflows.conformers_to_energies(
            itp_file=V46_ITP,
            pdb_file=V46_PDB,
            top_file=V46_TOP,
            num_conformers=3,
            output_prefix=os.path.join(tmpdir, "test")
        )
        
        # Check that the energy file exists
        assert os.path.exists(ener)
        
        # Check that we can read the energy file
        energies = analyze.get_energies(ener)
        assert len(energies) == 3
        assert isinstance(energies, np.ndarray)


def test_run_gromacs_energy_calculation(universe):
    """Test running GROMACS energy calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test trajectory
        traj_file = os.path.join(tmpdir, "test.trr")
        mol, conformers, _ = workflows.run_sampler(
            universe,
            num_conformers=2,
            output_filename=traj_file
        )
        
        # Run energy calculation
        edr_file = workflows.run_gromacs_energy_calculation(
            mdp_file=workflows.create_mdp_file("energy.mdp", rcoulomb=70.0),
            pdb_file=V46_PDB,
            top_file=V46_TOP,
            trajectory_file=traj_file,
            output_prefix=os.path.join(tmpdir, "test")
        )
        
        # Check that energy file exists
        assert os.path.exists(edr_file)
        
        # Check that we can read energies
        energies = analyze.get_energies(edr_file)
        assert len(energies) == 2
        assert isinstance(energies, np.ndarray) 