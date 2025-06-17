"""Tests for the workflows module."""

import pytest
import MDAnalysis as mda
import numpy as np
import os
import tempfile
import rdkit

from .. import workflows
from .. import analyze
from ..data.files import V46_PDB, V46_ITP, V46_TOP


@pytest.fixture
def universe():
    """Create a test universe with V46 molecule."""
    return mda.Universe(V46_ITP, V46_PDB)


def test_conformers_to_energies(universe, num_conformers=10):
    """Test the conformers_to_energies workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Run the workflow with auto box size
        result = workflows.conformers_to_energies(
            itp_file=V46_ITP,
            pdb_file=V46_PDB,
            top_file=V46_TOP,
            num_conformers=num_conformers,
            box="auto",  # Let workflow determine box size
            rcoulomb=70.0,
            output_prefix=os.path.join(tmpdir, "test"),
        )

        # Check that all files exist
        assert os.path.exists(result["energies"])
        assert os.path.exists(result["topology"])
        assert os.path.exists(result["conformers"])

        # Check that we can read the energy file
        energies = analyze.get_energies(result["energies"])
        assert len(energies) == num_conformers
        assert isinstance(energies, np.ndarray)


def test_run_gromacs_energy_calculation(universe, num_conformers=10, box=200):
    """Test running GROMACS energy calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        # Create a test trajectory
        traj_file = os.path.join(tmpdir, "test.trr")
        mol, conformers, _ = workflows.run_sampler(
            universe,
            num_conformers=num_conformers,
            box=box,
            output_filename=traj_file,
        )

        output_prefix = os.path.join(tmpdir, "test")

        pdb_boxed = workflows.create_boxed_pdb(
            universe, output_prefix, box=box, rcoulomb=70.0
        )

        # Run energy calculation
        result = workflows.run_gromacs_energy_calculation(
            mdp_file=workflows.create_mdp_file("energy.mdp", rcoulomb=70.0),
            pdb_file=pdb_boxed,
            top_file=V46_TOP,
            trajectory_file=traj_file,
            output_prefix=output_prefix,
        )

        # Check that energy file exists
        assert os.path.exists(result["energies"])
        assert os.path.exists(result["topology"])

        # Check that we can read energies
        energies = analyze.get_energies(result["energies"])
        assert len(energies) == num_conformers
        assert isinstance(energies, np.ndarray)


def test_create_boxed_pdb(
    universe, reference_L=166.597, rcoulomb=70.0, box="auto"
):
    """Test the create_boxed_pdb function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = os.path.join(tmpdir, "test")
        pdb_with_box = workflows.create_boxed_pdb(
            universe, output_prefix, box, rcoulomb
        )
        assert os.path.exists(pdb_with_box)
        # Load the created PDB with MDAnalysis
        u = mda.Universe(pdb_with_box)
        # Check that the box dimensions are correct
        assert np.allclose(
            u.dimensions[3:], [90, 90, 90]
        )  # angles are 90 degrees
        assert np.allclose(
            u.dimensions[:3], [reference_L, reference_L, reference_L]
        )


def test_run_sampler(universe, num_conformers=5):
    """Test the run_sampler function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_filename = os.path.join(tmpdir, "test_sampler.trr")
        mol, conformers, filename = workflows.run_sampler(
            universe,
            num_conformers=num_conformers,
            output_filename=output_filename,
            box="auto",
            rcoulomb=70.0,
        )
        assert os.path.exists(output_filename)
        assert filename == output_filename
        assert conformers.trajectory.n_frames == num_conformers
        assert isinstance(mol, rdkit.Chem.rdchem.Mol)
