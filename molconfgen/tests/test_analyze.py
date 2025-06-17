import pytest
import MDAnalysis as mda
import numpy as np
import os
from .. import workflows
from .. import analyze
from ..data.files import V46_ITP, V46_PDB, V46_TOP

NUM_CONFORMERS = 6  # number of conformers to generate for testing


@pytest.fixture(scope="module")
def setup_workflow(tmpdir_factory):
    """Run the workflow to generate necessary input files for testing analyze functions."""
    tmpdir = tmpdir_factory.mktemp("test_analyze")
    output_prefix = str(tmpdir / "tst_V46")
    result = workflows.conformers_to_energies(
        itp_file=V46_ITP,
        pdb_file=V46_PDB,
        top_file=V46_TOP,
        num_conformers=NUM_CONFORMERS,
        output_prefix=output_prefix,
    )
    return result


def test_analyze_energies(setup_workflow):
    """Test the analyze_energies function."""
    edr_file = setup_workflow["energies"]
    energies = analyze.get_energies(edr_file)
    assert len(energies) == NUM_CONFORMERS
    assert isinstance(energies, np.ndarray)


def test_analyze_conformers(setup_workflow):
    """Test the analyze_conformers function."""
    angles, energies = analyze.analyze_conformers(
        setup_workflow["topology"],
        setup_workflow["conformers"],
        setup_workflow["energies"],
    )
    assert isinstance(angles, np.ndarray)
    assert isinstance(energies, np.ndarray)
    assert len(angles) == NUM_CONFORMERS
    assert len(energies) == NUM_CONFORMERS
    # simple test
    assert np.all((-180 <= angles) & (angles <= 180))
