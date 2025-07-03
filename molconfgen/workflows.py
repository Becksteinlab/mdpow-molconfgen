"""Workflow functions for generating conformers and calculating their energies using GROMACS.

This module provides functions to:
1. Generate conformers of small molecules by sampling torsional angles
2. Calculate a rough estimate of conformational stability using GROMACS potential energies

The workflow uses MDAnalysis and RDKit for conformer generation, and GROMACS for
energy calculations within a given force field. Note that potential energies in
vacuum (without explicit solvent) provide only a rough estimate of conformational
stability and should not be used for quantitative predictions of relative
conformer populations.

Example
-------
To generate conformers and calculate their approximate energies for V46-2-methyl-1-nitrobenzene:

    from molconfgen import workflows

    # Generate conformers and calculate rough energy estimates in one step
    workflows.generate_and_simulate(
        itp_file="V46-2-methyl-1-nitrobenzene.itp",
        pdb_file="V46_bigbox.pdb",
        mdp_file="V46.mdp",
        top_file="V46.top",
        num_conformers=36,
        output_prefix="V46"
    )

    # The function will create the following files:
    # - V46_conformers.trr: Trajectory containing all generated conformers
    # - V46_mdout.mdp: GROMACS mdp output file
    # - V46topol.tpr: GROMACS topology file
    # - V46_traj.trr: Final trajectory (same as input, used for energy calculation)
    # - V46_ener.edr: Energy file containing potential energies for each conformer
    #   (uses gmx energy to extract the energies)

Alternatively, you can use the individual functions for more control:

    import MDAnalysis as mda
    from molconfgen import workflows

    # Load the molecule
    universe = mda.Universe("V46-2-methyl-1-nitrobenzene.itp", "V46_bigbox.pdb")

    # Generate conformers by sampling torsional angles
    mol, conformers, traj_file = workflows.run_sampler(
        universe,
        num_conformers=36,
        output_filename="V46_conformers.trr",
        box_size=150.0
    )

    # Calculate approximate energies for each conformer using GROMACS
    workflows.run_gromacs_simulation(
        mdp_file="V46.mdp",
        pdb_file="V46_bigbox.pdb",
        top_file="V46.top",
        trajectory_file=traj_file,
        output_prefix="V46"
    )

Notes
-----
- The input files (itp, pdb, mdp, top) must exist in the current directory
- The box_size parameter should match your simulation box size
- The number of conformers can be adjusted based on your needs
- All output files will be prefixed with the output_prefix parameter
- The energy file (ener.edr) contains the potential energy for each conformer
  in the trajectory. Use GROMACS tools (e.g., ``gmx energy) to extract the energies.
- The mdp file should be configured for single-point energy calculation
  (e.g., integrator = md, nsteps = 0)
- The  energies can be calculated as vacuum energies (for ``epsilon-r = 1``) or with the dielectric constant of water, ``epsilon-r = 80`` and provide only a rough
  estimate of conformational stability. For more accurate results, consider
  using explicit solvent simulations or other methods that account for
  solvation effects (e.g., implicit solvent models or explicit solvent free energy simulations).
"""

import MDAnalysis as mda
import numpy as np
import pathlib
from typing import Tuple, Optional, List, Union
from string import Template
import rdkit.Chem
import logging

from . import sampler, chem, output

logger = logging.getLogger("molconfgen")

# MDP file template
MDP_TEMPLATE = Template(
    """; gromacs mdp file for energy calculations
; created by molconfgen

$include_statement

integrator               = md
dt                       = 0.002
nsteps                   = 0
nstxout                  = 0 ; write coords every # step
constraints              = none

pbc                      = xyz
periodic_molecules       = no

coulombtype              = Cut-off
rcoulomb                 = $rcoulomb
epsilon-r                = $epsilon_r
epsilon_surface          = 0

vdwtype                  = cut-off   ; use shift for L-BFGS
rvdw                     = $rcoulomb ; must match rcoulomb for Verlet lists
rvdw-switch              = 0         ; 0.8 for l-bfcg

Tcoupl                   = no
Pcoupl                   = no
gen_vel                  = no
                        
continuation             = yes
"""
)


def create_mdp_file(
    output_file: str,
    include_paths: Optional[List[str]] = None,
    rcoulomb: float = 70.0,
    epsilon_r: float = 80.0,
) -> str:
    """create a gromacs mdp file for energy calculations.

    parameters
    ----------
    output_file : str
        path where the mdp file will be written
    include_paths : list[str], optional
        list of paths to include in the mdp file. if none, only current directory is included.
    rcoulomb : float, optional
        coulomb cutoff radius in Angstrom, by default 70.0 Angstrom
        for vacuum or dielectric constant = 80 (water) calculations
    epsilon_r : float, optional
        dielectric constant, by default 80.0 (water)

    returns
    -------
    str
        path to the created mdp file
    """
    logger.info(f"Creating MDP file at {output_file} (rcoulomb={rcoulomb}, epsilon_r={epsilon_r})")
    if include_paths is None:
        include_paths = ["."]

    # create include statement
    include_statement = "include = " + " ".join(
        f"-I{path}" for path in include_paths
    )

    # substitute template variables
    mdp_content = MDP_TEMPLATE.substitute(
        include_statement=include_statement,
        rcoulomb=rcoulomb / 10.0,  # convert to nm for GROMACS
        epsilon_r=epsilon_r,
    )

    with open(output_file, "w") as f:
        f.write(mdp_content)

    return output_file


def create_boxed_pdb(universe, output_prefix, box, rcoulomb):
    """Write a PDB file with the specified box size for GROMACS.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the molecule structure
    output_prefix : str
        Prefix for the output PDB file
    box : float, array_like, 'auto', or None
        Box size specification
    rcoulomb : float
        Coulomb cutoff radius in Angstrom

    Returns
    -------
    str
        Path to the created PDB file
    """
    logger.info(f"Writing boxed PDB file with prefix '{output_prefix}' and box={box}")
    pdb_with_box = f"{output_prefix}_boxed.pdb"
    output.write_pbc_trajectory(
        universe, pdb_with_box, box=box, rcoulomb=rcoulomb, frames=0
    )
    return pdb_with_box


def run_sampler(
    universe: mda.Universe,
    num_conformers: int = 36,
    output_filename: str = "conformers.trr",
    box: Optional[Union[float, List[float], str]] = "auto",
    rcoulomb: float = 70.0,
) -> Tuple[rdkit.Chem.rdchem.Mol, mda.Universe, str]:
    """Generate conformers for a molecule and write them to a trajectory file.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the molecule structure
    num_conformers : int, optional
        Number of conformers to generate, by default 36
    output_filename : str, optional
        Name of the output trajectory file, by default "conformers.trr"
    box : float, array_like, 'auto', or ``None``, optional
        There are four different options here to allow for customization
        of the box (default is "auto"):
        - ``None: leaves the trajectory unmodified
        - 'auto': calls :func:`largest_r and adds the value of `rcoulomb` and `buffer`
        - float: assumes the box is a cube with side lengths equal to the input
        - array_like: must be a 1x6 array with the first three entries
          representing the sides of the box and the last three entries
          representing the angles between them
    rcoulomb : float, optional
        coulomb cutoff radius in Angstrom, by default 70.0 Angstrom
        for vacuum or dielectric constant = 80 (water) calculations.
        This is used to define the box for the trajectory together with `box`.

    Returns
    -------
    Tuple[rdkit.Chem.rdchem.Mol, mda.Universe, str]
        The molecule object, conformer universe, and output filename
    """
    logger.info(f"Generating {num_conformers} conformers and writing to {output_filename}")
    mol = chem.load_mol(universe, add_labels=True)
    dihedrals = chem.find_dihedrals(mol, universe)

    conformers = sampler.generate_conformers(mol, dihedrals, num=num_conformers)
    output.write_pbc_trajectory(
        conformers, output_filename, box=box, rcoulomb=rcoulomb
    )
    return mol, conformers, output_filename


def run_gromacs_energy_calculation(
    mdp_file: str,
    pdb_file: str,
    top_file: str,
    trajectory_file: str,
    output_prefix: str = "simulation",
) -> dict:
    """Run a GROMACS energy calculation using a pre-generated trajectory.

    The energy calculation is performed with ``mdrun -rerun`` and the output is written to an ``edr`` file.

    We generate a suitable TPR file from the MDP, TOP, and PDB files. The TOP file should reference the ITP file.
    If it is located in the same location as the ITP file then ``grompp`` will automatically find the ITP file.
    Otherwise, adjust the search path in the MDP file's ``include`` statement.

    Parameters
    ----------
    mdp_file : str
        Path to the GROMACS mdp file
    pdb_file : str
        Path to the PDB file
    top_file : str
        Path to the GROMACS topology file (.top)
    trajectory_file : str
        Path to the trajectory file
    output_prefix : str, optional
        Prefix for output files, by default "simulation"

    Returns
    -------
    dict
        Dictionary containing:
        - energies: Path to the energy file (EDR)
        - topology: Path to the generated topology file (TPR)
    """
    import gromacs

    logger.info(f"Running GROMACS energy calculation with prefix '{output_prefix}'")
    # Generate the tpr file
    gromacs.grompp(
        f=mdp_file,
        c=pdb_file,
        p=top_file,
        po=f"{output_prefix}_mdout.mdp",
        o=f"{output_prefix}_topol.tpr",
    )

    # Run the energy calculation
    gromacs.mdrun(
        s=f"{output_prefix}_topol.tpr",
        rerun=trajectory_file,
        o=f"{output_prefix}_traj.trr",
        e=f"{output_prefix}_ener.edr",
    )

    return {
        "energies": f"{output_prefix}_ener.edr",
        "topology": f"{output_prefix}_topol.tpr",
    }


def conformers_to_energies(
    itp_file: str,
    pdb_file: str,
    top_file: str,
    mdp_file: Optional[str] = None,
    num_conformers: int = 36,
    box: Optional[Union[float, List[float], str]] = "auto",
    rcoulomb: float = 70.0,
    output_prefix: str = "simulation",
) -> dict:
    """Generate conformers and calculate their energies using GROMACS.

    Parameters
    ----------
    itp_file : str
        Path to the GROMACS ITP file (used as a topology file for MDAnalysis)
    pdb_file : str
        Path to the PDB file
    top_file : str
        Path to the GROMACS topology file (.top)
    mdp_file : str, optional
        Path to the GROMACS mdp file. If None, a default mdp file is created.
    num_conformers : int, optional
        Number of conformers to generate, by default 36
    box : float, array_like, 'auto', or ``None``, optional
        There are four different options here to allow for customization
        of the box (default is "auto"):
        - ``None: leaves the trajectory unmodified
        - 'auto': calls :func:`largest_r and adds the value of `rcoulomb` and `buffer`
        - float: assumes the box is a cube with side lengths equal to the input
        - array_like: must be a 1x6 array with the first three entries
          representing the sides of the box and the last three entries
          representing the angles between them
    rcoulomb : float, optional
        coulomb cutoff radius in Angstrom, by default 70.0 Angstrom
        for vacuum or dielectric constant = 80 (water) calculations.
        This is also used to define the box for the trajectory together with `box`.
    output_prefix : str, optional
        Prefix for output files, by default "simulation"

    Returns
    -------
    dict
        Dictionary containing:
        - topology: Path to the topology file
        - conformers: Path to the conformer trajectory file
        - energies: Path to the energy file
    """
    logger.info(f"Generating conformers and calculating energies (output_prefix='{output_prefix}', num_conformers={num_conformers})")
    universe = mda.Universe(itp_file, pdb_file)

    # Generate conformers
    # ------------------
    trajectory_file = f"{output_prefix}_conformers.trr"
    _, conformer_universe, conformer_trr = run_sampler(
        universe,
        num_conformers=num_conformers,
        output_filename=trajectory_file,
        box=box,
        rcoulomb=rcoulomb,
    )

    # Evaluate energies with GROMACS
    # ------------------------------
    # Create MDP file if not provided
    if mdp_file is None:
        mdp_file = create_mdp_file(f"{output_prefix}.mdp", rcoulomb=rcoulomb)

    # need input system with sufficiently large box for GROMACS
    # (calculate box size from largest_r and add buffer as for run_sampler)
    pdb_with_box = create_boxed_pdb(
        conformer_universe, output_prefix, box, rcoulomb
    )

    # Run GROMACS energy calculation
    result = run_gromacs_energy_calculation(
        mdp_file=mdp_file,
        pdb_file=pdb_with_box,
        top_file=top_file,
        trajectory_file=conformer_trr,
        output_prefix=output_prefix,
    )

    return {
        "topology": result["topology"],
        "conformers": conformer_trr,
        "energies": result["energies"],
    }
