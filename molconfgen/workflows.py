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
from typing import Tuple, Optional, List
from string import Template

from . import sampler, chem, output

# MDP file template
MDP_TEMPLATE = Template("""; gromacs mdp file for energy calculations
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
rvdw                     = 1.0       ; fixed value for force field
rvdw-switch              = 0         ; 0.8 for l-bfcg

Tcoupl                   = no
Pcoupl                   = no
gen_vel                  = no
""")

def create_mdp_file(output_file: str,
                   include_paths: Optional[List[str]] = None,
                   rcoulomb: float = 7.0,
                   epsilon_r: float = 80.0) -> str:
    """create a gromacs mdp file for energy calculations.
    
    parameters
    ----------
    output_file : str
        path where the mdp file will be written
    include_paths : list[str], optional
        list of paths to include in the mdp file. if none, only current directory is included.
    rcoulomb : float, optional
        coulomb cutoff radius in nm, by default 7.0
    epsilon_r : float, optional
        dielectric constant, by default 80.0 (water)
    
    returns
    -------
    str
        path to the created mdp file
    """
    if include_paths is None:
        include_paths = ["."]
    
    # create include statement
    include_statement = "include = " + " ".join(f"-I{path}" for path in include_paths)
    
    # substitute template variables
    mdp_content = MDP_TEMPLATE.substitute(
        include_statement=include_statement,
        rcoulomb=rcoulomb,
        epsilon_r=epsilon_r
    )
    
    with open(output_file, 'w') as f:
        f.write(mdp_content)
    
    return output_file

def run_sampler(universe: mda.Universe, num_conformers: int = 12, 
                output_filename: str = "conformers.trr",
                box_size: float = 150.0) -> Tuple[chem.Molecule, mda.Universe, str]:
    """Generate conformers for a molecule and write them to a trajectory file.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the molecule structure
    num_conformers : int, optional
        Number of conformers to generate, by default 12
    output_filename : str, optional
        Name of the output trajectory file, by default "conformers.trr"
    box_size : float, optional
        Size of the periodic box in Angstroms, by default 150.0
    
    Returns
    -------
    Tuple[chem.Molecule, mda.Universe, str]
        The molecule object, conformer universe, and output filename
    """
    mol = chem.load_mol(universe, add_labels=True)
    dihedrals = chem.find_dihedrals(mol, universe)

    conformers = sampler.generate_conformers(mol, dihedrals, num=num_conformers)
    output.write_pbc_trajectory(conformers, output_filename, box=box_size)
    return mol, conformers, output_filename

def run_gromacs_energy_calculation(trajectory_file: str,
                                   output_prefix: str = "simulation") -> None:
    """Run a GROMACS energy calculation using a pre-generated trajectory.

    The energy calculation is performed with ``mdrun -rerun`` and the output is written to an ``edr`` file.
    
    Parameters
    ----------
    mdp_file : str
        Path to the GROMACS mdp file
    pdb_file : str
        Path to the input PDB file
    top_file : str
        Path to the topology file
    trajectory_file : str
        Path to the input trajectory file
    output_prefix : str, optional
        prefix for output files, by default "simulation"

    Returns
    -------
    str
        Path to the energy file
    """
    import gromacs
    
    # Generate the tpr file
    gromacs.grompp(f=mdp_file, c=pdb_file, p=top_file,
                   po=f"{output_prefix}_mdout.mdp",
                   o=f"{output_prefix}_topol.tpr")
    
    # Run the simulation
    gromacs.mdrun(s=f"{output_prefix}_topol.tpr",
                  rerun=trajectory_file,
                  o=f"{output_prefix}_traj.trr",
                  e=f"{output_prefix}_ener.edr")

    return f"{output_prefix}_ener.edr"

def generate_and_simulate(itp_file: str, pdb_file: str, mdp_file: str, top_file: str,
                         num_conformers: int = 12,
                         box_size: float = 150.0,
                         output_prefix: str = "simulation") -> None:
    """Generate conformers and run a GROMACS simulation in one workflow.
    
    Parameters
    ----------
    itp_file : str
        Path to the ITP file
    pdb_file : str
        Path to the PDB file
    mdp_file : str
        Path to the GROMACS mdp file
    top_file : str
        Path to the topology file
    num_conformers : int, optional
        Number of conformers to generate, by default 12
    box_size : float, optional
        Size of the periodic box in Angstroms, by default 150.0
    output_prefix : str, optional
        Prefix for output files, by default "simulation"

    Returns
    -------
    str
        Path to the energy file
    """
    # Load the universe
    universe = mda.Universe(itp_file, pdb_file)
    
    # Generate conformers
    trajectory_file = f"{output_prefix}_conformers.trr"
    mol, conformers, _ = run_sampler(universe, 
                                   num_conformers=num_conformers,
                                   output_filename=trajectory_file,
                                   box_size=box_size)
    
    # Run GROMACS simulation
    energy_file = run_gromacs_energy_calculation(mdp_file, pdb_file, top_file,
                                                 trajectory_file,
                                                 output_prefix=output_prefix) 
    return energy_file