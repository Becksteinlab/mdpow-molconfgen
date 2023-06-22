# -*- coding: utf-8 -*-
# simple chemical analysis of molecules

import numpy as np
import rdkit.Chem


def unique_torsions(dihedral_atom_indices):
    """Return dihedrals that have a unique central bond.

    A central bond (i, j) is unique if there is no other
    dihedral (a, i, j, b) or (a', j, i, b') in the list;
    i.e., the direction of the bond is irrelevant.

    Arguments
    ---------
    dihedral_atom_indices : list of tuples
        List of 4-tuples, each containing the 4 atom indices
        that form the dihedral. Can also be an array of shape
        (N, 4) for N dihedrals.

    Returns
    -------
    unique : np.array
        The atom indices organized in a 2D array of shape (M, 4)
        where M ≤ N and no two central bonds are the same.
    """
    dihedral_atom_indices = np.asarray(dihedral_atom_indices)
    sorted_centrals = np.sort(dihedral_atom_indices[:, 1:3], axis=1)
    unique_bonds, dihedral_indices = np.unique(sorted_centrals, axis=0, return_index=True)
    return dihedral_atom_indices[dihedral_indices]


def find_dihedral_indices(
                   mol, unique=True,
                   SMARTS='[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]'):
    """Extract indices of all dihedrals in a molecule.

    Arguments
    ---------
    mol : rdkit molecule
       molecule
    unique : bool
       prune the results to only return unqiue torsions, i.e., no
       torsions that contain the same central bond
    SMARTS : str
        selection string

    Returns
    -------
    indices : list
         list of 4-tuples, each describing a dihedral in `mol`
    """
    pattern = rdkit.Chem.MolFromSmarts(SMARTS)
    atom_indices = mol.GetSubstructMatches(pattern)
    if unique:
        atom_indices = unique_torsions(atom_indices)
    return atom_indices
