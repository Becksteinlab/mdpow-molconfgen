# -*- coding: utf-8 -*-
# simple chemical analysis of molecules

import numpy as np
import rdkit.Chem
import MDAnalysis as mda


def load_mol(u, add_labels=False):
    """Create RDKIT mol from a Universe.

    If elements are missing from the Universe, they are guessed and
    added.

    .. Note:: The *whole* Universe is transformed into a molecule.  If
              subselections are required, this code needs to be
              changed.

    Arguments
    ---------
    u : Universe
        MDAnalysis universe containing topology and
        coordinates
    add_labels : bool, optiona;
        add MDAnalysis atom index labels to the `mol`

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object

    """

    try:
        mol = u.atoms.convert_to("RDKIT")
    except AttributeError:
        from MDAnalysis.topology.guessers import guess_types

        u.add_TopologyAttr("elements", guess_types(u.atoms.names))
        mol = u.atoms.convert_to("RDKIT")

    # TODO: we could check if coordinates are present and if not,
    # generate coordinates with RDKIT.

    if add_labels:
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))

    return mol


def load_mdpow_mol(topdir, add_labels=False):
    """Load Universe and RDKIT mol from MDPOW directory.

    This function assumes that there is one ITP file and one PDB file
    in the directory. These are used as the topology and coordinate
    files.

    Thus, the universe only contains the solute molecule (from the ITP
    file).

    Arguments
    ---------
    topdir : pathlib.Path
        top directory containing ITP and PDB file
    add_labels : bool, optiona;
        add MDAnalysis atom index labels to the `mol`

    Returns
    -------
    universe : Universe
        MDAnalysis Universe data structure
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object

    """

    topol = list(topdir.glob("*.itp"))[0]
    coords = list(topdir.glob("*.pdb"))[0]
    u = mda.Universe(topol, coords)
    return u, load_mol(u, add_labels=add_labels)


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
        Returns an empty array if no dihedrals are provided.
    """
    dihedral_atom_indices = np.asarray(dihedral_atom_indices)
    if len(dihedral_atom_indices) == 0:
        return np.array([], dtype=int).reshape(0, 4)
    sorted_centrals = np.sort(dihedral_atom_indices[:, 1:3], axis=1)
    unique_bonds, dihedral_indices = np.unique(
        sorted_centrals, axis=0, return_index=True
    )
    return dihedral_atom_indices[dihedral_indices]


def find_dihedral_indices(
    mol, unique=True, SMARTS="[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]"
):
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


def find_dihedrals(mol, universe, unique=True):
    """Find dihedrals in mol and return MDAnalysis dihedral objects.

    Arguments
    ---------
    mol : Molecule
       rdkit molecule of the compound
    universe : MDAnalysis.Universe
       The Universe from which `mol` was obtained.
       Atom indices must match!
    unique : bool
       Only return unique torsions (no duplicated
       central bonds).

    Returns
    -------
    dihedrals : list
       list of :class:`MDAnalysis.core.topologyobjects.Dihedral`
       instances, selected via the dihedral indices from `universe`


    .. SeeAlso:: :func:`find_dihedral_indices`
    """
    dh_ix = find_dihedral_indices(mol, unique=unique)
    return [universe.atoms[indices].dihedral for indices in dh_ix]
