# -*- coding: utf-8 -*-
# exhaustive small molecule dihedral sampling

import pathlib
import itertools

import numpy as np
from tqdm.auto import tqdm

import rdkit
from rdkit.Chem.rdMolTransforms import SetDihedralDeg


def anglespace(start, num, total=360.0):
    """generate num-1 equally spaced angles in degree over the range total, starting at start"""
    return start + np.linspace(0, total, num, endpoint=False)


def dihedralangles(dihedral, num, **kwargs):
    phi0 = dihedral.value()
    return anglespace(phi0, num, **kwargs)


def generate_conformers(mol, dihedrals, num=12):
    """Create a Universe with all dihedrals in a in-memory trajectory.

    The resulting trajectory will contain :math:`N^M` frames were
    :math:`N` is the number of dihedrals given in `dihedrals` and `M`
    is the number of intervals `num`.

    Arguments
    ---------
    mol : rdkit.Chem.rdchem.Mol
        molecule to be sampled, must match the
        molecule from which the dihedrals were obtained

    dihedrals : list
        list of MDAnalysis Dihedral instances

    num : int, opt
        sample a complete 2π dihedral at num-1 regular intervals
        of size 2π/num; `num` will be used for all `dihedrals`

    Returns
    -------
    u : Universe
       The Universe is a copy of the original Universe but with an
       in-memory trajectory that contains all conformers.

    Example
    -------
    Given a universe ``u``, generate a list of dihedrals::

       dih_1 = u.atoms[[10, 0, 1, 3]].dihedral
       dih_2 = u.atoms[[7, 10, 0, 1]].dihedral
       dihedrals = [dih_2, dih_1]

    Get a molecule ::

       mol = load_mol(u)

    Perform the conformer generation::

       samples = generate_conformers(mol, dihedrals, num=12)

    Write a trajectory of the conformers to a TRR trajectory file::

       samples.atoms.write("conformers.trr", frames="all")

    The *conformer.trr* file can be used together with the topology
    file that was used to generate the universe ``u`` in the first
    place.

    """
    u = dihedrals[0].atoms.universe.copy()

    phi_ranges = [dihedralangles(d, num) for d in dihedrals]
    n_frames = np.prod([len(x) for x in phi_ranges])

    trajectory = np.empty((n_frames, u.atoms.n_atoms, 3), dtype=np.float32)

    conformer = mol.GetConformer()

    for i_frame, phis in tqdm(
        enumerate(itertools.product(*phi_ranges)), total=n_frames
    ):
        for dih, phi in zip(dihedrals, phis):
            # set all dihedrals; tolist() ensures we get a standard Python int
            # because np.int64 won't work with rdkit
            SetDihedralDeg(conformer, *dih.atoms.indices.tolist(), phi)
        trajectory[i_frame] = conformer.GetPositions()

    u.load_new(trajectory)

    return u
