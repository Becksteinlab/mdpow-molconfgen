"""Script to check dihedral indices for V46 molecule."""

import MDAnalysis as mda
from .. import chem
from ..data.files import V46_PDB, V46_ITP

# Load the universe
u = mda.Universe(V46_ITP, V46_PDB)

# Load molecule from universe
mol = chem.load_mol(u)

# Find dihedral indices
dihedral_indices = chem.find_dihedral_indices(mol)

print("Dihedral indices found:", dihedral_indices)

# Print atom names for each dihedral
for dihedral in dihedral_indices:
    print("\nDihedral atoms:")
    for idx in dihedral:
        atom = u.atoms[idx]
        print(f"  {atom.name} (index {idx})")
