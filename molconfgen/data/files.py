"""
Location of data files
======================

Use as ::

    from molconfgen.data.files import *

"""

__all__ = [
    # V46-2-methyl-1-nitrobenzene files
    "V46_MOL2",
    "V46_PDB",
    "V46_ITP",
    "V46_TOP",
]

import importlib.resources as resources

# V46-2-methyl-1-nitrobenzene files
V46_DIR = resources.files(__name__) / "V46-2-methyl-1-nitrobenzene"
V46_MOL2 = V46_DIR / "V46-2-methyl-1-nitrobenzene.mol2"
V46_PDB = V46_DIR / "V46-2-methyl-1-nitrobenzene.pdb"
V46_ITP = V46_DIR / "V46-2-methyl-1-nitrobenzene.itp"
V46_TOP = V46_DIR / "V46.top"
