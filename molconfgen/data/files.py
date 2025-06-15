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
    # V36-methylacetate files
    "V36_PDB",
    "V36_ITP",
    "V36_TOP",
]

import importlib.resources as resources

# V46-2-methyl-1-nitrobenzene files
V46_DIR = resources.files(__name__) / "V46-2-methyl-1-nitrobenzene"
V46_MOL2 = V46_DIR / "V46-2-methyl-1-nitrobenzene.mol2"
V46_PDB = V46_DIR / "V46-2-methyl-1-nitrobenzene.pdb"
V46_ITP = V46_DIR / "V46-2-methyl-1-nitrobenzene.itp"
V46_TOP = V46_DIR / "V46.top"

# V36-methylacetate files
V36_DIR = resources.files(__name__) / "V36-methylacetate"
V36_PDB = V36_DIR / "V36.pdb"
V36_ITP = V36_DIR / "V36-methylacetate.itp"
V36_TOP = V36_DIR / "V36.top"
