"""
Location of data files
======================

Use as ::

    from molconfgen.data.files import *

"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
]

import importlib.resources as resources

MDANALYSIS_LOGO = str(resources.files(__name__) / "mda.txt")
