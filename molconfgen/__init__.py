"""
mdpow-molconfgen
================

Generation of conformers of small molecules.
"""

# Version is handled by versioningit
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# Add imports here
from . import chem
from . import sampler

