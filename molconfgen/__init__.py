"""
mdpow-molconfgen
================

Generation of conformers of small molecules.
"""

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


# Add imports here
from . import chem
from . import sampler


from . import _version
__version__ = _version.get_versions()['version']
