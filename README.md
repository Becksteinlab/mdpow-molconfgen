# mdpow-molconfgen #

[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/becksteinlab/mdpow-molconfgen.svg)](https://github.com/becksteinlab/mdpow-molconfgen/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/becksteinlab/mdpow-molconfgen/latest)  [![Documentation Status](https://readthedocs.org/projects/mdpow-molconfgen/badge/?version=latest)](https://mdpow-molconfgen.readthedocs.io/en/latest/?badge=latest)|
| :------ | :------- |
| **Status** | [![GH Actions Status](https://github.com/becksteinlab/mdpow-molconfgen/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/becksteinlab/mdpow-molconfgen/actions?query=branch%3Amain+workflow%3Agh-ci) [![codecov](https://codecov.io/gh/becksteinlab/mdpow-molconfgen/branch/main/graph/badge.svg)](https://codecov.io/gh/becksteinlab/mdpow-molconfgen/branch/main) |
| **Community** | [![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](https://www.gnu.org/licenses/gpl-2.0)  [![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)|

Generation of conformers of small molecules.

## Background ##

1. find all N major torsions
2. generate all conformers by rotating all torsions in increments
   delta for a total of (2π/delta)^N conformers
3. write to a trajectory
4. evaluate the force field energy with `gmx mdrun -rerun`. 
5. find minima in the N-dimensional energy landscape

### Implementation notes ###

1. Load molecules with MDAnalysis.
2. Convert to RDKit molecule.
3. Perform torsion drive with [rdkit.Chem.rdMolTransforms](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html)


### Initial testing systems ###
From the [COW dataset](https://github.com/Becksteinlab/sampl5-distribution-water-cyclohexane/tree/master/11_validation_dataset92): 

- V36-methylacetate : 1 dihedral
- V46-2-methyl-1-nitrobenzene : steric hindrance
- V39-butylacetate : 4 dihedrals

## First steps

### Community

mdpow-molconfgen is bound by a [Code of Conduct](https://github.com/becksteinlab/mdpow-molconfgen/blob/main/CODE_OF_CONDUCT.md).

### Installation

To build mdpow-molconfgen from source,
we highly recommend using virtual environments.
If possible, we strongly recommend that you use 
[mamba](https://mamba.readthedocs.io/en/latest/index.html) as your package manager.
Below we provide instructions both for `mamba` and
for `pip`.

#### With mamba

Ensure that you have [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installed.

Create a virtual environment and activate it:

```
mamba create --name mdpow-molconfgen
mamba activate mdpow-molconfgen
```

Install the development and documentation dependencies:

```
mamba env update --name mdpow-molconfgen --file devtools/conda-envs/test_env.yaml
mamba env update --name mdpow-molconfgen --file docs/requirements.yaml
```

Build this package from source:

```
pip install -e .
```

If you want to update your dependencies (which can be risky!), run:

```
mamba update --all
```

And when you are finished, you can exit the virtual environment with:

```
mamba deactivate
```

#### With pip

To build the package from source, run:

```
pip install -e .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```
pip install -e ".[test,doc]"
```

### Copyright

The mdpow-molconfgen source code is hosted at https://github.com/becksteinlab/mdpow-molconfgen
and is available under the GNU General Public License, version 2 (see the file [LICENSE](https://github.com/becksteinlab/mdpow-molconfgen/blob/main/LICENSE)).

Copyright (c) 2023, Oliver Beckstein


#### Acknowledgements
 
Project based on the 
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using mdpow-molconfgen in published work.

