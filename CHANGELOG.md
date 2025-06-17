# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
The rules for this file:
  * entries are sorted newest-first.
  * summarize sets of changes - don't reproduce every git log comment here.
  * don't ever delete anything.
  * keep the format consistent:
    * do not use tabs but use spaces for formatting
    * 79 char width
    * YYYY-MM-DD date format (following ISO 8601)
  * accompany each entry with github issue/PR number (Issue #xyz)
-->

## 0.3.0

????-??-??

### Authors
- orbeckst

### Added
- add frames kwarg to output.write_pbc_trajectory() to select written frames;
  default is "all".
- allow to pass through **kwargs in analyze.conformers_energies() to Dihedral analysis

### Fixed
- fixed workflows.create_boxed_pdb() writing a whole trajectory to PDB

### Changed
<!-- Changes In Existing Functionality -->

### Deprecated
<!-- Soon-To-Be Removed Features -->

### Removed
<!-- Removed Features -->


## 0.2.0

2025-06-17

### Authors
- rich-squared
- orbeckst

### Added
- adds new output module with `output.write_pbc_trajectory()` to write trajectory with fixed box size.(Issue #2)
- adds new workflows module that includes function to generate conformations and calculate the potential energies with GROMACS
- add testing

### Changed
- add gromacswrapper as a new dependency; you also need to install GROMACS to be able to evaluate potential energies
- removed legacy setup.py and setup.cfg build system
- use versioningit instead of versioneer; default version in absence of version info is now 0.0.0

## 0.1.1

2024-09-30

### Authors
<!-- GitHub usernames of contributors to this release -->
- orbeckst

### Fixed
- forgot to ship versioneer.py (issue #1)


## 0.1.0

2024-09-30

### Authors
<!-- GitHub usernames of contributors to this release -->
- orbeckst

### Fixed
- Minor fixes: installs again.


## 0.0.1

2023-06-27

### Authors
<!-- GitHub usernames of contributors to this release -->
- orbeckst

### Added
- initial release with base functionality (but no testing)

### Fixed
<!-- Bug fixes -->

### Changed
<!-- Changes in existing functionality -->

### Deprecated
<!-- Soon-to-be removed features -->

### Removed
<!-- Removed features -->
