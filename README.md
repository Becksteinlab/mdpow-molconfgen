# Assessing conformational space of small molecules #

## Algorithm ##

1. find all N major torsions
2. generate all conformers by rotating all torsions in increments
   delta for a total of (2π/delta)^N conformers
3. write to a trajectory
4. evaluate the force field energy with `gmx mdrun -rerun`. 
5. find minima in the N-dimensional energy landscape

## Implementation notes ##

Load molecules with MDAnalysis.

Convert to RDKit molecule.

Perform torsion drive with
https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html


## Initial testing systems ##
### COW dataset ###

`/Users/oliver/Projects/Methods/SAMPL5/sampl5-distribution-water-cyclohexane/11_validation_dataset92`

- V36-methylacetate : 1 dihedral
- V46-2-methyl-1-nitrobenzene : steric hindrance
- V39-butylacetate : 4 dihedrals




   
