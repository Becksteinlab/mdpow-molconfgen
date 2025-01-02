# Richard Richardson wrote this ;)
# Function to add a box to the output of
# sampler.generate_conformers

import MDAnalysis as mda
import MDAnalysis.transformations
import numpy as np


# Function for finding the largest radius in
# the trajectory
# this will come in handy later
def largest_r(u):
    r = 0
    for ts in u.trajectory:
        ag = u.atoms
        rnew = ag.bsphere()[0]
        if rnew > r:
            r = rnew
    return r

def write_pbc_trajectory(u, filename, box = None):
    ''' u is MDAnalysis universe with a trajectory
     filename is the name of the trajectory output file
     box argument describes how to set the box dimensions

     I am not quite sure how to read a lone trajectory file
     and sampler.generate_conformers returns a universe with
     a trajectroy so for now I'm going to assume a universe 
     with a trajectory '''

# first case is for if they want no box
# just return the boxless trajectory I guess
    if box == None:
        u.atoms.write(filename, frames = "all")
        return filename    

# second case is for if they want a box but 
# don't define it themselves
# here we define a "good" box as 3 times the 
# largest bsphere radius
    if box == 'auto':
        # call largest_r to find the largest r
        r = largest_r(u)
        dim = np.array([3*r,3*r,3*r,90,90,90])
        transform = mda.transformations.boxdimensions.set_dimensions(dim)
        u.trajectory.add_transformations(transform)
        u.atoms.write(filename, frames = "all")
        return filename


# third case is for if they input a float for the box dimensions
# if they input a box that is too small, return an error message?
# smallest minimum box is more than 2 times the largest bsphere radius
    if isinstance(box,float) or isinstance(box,int):
        r = largest_r(u)
        if box <= 2*r:
            raise ValueError('your box should probably be a bit bigger')
        else:
            dim = np.array([box,box,box,90,90,90])
            transform = mda.transformations.boxdimensions.set_dimensions(dim)
            u.trajectory.add_transformations(transform)
            u.atoms.write(filename, frames = "all")
            return filename


# fourth case is for if they input a regular box discription:
# [A, B, C. alpha, beta, gamma]
# I'm not entirely sure how to check for some sort of irregular box,
# or how to even check if the input is correct for this one
# so we'll just leave it for now and treat this as a rough draft.
# I'll come back and work on it after christmas, and in the 
# meantime there's other parts of this project I can work on


        



