#!/usr/bin/env python

import sys
import numpy as np
from hbond_definition import *
from read_geometries import read_geoms
from group_by_molecule import group_by_molecule

try:
    ifile = sys.argv[1]
except:
    print("I need an xyz formatted file of water molecules. Exiting.")
    sys.exit(1)

def findWater(geom, Num):
    sortedIndices = []
    for i, line in enumerate(geom):
        dist = np.linalg.norm(geom[Num] - line)
        if dist > 0.01 and dist < 1.20: #assuming angstroms
            sortedIndices.append(i)
    return sortedIndices

def sort_cluster(atoms, indices):
    '''
    Takes a list of nummpy arrays with each array having xyz coordinates.
    Indices is a list of indices indicating which array corresponds to an O atom.
    '''
    sorted_atoms = []
    sorted_indices = []
    for O_idx in indices:
        sorted_indices.append(O_idx)
        tmp__ = findWater(atoms, O_idx)
        sorted_indices.append(tmp__[0])
        sorted_indices.append(tmp__[1])
    for idx in sorted_indices:
        sorted_atoms.append(atoms[idx])
    return np.asarray(sorted_atoms)

def hoh_bisector(roh1, roh2):
    '''
    Computes the bisector vector two oh bonds on the same water.
    '''
    return np.linalg.norm(roh2)*roh1 + np.linalg.norm(roh1)*roh2

def swb_angle(free_oh_vec, hoh_bisector):
    '''
    Computes the angle between the non-donating OH bond vector and
    the HOH angle bisector vector. Returns angle from 0 to pi in radians.
    '''
    return np.arccos(np.dot(free_oh_vec, hoh_bisector)/(np.linalg.norm(free_oh_vec)*np.linalg.norm(hoh_bisector)))

def num_hbonds(hbond_list):
    num = 0
    for x in hbond_list:
        num += len(x)
    return num

header, labels, cluster = read_geoms(ifile)
for iCluster in range(len(cluster)):
    O_indices = []
    for i, label in enumerate(labels[iCluster]):
        if label == 'O' or label == 'o':
            O_indices.append(i)
    try:
        molecules = group_by_molecule(sort_cluster(cluster[iCluster], O_indices))
    except:
        print("Could not group molecules into waters. Exiting.")
        sys.exit(1)
    hbond_list__ = hbond_list(molecules)
    num_t1_bonds = 0
    num_hbonds__ = num_hbonds(hbond_list__)
    for iWater, water in enumerate(hbond_list__):
        #only look at the hbonds where the molecule has a free OH
        if len(water) == 1:
            for hbond in water:
                if len(hbond_list__[hbond[0]]) == 2: #only take acceptors which are double donors
                    swb_angle__ = 0.0
                    if hbond[1] == 1:
                        oh1 = molecules[hbond[0]][0] - molecules[hbond[0]][1]
                        oh2 = molecules[hbond[0]][0] - molecules[hbond[0]][2]
                        swb_angle__ = swb_angle(molecules[iWater][0]-molecules[iWater][2], hoh_bisector(oh1, oh2))
                    elif hbond[1] == 2:
                        oh1 = molecules[hbond[0]][0] - molecules[hbond[0]][1]
                        oh2 = molecules[hbond[0]][0] - molecules[hbond[0]][2]
                        swb_angle__ = swb_angle(molecules[iWater][0]-molecules[iWater][1], hoh_bisector(oh1, oh2))
                    if swb_angle__ > np.pi/2:
                        num_t1_bonds += 1
                #elif swb_angle__ > np.pi/2:
                #    num_t1_bonds += 1
    #if num_t1_bonds + num_c1_bonds != num_hbonds__:
    #    print "Number of swb hbonds does not equal total number of hbonds. Something is wrong. Exiting."
    #    sys.exit(1)
    print( "Cluster ", iCluster, " has: ", num_t1_bonds, " t1d bonds. Total hbonds: ", num_hbonds__)
