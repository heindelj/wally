#!/usr/bin/env python

import numpy as np
from read_geometries import read_geoms
from hbond_definition import r_psi_hbond
from group_by_molecule import group_by_molecule
import sys
import string

try:
    ifile, nwat = sys.argv[1], sys.argv[2]
except:
    print ("I did not receive an input file. Should be xyz formatted water cluster geometry. [xyz] [nwat]")
    sys.exit(1)

def adjacency_matrix(molecules):
    '''
    Determines if all N(N-1)/2 pairs of molecules is hydrogen bonded.
    '''
    adj_list = []
    adj_m = np.zeros((len(molecules), len(molecules)))
    for i in range(len(molecules)):
        for j in range(len(molecules)):
    
    #for i in range(len(molecules) - 1):
    #    for j in range(i + 1, len(molecules)):
    #        if r_psi_hbond(molecules[i], molecules[j]) != -1:
    #            adj_list.append(1)
    #        else:
    #            adj_list.append(0)
            
            if i == j:
                adj_m[i,j] = 0
            elif r_psi_hbond(molecules[i], molecules[j]) != -1:
                #this check because some structures hydrogen bond with themselves and break this
                if adj_m[i,j] == 0:
                    adj_m[i,j] += 1
                    adj_m[j,i] += 1
            
    #use this for column ordered list
    ###uncomment if you don't want 5-coordinated clusters###
    #for row in adj_m:
    #    if np.sum(row) >= 5:
    #        print "This has a 5-coordinated water. Can't handle that yet."
    #        idx_5 = []
    #        for i, idx in enumerate(row):
    #            if idx == 1:
    #                idx_5.append(i)
    #        print "The coordinated oxygens are: ", idx_5
    #        #sys.exit(1)
    #        return -1
    k = 1
    i = 1
    while k < len(molecules):
        adj_list.extend(map(int, adj_m[0:k,i]))
        k += 1
        i += 1
    return adj_list

def bit_string(adj_list):
    '''
    Takes the adjacency list and returns a list with each element a bit string.
    Appends zeros so that the total bit string length is divisible by six.
    '''
    bit_string = []
    div = len(adj_list) % 6
    for i in range(div):
        adj_list.append(0)
    for i in range(int(len(adj_list)/6)):
        bit_string.append(''.join(map(str, adj_list[i*6:i*6+6])))
    return bit_string

def graph6_format(n, bit_string):
    '''
    Returns the graph6 format of a graph as a string.
    '''
    if n <= 62:
        n += 63
    else:
        print ("I need n <= 62 right now.")
        sys.exit(1)
    bit_list = []
    for elem in bit_string:
        bit_list.append(int(elem, 2) + 63)
    R = ' '.join(map(str, bit_list))
    g6string = str(n) + ' ' + R
    g6ascii = []
    for iascii in g6string.split():
        g6ascii.append(chr(int(iascii)))
    g6ascii = ''.join(map(str, g6ascii))
    return g6ascii

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

header, labels, cluster = read_geoms(ifile)
for iCluster in range(len(cluster)):
    O_indices = []
    for i, label in enumerate(labels[iCluster]):
        if label == 'O' or label == 'o':
            O_indices.append(i)
    try:
        molecules = group_by_molecule(sort_cluster(cluster[iCluster], O_indices))
    except:
        print ("Invalid Structure.")
        continue

    adj_list = adjacency_matrix(molecules)
    if adj_list != -1:
        print (graph6_format(int(nwat), bit_string(adj_list)))
    else:
        print ("Failed on this structure.")
