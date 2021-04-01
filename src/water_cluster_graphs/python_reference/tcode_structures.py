#!/usr/bin/env python

import numpy as np
from read_geometries import read_geoms
from hbond_definition import r_psi_hbond
from group_by_molecule import group_by_molecule
import sys
import numpy as np

sys.path.append('/global/homes/h/heindelj/bin/TTM21F_minimization_python/')
from TTM21F_minimize import *

try:
    tcodefile, ref_geom = sys.argv[1], sys.argv[2]
    outfile = "tcode_guess_structures.xyz"
except IndexError:
    print ("[tcode file] [ref structure xyz file]")
    sys.exit(1)

if len(sys.argv) == 4:
    outfile = sys.argv[3]

def chunks(l):
    it = iter(l)
    return list(zip(it, it))

def findWater(geom, Num):
    sortedIndices = []
    for i, line in enumerate(geom):
        dist = np.linalg.norm(geom[Num] - line)
        if np.any(line): 
            if dist > 0.01 and dist < 1.20: #assuming angstroms
                sortedIndices.append(i)
    if len(sortedIndices) == 0:
        sortedIndices.append(-1)
    if len(sortedIndices) == 1:
        sortedIndices.append(-1)
    return sortedIndices

def sort_cluster(atoms, indices):
    '''
    Takes a list of numpy arrays with each array having xyz coordinates.
    Indices is a list of indices indicating which array corresponds to an O atom.
    '''
    sorted_atoms = []
    sorted_indices = []
    for O_idx in indices:
        sorted_indices.append(O_idx)
        tmp__ = findWater(atoms, O_idx)
        for idx in tmp__:
            sorted_indices.append(idx)
    for idx in sorted_indices:
        if idx >= 0:
            sorted_atoms.append(atoms[idx])
        else:
            sorted_atoms.append(np.zeros(3))
    return np.asarray(sorted_atoms)

def angle(r1, r2):
    '''
    Compute angle between two vectors, r1 and r2
    '''
    return np.arccos(np.clip(np.dot(r1, r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)), -1.0, 1.0))

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def dangling_hydrogen(v, beta):
    '''
    Takes an OH vector, v, and generates a random vector at an angle beta
    from v. beta is given in degrees
    '''
    beta = beta*np.pi/180
    v = v/np.linalg.norm(v)
    a = np.random.randn(3)
    cross_vector = np.cross(v, a)/(np.linalg.norm(v)*np.linalg.norm(a))
    s = np.random.uniform()
    r = np.random.uniform()
    h = np.cos(beta)
    phi = 2*np.pi*s
    z = h + (1-h)*r
    T = np.sqrt(1 - z**2)
    x = np.cos(phi)*T
    y = np.sin(phi)*T
    w = np.array(a*x + cross_vector*y + v*z)
    return w

def find_COM(atoms, O_indices):
    '''
    Takes an atoms array and which of these are oxygens and returns the vector pointing
    to the center of mass of the cluster.
    '''
    vCOM = np.zeros(3)
    for i_oxy in O_indices:
        vCOM += np.asarray(atoms[i_oxy])
    return vCOM/len(O_indices)

def dangling_hydrogen_from_com(vO, vCOM):
    '''
    Takes the O atom position as input.
    Places a hydrogen at a distance of .96 angstroms from a hydrogen along the line to the center
    of mass (by oxygens only) of the cluster. All free OH's are made to point outwards.
    '''
    line = vO - vCOM
    line = line*(np.linalg.norm(line) - 0.96)/np.linalg.norm(line)
    return line + vCOM
    
def structure_from_tcode(tcode, atoms, O_indices):
    '''
    Takes a tcode, an atoms array, and which of those are oxygen atoms
    then returns a completed structure.
    '''
    nat = len(atoms)
    structure = np.zeros((len(atoms), 3))
    #get the oxygen atoms
    for i, Oatom in enumerate(O_indices):
        structure[i,:] = atoms[Oatom]
    tcode_strings = tcode.split()
    del tcode_strings[0:2]
    pairs = chunks(map(int, tcode_strings))
    #place the hydrogen-bonded H atoms
    for i, pair in enumerate(pairs):
        vec = structure[pair[1],:] - structure[pair[0],:]
        t = 0.9572/np.linalg.norm(vec)
        h_location = structure[pair[0],:] + t*vec
        structure[len(O_indices) + i,:] = h_location
    #place the dangling hydrogens
    structure = sort_cluster(structure, range(int(nat/3)))
    #make sure the angles of all double-donating structures are properly 104.5
    for i, vec in enumerate(structure):
        if (i - 2) % 3 == 0 and np.any(vec) == True: #only take the second oh bonds and check they're nonzero
            oh1 = structure[i-1] - structure[i-2]
            oh2 = structure[i] - structure[i-2]
            theta = angle(oh1, oh2)
            axis = np.cross(oh1, oh2)
            axis = axis/np.linalg.norm(axis)
            structure[i] = np.dot(rotation_matrix(-axis, (theta - 104.52/180.0*np.pi)/2), oh2) + structure[i-2]
            structure[i-1] = np.dot(rotation_matrix(axis, (theta - 104.52/180.0*np.pi)/2), oh1) + structure[i-2]
    for i, vec in enumerate(structure):
        if not np.any(vec):
            #this allows placing hydrogen for oxygens which are double acceptors as well
            Oindex = i - 1
            if Oindex not in O_indices:
                Oindex -= 1
            w = dangling_hydrogen_from_com(structure[Oindex], find_COM(atoms, O_indices))
            structure[i] = w + structure[Oindex]
            t = 0.9572/np.linalg.norm(w - structure[Oindex])
            structure[i] = structure[Oindex] - t*(w - structure[Oindex])
            #now let's be sure the dangling OH distance is actually 0.96 angstroms
            dist = structure[i] - structure[Oindex]
            t = 0.9572/np.linalg.norm(structure[i] - structure[Oindex])
            structure[i] = structure[Oindex] + t*dist
    #now make the HOH angle 104.5
    for i in O_indices:
        oh1 = structure[i+1] - structure[i]
        oh2 = structure[i+2] - structure[i]
        # check that H atoms aren't overlapping
        if np.linalg.norm(oh1 - oh2) >= 0.000001:
            theta = angle(oh1, oh2)
            axis = np.cross(oh1, oh2)
            axis = axis/np.linalg.norm(axis)
            structure[i+2] = np.dot(rotation_matrix(-axis, theta - 104.52/180.0*np.pi), oh2) + structure[i]
            oh2 = structure[i]-structure[i+2]
        else:
            #get the atoms off of each other
            oh1 += 0.01
            oh2 -= 0.01
            theta = angle(oh1, oh2)
            axis = np.cross(oh1, oh2)
            axis = axis/np.linalg.norm(axis)
            structure[i+1] = np.dot(rotation_matrix(-axis, theta + 52.26/180.0*np.pi), oh1) + structure[i]
            structure[i+2] = np.dot(rotation_matrix(-axis, theta - 52.26/180.0*np.pi), oh2) + structure[i]
            dist1 = structure[i+1] - structure[i]
            dist2 = structure[i+2] - structure[i]
            t1 = 0.9572/np.linalg.norm(structure[i+1] - structure[i])
            t2 = 0.9572/np.linalg.norm(structure[i+2] - structure[i])
            structure[i+1] = structure[i] + t1*dist1
            structure[i+2] = structure[i] + t2*dist2
        #this catches the times that we accidentally rotate the wrong way
        #due to the ambiguity in the sign we get back from the cross product
        if angle(oh1, oh2) - 104.52 >= 1.0**-6:
           theta = angle(oh1, oh2)
           structure[i+2] = np.dot(rotation_matrix(axis, theta - 104.52/180.0*np.pi), oh2) + structure[i]
    return structure

header, labels, atoms = read_geoms(ref_geom)
O_indices = []
for i, label in enumerate(labels[0]):
    if label == 'O' or label == 'o':
        O_indices.append(i)

with open(tcodefile, 'r') as tc:
    for tcode in tc:
        structure = structure_from_tcode(tcode, atoms[0], O_indices)
        N=len(structure)
        structure = [str(structure[x]) for x in range(len(structure))]
        for i, coords in enumerate(structure):
            if i % 3 == 0:
                structure[i] = "O " + structure[i]
            else:
                structure[i] = "H " + structure[i]
        with open(outfile, 'a') as fout:
            fout.write(str(N) + "\n")
            fout.write("\n")
            for line in structure:
                printable = line.replace("[","").replace("]","")
                fout.write(printable)
                fout.write("\n")
