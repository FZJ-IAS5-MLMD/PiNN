import numpy as np
import linecache
from os import path
import mimicpy
from mimicpy.parsers.mpt import MPT
from pinn.io import list_loader
from pinn.io.trr import check_trr, read_trr, get_trr_frames

# from ase
atomic_numbers = {'X': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,\
 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,\
  'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,\
   'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,\
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, \
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, \
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,\
     'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}


def _read_energy(ener_file, frame):
    # load full file into cache once for fast access of lines
    if not path.exists(ener_file):
        raise FileNotFoundError(f"{ener_file} not found")
    return float(linecache.getline(ener_file, frame+1).split()[3])

@list_loader(pbc=True, force=True)
def _frame_loader(frame):
    mpt_file, trr_file, ener_file, i = frame
    
    mpt = MPT.fromFile(mpt_file)
    elems = [atomic_numbers[i] for i in mpt['element']]
    
    # open the ith trajectory of the trr file
    ts = read_trr(trr_file, i)
    
    if len(elems) != ts['natoms']: # check if elem list and atoms in trr match
        raise ValueError("Number of atoms in TRR and MPT not equal! In TRR: "+str(len(trr.atoms))+", in MPT:"+str(len(elems)))
    
    coords = ts['x']
    forces = ts['f']
    cell = ts['box']
    
    energy = _read_energy(ener_file, i)
        
    data = {'coord': coords, 'cell': cell, 'elems': elems,
            'e_data': energy, 'f_data': forces}
    return data

def _load_single_set(mpt_file, trr_file, ener_file):
    n = get_trr_frames(trr_file)
    
    frame_list = [(mpt_file, trr_file, ener_file, i) for i in range(n)]
    
    return frame_list

def load_mimic(mpt_file, trr_file, ener_file, **kwargs):
    mimicpy.setLogger(0) # stop any messages
    
    if isinstance(trr_file, str):
        flist = [(mpt_file, trr_file, ener_file)]
    else:
        flist = zip(mpt_file, trr_file, ener_file)
    
    frame_list = []
    
    for fname in flist:
        mpt_file, trr_file, ener_file = fname
        frame_list += _load_single_set(mpt_file, trr_file, ener_file)
        
    return _frame_loader(frame_list, **kwargs)
