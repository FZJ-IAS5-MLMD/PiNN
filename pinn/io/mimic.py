import numpy as np
import linecache
from os import path
from ase.data import atomic_numbers
import mimicpy
from mimicpy.parsers.mpt import MPT
from pinn.io import list_loader
from pinn.io.trr import check_trr, read_trr, get_trr_frames

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
