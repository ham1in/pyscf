import os
nthreads = 48
cwd = os.getcwd()
os.environ['OWP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)

from pyscf import lib
lib.num_threads(nthreads)

import numpy as np
from pyscf.pbc import gto,scf
import pyscf.pbc as pbc
from pyscf.pbc.scf.khf import khf_ss

def build_diamond_cell(nk = (1,1,1),kecut=100):
    cell = pbc.gto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
         C 0.0 0.0 0.0
         C 1.68516327271508 1.68516327271508 1.68516327271508
        '''
    cell.a = '''
         0.0 3.370326545430162 3.370326545430162
         3.370326545430162 0.0 3.370326545430162
         3.370326545430162 3.370326545430162 0.0  
        '''
    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = {'C':'gth-szv'}
    cell.precision = 1e-8
    cell.pseudo = 'gth-pbe'
    cell.ke_cutoff = kecut
    cell.max_memory = 1000

    cell.build()
    cell.omega = 0
    kpts = cell.make_kpts(nk, wrap_around=False)    
    return cell, kpts

diamond, kpts = build_diamond_cell(nk=(3,3,3))
ss, standard, s_Madelung, timescf, timess = khf_ss(diamond,kpts)

