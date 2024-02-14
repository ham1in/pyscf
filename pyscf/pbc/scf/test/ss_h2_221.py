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

def build_h2_cell(nk=(1, 1, 1), kecut=200):
    cell = pbc.gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = '''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0

    cell.basis = {'H': 'gth-szv'}
    cell.pseudo = 'gth-pbe'

    cell.ke_cutoff = kecut
    cell.max_memory = 1000
    cell.precision = 1e-8
    cell.build()
    cell.omega = 0
    kpts = cell.make_kpts(nk, wrap_around=False)
    return cell, kpts

cell, kpts = build_h2_cell(nk=(2,2,2))

kmf = pbc.scf.KHF(cell)
ss, standard, s_Madelung, timescf, timess = khf_2d(diamond,kpts)

def khf_2d(kmf, nks, uKpts, made, dm_kpts = None, N_local = 5):