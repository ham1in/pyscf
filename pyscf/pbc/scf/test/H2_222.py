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

h2, kpts = build_h2_cell(nk=(2,2,2))
#mf = scf.KHF(h2, kpts)
#print()
#print(mf.kernel())
ss = khf_ss(h2,kpts)

