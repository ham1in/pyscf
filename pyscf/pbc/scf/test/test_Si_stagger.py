from pyscf import lib
from pyscf import pbc as pbc
from pyscf.pbc import gto, scf, df, dft
from pyscf.pbc.tools import madelung
from pyscf.pbc.tools import get_monkhorst_pack_size
from pyscf.pbc.scf.khf import minimum_image
from pyscf.pbc.scf.khf import khf_stagger
import numpy as np
import os

lib.num_threads(os.getenv('OMP_NUM_THREADS'))

KPT_NUM = [2, 2, 2]

def build_cell(nk = (1,1,1),kecut=100):
    cell = pbc.gto.Cell()
    cell.a = '''
    3.3336, 0.0000, 1.9246
    1.1112, 3.1429, 1.9246
    0.0000, 0.0000, 3.8493
    '''
    cell.atom = '''
    Si    3.889168, 2.750058, 6.736237
    Si    0.555596, 0.392865, 0.962319
    '''
    cell.unit = 'angstrom'

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = {'Si':'gth-szv'}
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    cell.ke_cutoff = kecut
    cell.max_memory = 5000
    cell.build()
    cell.omega = 0
    kpts = cell.make_kpts(nk, wrap_around=False)    
    return cell, kpts
    
cell, kpts = build_cell(KPT_NUM)

#Energy calculation
stagger_ex_m ,stagger_e, e_ex= khf_stagger(cell,kpts)

print("Stagger Energy")
print(stagger_ex_m)