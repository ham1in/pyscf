from pyscf import lib
from pyscf import pbc as pbc
from pyscf.pbc import gto, scf, df, dft
from pyscf.pbc.scf.khf import khf_2d

def build_H2_cell(nk = (1,1,1),kecut=100):
    cell = pbc.gto.Cell()
    cell.atom='''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   24.0
        '''
    cell.unit = 'B'

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = {'H':'gth-szv'}
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    cell.dimension = 2
    cell.lowdim_ft_type = 'analytic_2d_1'
    cell.ke_cutoff = kecut
    cell.max_memory = 5000
    cell.build()
    cell.omega = 0
    kpts = cell.make_kpts(nk, wrap_around=False)
    return cell, kpts

nks = [3,3,1]
H2, kpts = build_H2_cell(nks)
kmf = scf.KRHF(H2, kpts)

import pickle
with open('H2_HF_33_vac24.pkl','rb') as file:
     data = pickle.load(file)

e_ss = khf_2d(kmf, nks,data["uKpts"],data["e_ex_m"], N_local = 3)

print("Regular energy")
print(data["e_ex_m"])

print("Singularity Subtraction energy")
print(e_ss)



