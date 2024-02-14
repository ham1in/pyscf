from pyscf import lib
from pyscf import pbc as pbc
from pyscf.pbc import gto, scf, df, dft
from pyscf.pbc.scf.khf import khf_2d

def build_bn_monolayer_cell(nk=(1, 1, 1), kecut=100):
    cell = pbc.gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = '''
        B   2.36527819806   1.36559400436   1.96955217648
        N   2.36527819806   -1.36559400436  1.96955217648

        '''
    cell.a = '''
        2.37390045859   -4.11171620638  0.00000000000
        2.37390045859   4.11171620638   0.00000000000
        0.00000000000   0.00000000000   14.56461897153

        '''
    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'

    cell.ke_cutoff = kecut
    cell.max_memory = 1000
    cell.precision = 1e-8

    kpts = cell.make_kpts(nk, wrap_around=True)
    return cell, kpts

nks = [2,2,1]
H2, kpts = build_bn_monolayer_cell(nks)
kmf = scf.KRHF(H2, kpts)

import pickle
filename = 'BN_HF_' + str(nks[0]) + str(nks[1])  +'.pkl'
with open(filename,'rb') as file:
     data = pickle.load(file)

e_ss = khf_2d(kmf, nks,data["uKpts"],data["e_ex_m"], N_local = 9)

print("Regular energy")
print(data["e_ex_m"])

print("Singularity Subtraction energy")
print(e_ss)



