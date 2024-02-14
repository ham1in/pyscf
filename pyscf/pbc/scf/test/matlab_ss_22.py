from pyscf import lib
from pyscf import pbc as pbc
from pyscf.pbc import gto, scf, df, dft
from pyscf.pbc.scf.khf import khf_ss_2d_matlab
import scipy.io
import numpy as np
def build_gaussian_cell(nk=(1, 1, 1), kecut=100):
    cell = gto.Cell()
    cell.atom = '''
        H 0.00 0.00 0.00
        H 0.50 0.00 0.00
        
        '''
    cell.a = '''
          1.0   0.0   0.0
          0.0   2.0   0.0
          0.0   0.0   3.0
          '''
    cell.unit = 'B'

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = {'H': 'gth-szv'}
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    cell.dimension = 2
    cell.lowdim_ft_type = 'analytic_2d_1'
    cell.ke_cutoff = kecut
    cell.max_memory = 5000
    cell.build()
    cell.omega = 0
    cell.mesh = np.array([10,20,30])
    kpts = cell.make_kpts(nk, wrap_around=False)
    return cell, kpts


nks = [4,4,1]
gaussian, kpts = build_gaussian_cell(nks)
kmf = scf.KRHF(gaussian, kpts)

import pickle
with open('H2_HF_22_vac24.pkl','rb') as file:
     data = pickle.load(file)


uKpt = scipy.io.loadmat("uKpt.mat")['uKpt']
uKpt = np.swapaxes(uKpt, 0, 2);
e_ex_m = scipy.io.loadmat("Ex_M.mat")['Ex_M']
e_ss = khf_ss_2d_matlab(kmf, nks,uKpt,e_ex_m, N_local = 7,load_from_mat=True)

print("Regular energy")
print(e_ex_m)

print("Singularity Subtraction energy")
print(e_ss)

print("Correction")
print(e_ss-e_ex_m)

