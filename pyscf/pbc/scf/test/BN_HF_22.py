from pyscf import pbc as pbc
from pyscf.pbc import gto, scf, df, dft
from pyscf.pbc.scf.khf import make_ss_inputs
import numpy as np

KPT_NUM = [4, 4, 1]


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
    cell.dimension = 2
    cell.low_dim_ft_type = 'analytic_2d_1'

    kpts = cell.make_kpts(nk, wrap_around=True)
    return cell, kpts

H2, kpts = build_bn_monolayer_cell(KPT_NUM)
kmf = scf.KRHF(cell=H2, kpts=kpts)
kmf.exxdiv = None
kmf.kernel()

dm_kpts =kmf.make_rdm1()
mo_coeff_kpts = kmf.mo_coeff_kpts
E_standard, E_madelung, uKpts = make_ss_inputs(kmf=kmf,kpts=kpts,dm_kpts=dm_kpts,mo_coeff_kpts=mo_coeff_kpts)

##Saving the above data - expedite calculations by doing SCF calculation once for each system.
import pickle
import os

full_path = os.path.realpath(__file__)
filename = os.path.basename(full_path)[:-3]
# filename = "B"  # changeme
filename = "BN_HF_" + str(KPT_NUM[0]) + str(KPT_NUM[1])


data = {
    "e_ex": E_standard,
    "e_ex_m": E_madelung,
    "uKpts": uKpts
}
with open(filename + ".pkl", 'wb') as file:
    pickle.dump(data, file)

