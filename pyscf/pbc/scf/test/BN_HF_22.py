from pyscf import lib
from pyscf import pbc as pbc
from pyscf.pbc import gto, scf, df, dft
from pyscf.pbc.tools import madelung
from pyscf.pbc.tools import get_monkhorst_pack_size
from pyscf.pbc.scf.khf import minimum_image
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

    kpts = cell.make_kpts(nk, wrap_around=True)
    return cell, kpts


H2, kpts = build_bn_monolayer_cell(KPT_NUM)

# Energy calculation
kmf = scf.KRHF(H2, kpts)
kmf.exxdiv = None
kmf.kernel()
Madelung = madelung(H2, kpts)
nocc = kmf.cell.tot_electrons() // 2
nk = get_monkhorst_pack_size(kmf.cell, kmf.kpts)
Nk = np.prod(nk)
dm = kmf.make_rdm1()
_, K = kmf.get_jk(cell=kmf.cell, dm_kpts=dm, kpts=kmf.kpts, kpts_band=kmf.kpts)
E_standard = -1. / Nk * np.einsum('kij,kji', dm, K) * 0.5
E_standard /= 2
E_madelung = E_standard - nocc * Madelung
print(E_madelung)

# Saving the wavefunction data (Strange MKL error just feeding mo_coeff...)
mo_coeff_kpts = kmf.mo_coeff_kpts
Lvec_real = kmf.cell.lattice_vectors()
NsCell = kmf.cell.mesh
L_delta = Lvec_real / NsCell[:, None]
dvol = np.abs(np.linalg.det(L_delta))
xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
rptGrid3D = mesh_idx @ L_delta
aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kmf.kpts)

qGrid = minimum_image(kmf.cell, kpts - kpts[0, :])
kGrid = minimum_image(kmf.cell, kpts)

nbands = nocc
nG = np.prod(NsCell)
uKpts = np.zeros((Nk, nbands, nG), dtype=complex)
for k in range(Nk):
    for n in range(nbands):
        utmp = aoval[k] @ np.reshape(mo_coeff_kpts[k][:, n], (-1, 1))
        exp_part = np.exp(-1j * (rptGrid3D @ np.reshape(kGrid[k], (-1, 1))))
        uKpts[k, n, :] = np.squeeze(exp_part * utmp)

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

