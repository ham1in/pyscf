import unittest
import tempfile
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import khf
from pyscf.pbc import df

import os
cwd = os.getcwd()
nthreads = 8
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)


def nk_output_str(nk):
    return '-nk' + str(nk[0]) + str(nk[1]) + str(nk[2])


def kecut_output_str(kecut):
    return '-kecut' + str(kecut)

L = 4
cell = pbcgto.Cell()
kmesh = [2,2, 1]
cell.build(unit='B',
           a=np.eye(3) * 4,
           mesh=[25, 25, 40],
           atom='''He 2 0 0; He 3 0 0''',
           dimension=2,
           low_dim_ft_type='analytic_2d_1',
           verbose=5,
           rcut=7.427535697575829,
            basis =  'gth-szv',
            # output = cwd+'/He-pyscf-stagger' + nk_output_str(kpts) + '.out'
           )
kpts = cell.make_kpts(kmesh)
# Compute Staggered Mesh Exact Exchange for 2D System
# mf.with_df = df.FFTDF(cell)
Ek_stagger_M, Ek_stagger, Ek_standard = khf.khf_stagger(icell=cell,ikpts=kpts,version = "Non_SCF",df_type=df.FFTDF)
print('Ek_stagger_M, Ek_stagger (a.u.) is')
print(Ek_stagger_M,Ek_stagger)
np.testing.assert_almost_equal(Ek_stagger_M, -2.2766118557347053, 4)

# f = open(cell.output, "a")
# f.write("E_stagger_M: %.10E\n" % (E_stagger_M))
# f.write("E_stagger: %.10E\n" % (E_stagger))

# Compute Regular Exact Exchange for 2D System
mf = khf.KRHF(cell)
mf.with_df = df.FFTDF(cell)
mf.kpts = cell.make_kpts(kmesh)
Nk = np.prod(kmesh)
e1 = mf.kernel()
dm_un = mf.make_rdm1()
Jo, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un, kpts = mf.kpts, kpts_band = mf.kpts)
Ek = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
Ek /=2

Ek = Ek.real
print('Ek_regular (a.u.) is ')

print(Ek)
np.testing.assert_almost_equal(Ek, -2.2845080096640933, 4)
np.testing.assert_almost_equal(Ek, Ek_standard, 4)
# f.write("Computed Ek: %.10E\n" % (ek.real))


# e1 = mf.kernel()
# self.assertAlmostEqual(e1, -3.53769771, 4)