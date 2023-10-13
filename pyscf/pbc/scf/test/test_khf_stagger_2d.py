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
kpts = [2,1, 1]
cell.build(unit='B',
           a=np.eye(3) * 4,
           mesh=[25, 25, 40],
           atom='''He 2 0 0; He 3 0 0''',
           dimension=2,
           low_dim_ft_type='analytic_2d_1',
           verbose=0,
           rcut=7.427535697575829,
           # basis={'He': [[0, (0.8, 1.0)],
           #               # [0, (1.0, 1.0)],
           #               [0, (1.2, 1.0)]
           #               ]}),
            basis = {'He': 'gth-szv'},
            output = cwd+'/He-pyscf-stagger' + nk_output_str(kpts) + '.out'
           )
kpts = cell.make_kpts(kpts)
E_stagger_M, E_stagger = khf.khf_stagger(icell=cell,ikpts=kpts,version = "Non_SCF")
print('E_stagger_M, E_stagger is')
print(E_stagger_M,E_stagger)

f = open(cell.output, "a")
f.write("E_stagger_M: %.10E\n" % (E_stagger_M))
f.write("E_stagger: %.10E\n" % (E_stagger))


# mf.with_df = df.AFTDF(cell)
# mf.with_df.eta = 0.2
# mf.with_df.mesh = cell.mesh

# e1 = mf.kernel()
# self.assertAlmostEqual(e1, -3.53769771, 4)