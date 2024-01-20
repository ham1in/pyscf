#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#


import unittest
import tempfile
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import khf
from pyscf.pbc import df
from pyscf import lib
import os

cwd = os.getcwd()
nthreads = 16
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)

lib.num_threads(os.getenv('OMP_NUM_THREADS'))


def nk_output_str(nk):
    return '-nk' + str(nk[0]) + str(nk[1]) + str(nk[2])


def kecut_output_str(kecut):
    return '-kecut' + str(kecut)

L = 4
cell = pbcgto.Cell()
kmesh = [2,2, 1]
A = np.eye(3) * 4,

A = np.array([[4,0,0],
              [0,4,0],
              [0,0,8]])

cell.build(unit='B',
           a=A,
           mesh=[11, 11, 21],
           atom='''He 2 0 0; He 3 0 0''',
           dimension=3, # for mean method, we're not using truncated coulomb for now
           low_dim_ft_type='inf_vacuum',
           verbose=5,
           rcut=7.427535697575829,
            basis =  'gth-szv',
            # output = cwd+'/He-pyscf-stagger' + nk_output_str(kpts) + '.out'
           )
kpts = cell.make_kpts(kmesh)

# Compute Mean Method Exchange

print('testing mean method')
mf = khf.KRHF(cell,exxdiv='ewald')
mf.with_df = df.FFTDF(cell,kpts).build()

mf.kpts = cell.make_kpts(kmesh)
Nk = np.prod(kmesh)
mf.exxdiv = 'ewald'
e1 = mf.kernel()
dm_un = mf.make_rdm1()

#Regular exchange
_, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un, kpts = mf.kpts, kpts_band = mf.kpts, with_j = False)
Ek = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
Ek /=2
print('Ek (a.u.) is ')
print(Ek)

# Mean method
mf.exxdiv = 'mean'
cell.dimension = 2
_, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un, kpts = mf.kpts, kpts_band = mf.kpts, with_j = False)
Ek_mean = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
Ek_mean /=2

Ek_mean = Ek_mean.real
print('Ek_mean (a.u.) is ')
print(Ek_mean)

# Compute Regular Exact Exchange for 2D System
# cell.dimension = 3
# mf = khf.KRHF(cell)
# mf.with_df = df.FFTDF(cell)
# mf.kpts = cell.make_kpts(kmesh)
# Nk = np.prod(kmesh)
# e1 = mf.kernel()
# dm_un = mf.make_rdm1()
# Jo, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un, kpts = mf.kpts, kpts_band = mf.kpts)
# Ek = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
# Ek /=2
#
# Ek = Ek.real
# print('Ek_regular (a.u.) is ')
#
# print(Ek)
# np.testing.assert_almost_equal(Ek, -2.2510515644, 4)
# np.testing.assert_almost_equal(Ek, Ek_standard, 4)
# f.write("Computed Ek: %.10E\n" % (ek.real))


# e1 = mf.kernel()
# self.assertAlmostEqual(e1, -3.53769771, 4)