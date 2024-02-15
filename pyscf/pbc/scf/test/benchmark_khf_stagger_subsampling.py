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
# Authors: Stephen Quiton <stephen.quiton@berkeley.edu>



import unittest
import tempfile
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import khf
from pyscf.pbc.scf.subsample_kpts import subsample_kpts
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


def build_bn_hex_cell(nk=(1, 1, 1), kecut=100):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom = '''
        B   2.36527819806   1.36559400436   1.96955217648
        B   2.36527819806   -1.36559400436  5.90865652944
        N   2.36527819806   1.36559400436   5.90865652944
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

def build_bn_monolayer_cell(nk=(1, 1, 1), kecut=100):
    cell = pbcgto.Cell()
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
nkx = 4
kmesh = [nkx, nkx, 1]
cell, kpts= build_bn_monolayer_cell(nk=kmesh,kecut=56)

cell.lowdim_ft_type = 'analytic_2d_1'
cell.dimension = 2

cell.build()

print('Kmesh:', kmesh)

mf = khf.KRHF(cell, exxdiv='ewald')
df_type = df.FFTDF
mf.with_df = df_type(cell, kpts).build()

Nk = np.prod(kmesh)
mf.exxdiv = 'ewald'
e1 = mf.kernel()
dm = mf.make_rdm1()

# Regular energy components

h1e = mf.get_hcore()
ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm).real

Jo, Ko = mf.get_jk(cell=mf.cell, dm_kpts=dm, kpts=mf.kpts, kpts_band=mf.kpts, with_j=True)



Ek = -1. / Nk * np.einsum('kij,kji', Ko, dm) * 0.5
Ej = 1. / Nk * np.einsum('kij,kji', Jo, dm)

Ek /= 2.
Ek = Ek.real
Ej /= 2.
Ej = Ej.real

print('Ek (a.u.) is ', Ek)
print('Ej (a.u.) is ', Ej)
print('Ehcore (a.u.) is ', ehcore)
print('Enuc (a.u.) is ', mf.energy_nuc().real)
print('Ecoul (a.u.) is ', Ek + Ej)

# Subsample 8 kpts


div_vector = [2,2]

nk_list, nks_list, Ej_list, Ek_list = subsample_kpts(mf=mf,dim=2,div_vector=div_vector, df_type=df_type)

print('=== Kpoint Subsampling Results === ')

print('\nnk list')
print(nk_list)
print('\nnks list')
print(nks_list)
print('\nEk list')
print(Ek_list)




nk_list, nks_list, Ej_list, Ek_list = subsample_kpts(mf=mf,dim=2,div_vector=div_vector,stagger_type="Non_SCF",df_type=df_type)


print('=== Kpoint Subsampling Results (with Stagger) === ')

print('\nnk list')
print(nk_list)
print('\nnks list')
print(nks_list)
print('\nEk list')
print(Ek_list)


