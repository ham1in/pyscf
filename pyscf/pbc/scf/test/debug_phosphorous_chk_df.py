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
nthreads = 8
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)

lib.num_threads(os.getenv('OMP_NUM_THREADS'))


def nk_output_str(nk):
    return '-nk' + str(nk[0]) + str(nk[1]) + str(nk[2])


def kecut_output_str(kecut):
    return '-kecut' + str(kecut)


def build_phosphorous_cell(nk = (1,1,1),kecut=100,with_gamma_point=True,wrap_around=True):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
P   0.0000000   12.3073329  7.8236391
P   3.1137830   18.6749386  0.7625871
P   0.0000000   18.6749386  3.5305260
P   3.1137830   12.3073329  5.0557003
P   3.1137830   1.9799090   7.8236391
P   0.0000000   8.3475148   0.7625871
P   3.1137830   8.3475148   3.5305260
P   0.0000000   1.9799090   5.0557003
        '''

              
    cell.a = '''
6.227566008270  0.000000000000  0.000000000000
0.000000000000  20.654847635604 0.000000000000
0.000000000000  0.000000000000  8.586226257151
        '''

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-12
    #cell.ke_cutoff = 55.13
    cell.ke_cutoff = kecut
    cell.max_memory = 120000
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
    return cell, kpts

wrap_around = True
nkx = 4
kmesh = [nkx, nkx, nkx]
vac_dim = 3.0 # bohr
cell, kpts= build_phosphorous_cell(nk=kmesh,kecut=56,wrap_around=wrap_around)
cell.dimension = 3
cell.build()

print('Kmesh:', kmesh)

mf = khf.KRHF(cell, exxdiv='ewald')
from pyscf import scf as mol_scf
# mf.chkfile = 'phosphorous-kmf-nk444.chk'

# df_type = df.GDF
# df = df_type(cell, kpts)
# df._cderi_to_save = 'phosphorous-df-nk444.h5'
# import time
# df_build_time = time.time()
# mf.with_df = df.build()
# df_build_time = time.time() - df_build_time
# print('Time to build df:', df_build_time)



# Load GDF's CDERIs
df_dir = '/home/sjquiton/pyscf-files/singularity_subtraction/'
df_file = df_dir+'phosphorous-df-nk444.h5'
df_type = df.GDF
df = df_type(cell, kpts)
df._cderi = df_file
df._cderi_to_save = None
mf.with_df = df.build()

mf = mol_scf.addons.remove_linear_dep_(mf).run()


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

# div_vector = [2,2]
from pyscf.pbc.scf.khf import compute_SqG_anisotropy
