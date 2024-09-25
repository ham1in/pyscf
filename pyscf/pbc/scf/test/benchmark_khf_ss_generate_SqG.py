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
nthreads = 24
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)

lib.num_threads(os.getenv('OMP_NUM_THREADS'))


def nk_output_str(nk):
    return '-nk' + str(nk[0]) + str(nk[1]) + str(nk[2])


def kecut_output_str(kecut):
    return '-kecut' + str(kecut)

def build_diamond_cell(nk = (1,1,1),kecut=100,wrap_around=True):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
         C 0.0 0.0 0.0
         C 1.68516327271508 1.68516327271508 1.68516327271508
        '''
    cell.a = '''
         0.0 3.370326545430162 3.370326545430162
         3.370326545430162 0.0 3.370326545430162
         3.370326545430162 3.370326545430162 0.0  
        '''
    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = {'C':'gth-szv'}
    cell.precision = 1e-8
    cell.pseudo = 'gth-pbe'
    cell.ke_cutoff = kecut
    cell.max_memory = 1000

    cell.build()
    cell.omega = 0
    kpts = cell.make_kpts(nk, wrap_around=wrap_around)    
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
    cell.dimension = 3

    kpts = cell.make_kpts(nk, wrap_around=True)
    return cell, kpts

def build_H2_cell(nk = (1,1,1),kecut=100,wrap_around=False):
    cell = pbcgto.Cell()
    cell.atom='''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    # cell.atom='''
    #     H 1.50   1.50   2.10
    #     H 1.50   1.50   3.90
    #     '''
    # cell.a = '''
    #     3.0   0.0   0.0
    #     0.0   3.0   0.0
    #     0.0   0.0   24.0
    #     '''
    cell.unit = 'B'

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = {'H':'gth-szv'}
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    cell.dimension = 3
    cell.ke_cutoff = kecut
    cell.max_memory = 5000
    cell.build()
    cell.omega = 0
    kpts = cell.make_kpts(nk, wrap_around=wrap_around)
    return cell, kpts

def build_Si_cell(nk = (1,1,1),kecut=100,with_gamma_point=True,wrap_around=True):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
Si  0.00000000000   0.00000000000   0.00000000000
Si  2.57177646209   2.57177646209   2.57177646209
        '''

              
    cell.a = '''
5.14355292417   5.14355292417   0.00000000000
5.14355292417   0.00000000000   5.14355292417
0.00000000000   5.14355292417   5.14355292417
        '''

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    #cell.ke_cutoff = 55.13
    cell.ke_cutoff = kecut
    cell.max_memory = 240000
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
    return cell, kpts
wrap_around = True
nkx = 4
kmesh = [nkx, nkx, nkx]
cell, kpts= build_Si_cell(nk=kmesh,kecut=100,wrap_around=wrap_around)
cell.dimension = 3

cell.build()

print('Kmesh:', kmesh)

mf = khf.KRHF(cell, exxdiv='ewald')
df_type = df.GDF
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

# div_vector = [2,2]
from pyscf.pbc.scf.khf import build_SqG, minimum_image,make_ss_inputs

mf.exxdiv = None # so that standard energy is computed without madelung

# Store output from make_ss_inputs in a numpy file
results = {
    'mo_coeff_kpts': np.array(mf.mo_coeff_kpts),
    'dm_kpts': np.array(dm),
}


# Load all info
cell = mf.cell
kpts = mf.kpts
nks = np.array(kmesh)
nocc = cell.tot_electrons() // 2
nkpts = np.prod(nks)

Lvec_real = mf.cell.lattice_vectors()
NsCell = mf.cell.mesh
L_delta = Lvec_real / NsCell[:, None]
dvol = np.abs(np.linalg.det(L_delta))
xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
rptGrid3D = mesh_idx @ L_delta
qGrid = minimum_image(cell, kpts - kpts[0, :])
kGrid = minimum_image(cell, kpts)

Lvec_recip = cell.reciprocal_vectors()
Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip
 
#   Step 1.3: evaluate MO periodic component on a real fine mesh in unit cell
nbands = nocc
nG = np.prod(NsCell)

E_standard, E_madelung, uKpts, qGrid, kGrid = make_ss_inputs(kmf=mf, kpts=kpts, dm_kpts=dm,mo_coeff_kpts=mf.mo_coeff_kpts)
debug_options = {'filetype':['pkl','mat']}
SqG = build_SqG(nkpts, nG,nbands, kGrid, qGrid, mf, uKpts, rptGrid3D, dvol, NsCell, GptGrid3D, nks=nks, debug_options=debug_options)




# results["M"] = M
# import pickle
# with open('diamond_444.pkl', 'wb') as f:
#     pickle.dump(results, f)
