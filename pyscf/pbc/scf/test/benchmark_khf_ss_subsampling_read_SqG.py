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

def build_h2_cell(nk = (1,1,1),kecut=100,vac_dim=6.0,wrap_around=True):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
        H 0.00 0.00 0.00
        H 0.00 0.00 1.80
        '''
    cell.a = np.eye(3)*vac_dim

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0

    
    
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    
    cell.ke_cutoff = kecut
    cell.max_memory = 1000
    cell.precision = 1e-8
    #for i in range(len(cell.atom)):
    #   cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
    cell.build()
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
0.00000000000   5.14355292417   5.14355292417
5.14355292417   0.00000000000   5.14355292417
5.14355292417   5.14355292417   0.00000000000
        '''

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    #cell.ke_cutoff = 55.13
    cell.ke_cutoff = kecut
    cell.max_memory = 240000
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
    return cell, kpts

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
    cell.precision = 1e-8
    #cell.ke_cutoff = 55.13
    cell.ke_cutoff = kecut
    cell.max_memory = 100
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
    return cell, kpts

nkx = 2
kmesh = [nkx, nkx, nkx]
wrap_around=True
cell, kpts= build_phosphorous_cell(nk=kmesh,kecut=56)
cell.dimension = 3
cell.build()
print('Kmesh:', kmesh)




# e1 = mf.kernel()
# dm = mf.make_rdm1()
# mo_coeff = np.array(mf.mo_coeff_kpts)


# # Read dm and mo_coeff from pkl files   
# import pickle
# # with open('Si_444_right_no-molopt.pkl', 'rb') as f:
# # with open('H2-compute_dm_mo-nk888.pkl', 'rb') as f:
# with open('phosphorous_dm-mo_nk222.pkl', 'rb') as f:
#     ss_input = pickle.load(f)

# dm = np.array(ss_input['dm_kpts'])
# mo_coeff = np.array(ss_input['mo_coeff_kpts'])


# Read scf Result
from pyscf.lib import chkfile
chkfile_result =chkfile.load('phosphorous-kmf-nk222.chk','scf')
mf = khf.KRHF(cell, exxdiv='ewald')
mf.__dict__.update(chkfile_result)
Nk = np.prod(kmesh)

# Load GDF's CDERIs
df_type = df.GDF
df = df_type(cell, kpts)
df._cderi = 'phosphorous-df-nk222.h5'
df._cderi_to_save = None
mf.with_df = df.build()

# mf.with_df = df_type(cell, kpts).build()
dm_kpts = mf.make_rdm1()

# Regular energy components

h1e = mf.get_hcore()
ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm_kpts).real

Jo, Ko = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=mf.kpts, kpts_band=mf.kpts, with_j=True)

Ek = -1. / Nk * np.einsum('kij,kji', Ko, dm_kpts) * 0.5
Ej = 1. / Nk * np.einsum('kij,kji', Jo, dm_kpts)

Ek /= 2.
Ek = Ek.real
Ej /= 2.
Ej = Ej.real

print('Ek (a.u.) is ', Ek)
print('Ej (a.u.) is ', Ej)
print('Ehcore (a.u.) is ', ehcore)
print('Enuc (a.u.) is ', mf.energy_nuc().real)
print('Ecoul (a.u.) is ', Ek + Ej)

div_vector = [1,2]

import pyscf.pbc.scf.ss_localizers as ss_localizers
# localizer = lambda q, r1, M: ss_localizers.localizer_gauss_unbounded(q,r1,M=M)
def localizer(q,r1,M=np.array([1,1,1])):
    # return ss_localizers.localizer_gauss_unbounded(q,r1,M=M)
    return ss_localizers.localizer_unity(q,r1)
    # return ss_localizers.localizer(q,r1)

# localizer = lambda q,r1,M: ss_localizers.localizer_gauss(q,r1)
# Setup ss_params dict


# Compute SqG anisotropy, use for subtract_nocc_sigma
from pyscf.pbc.scf.khf import compute_SqG_anisotropy

sigmas = compute_SqG_anisotropy(cell=mf.cell,nks=kmesh, N_local=[5,17,8],dm_kpts=dm_kpts,mo_coeff_kpts=mf.mo_coeff_kpts,
                                SqG_filename='phosphorous_SqG_nk222.npy')

ss_params = {
    'debug': False,
    # 'r1_prefactor':10.0,
    'nlocal': 3,
    'localizer': localizer,
    'subtract_nocc': 2,
    'subtract_nocc_sigma': 0.65*sigmas,
    'use_sqG_anisotropy': False,
    'nufft_gl': True,
    'n_fft': 350,
    # 'M':ss_input['M'],
    'vhR_symm': False,
    'SqG_filenames':['phosphorous_SqG_nk222.npy',None],
    # 'SqG_filenames':[None,None],
    # 'H_use_unscaled': True,
    'delta':0.2,
    'gamma':1e-8,
    'r1_power_law_exponent':-5,
    'r1_power_law_start':1,
}

results = subsample_kpts(mf=mf,dim=3,div_vector=div_vector, df_type=df_type, khf_routine="singularity_subtraction",
                         wrap_around=wrap_around,ss_params=ss_params,sanity_run=False,mo_coeff_kpts=mf.mo_coeff_kpts,
                         dm_kpts=dm_kpts)