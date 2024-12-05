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
from pyscf.pbc.scf.khf import khf_ssng
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
    cell.max_memory = 120000
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
    return cell, kpts

def build_SnS_cell(nk = (1,1,1),kecut=100,with_gamma_point=True,wrap_around=True):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
Sn  1.8693430   3.1653933   13.3823814
Sn  5.6080291   5.4003768   8.1454120
Sn  5.6080291   7.4482784   18.9093087
Sn  1.8693430   1.1174917   2.6184847
S   1.8693430   0.0907698   7.4808535
S   5.6080291   8.4750004   14.0469399
S   5.6080291   4.3736548   3.2830432
S   1.8693430   4.1921153   18.2447503
        '''

              
    cell.a = '''
7.477372123464  0.000000000000  0.000000000000
0.000000000000  8.565770162297  0.000000000000
0.000000000000  0.000000000000  21.527793440959
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

def build_SnTe_cell(nk = (1,1,1),kecut=100,with_gamma_point=True,wrap_around=True):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
Sn  0.00000000000   0.00000000000   0.00000000000
Te  6.02031374618   6.02031374618   6.02031374618
        '''

              
    cell.a = '''
0.00000000000   6.02031374618   6.02031374618
6.02031374618   0.00000000000   6.02031374618
6.02031374618   6.02031374618   0.00000000000

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
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    #cell.ke_cutoff = 55.13
    cell.ke_cutoff = kecut
    cell.max_memory = 240000
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
    return cell, kpts


nkx = 2
nks = [nkx, nkx, nkx]
cell, kpts= build_Si_cell(nk=nks,kecut=56)
cell.dimension = 3
cell.build()

print('Kmesh:', nks)

mf = khf.KRHF(cell, exxdiv='ewald')
df_type = df.GDF
mf.with_df = df_type(cell, kpts).build()

Nk = np.prod(nks)
mf.exxdiv = 'ewald'
e1 = mf.kernel()


num_gaussians = 1
force_centered = True
force_isotropic = True
fit_with_coul = True
sigma_multiplier = 0.7

N_local = [9,9,9]
results = khf_ssng(mf, nks, num_gaussians=num_gaussians, force_centered=force_centered, force_isotropic=force_isotropic, 
                    fit_with_coul=fit_with_coul,N_local=N_local,sigma_multiplier=sigma_multiplier)