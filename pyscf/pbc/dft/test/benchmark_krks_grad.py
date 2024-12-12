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
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import numint as pbcnumint
from pyscf import dft
from pyscf.dft import numint


# from pyscf.pbc.dft import r_numint


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

# wrap_around = True
# nkx = 2
# kmesh = [nkx, nkx, nkx]
# with_gamma_point = False
# cell, kpts= build_diamond_cell(nk=kmesh,kecut=56,wrap_around=wrap_around)
# cell.dimension = 3
# cell.build()
# Nk = np.prod(kmesh)




# global cell, kpts, disp


cell = pbcgto.Cell()
cell.atom='''
C 0.0 0.0 0.0
C 1.68516327271508 1.68516327271508 1.68516327271508
'''

cell.a = '''
0.0 3.370326545430162 3.370326545430162
3.370326545430162 0.0 3.370326545430162
3.370326545430162 3.370326545430162 0.0
'''
cell.basis = 'gth-szv'
# cell.verbose = 7
cell.pseudo = 'gth-pbe'
cell.unit = 'bohr'
cell.mesh = [13] * 3
cell.output = '/dev/null'


nks = np.array([1, 1, 3])
kpts = cell.make_kpts(nks)
Nk = np.prod(nks)
disp = 1e-5

cell.build()





# Setup DFT grad
# disp = 1e-5

mf = pbcdft.KRKS(cell, kpts)
mf.xc = 'PBE'
# mf.exxdiv = None
# mf.conv_tol = 1e-10
# mf.conv_tol_grad = 1e-6
# g_scan = mf.nuc_grad_method().as_scanner()
# g = g_scan(cell)[1]
# # self.assertAlmostEqual(lib.fp(g), -0.19544969829285652, 6)

# mfs = g_scan.base.as_scanner()
# e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
# e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
# # self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)



# mf.xc = f'{HF_X:} * HF + {LDA_X:} * LDA + {B88_X:} * B88, {LYP_C:} * LYP + {VWN_C:} * VWN'
e1 = mf.kernel()
dm_kpts = mf.make_rdm1()
ni = pbcnumint.KNumInt()
_,exc, vxc = pbcnumint.nr_rks(ni,cell, mf.grids, "PBE", dm_kpts, kpts=kpts)
_,ex, vx = pbcnumint.nr_rks(ni,cell, mf.grids, "PBE,", dm_kpts, kpts=kpts)
_,ec, vc = pbcnumint.nr_rks(ni,cell, mf.grids, ",PBE", dm_kpts, kpts=kpts)


# ao_value = numint.eval_ao(mf.cell, mf.grids.coords, kpts=mf.kpts)
# ao_kpts = numint.eval_ao_kpts(cell, mf.grids.coords, kpts=kpts, deriv=0)
# ao_kpts = np.array(ao_kpts)

# rho = numint.eval_rho(cell, ao_kpts, dm_kpts, xctype='GGA')
# ni = pbcnumint.KNumInt()
# rho = pbcnumint.get_rho(ni,mf.cell, dm_kpts, mf.grids, kpts=mf.kpts)
# ex, vx = dft.libxc.eval_xc('PBE0', rho)[:2]
# ec, vc = dft.libxc.eval_xc(',PBE0', rho)[:2]


# vx = get_

h1e = mf.get_hcore()
ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm_kpts).real

Jo, Ko = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=mf.kpts, kpts_band=mf.kpts, with_j=True)

Exc = 1. / Nk * np.einsum('kij,kji', vxc, dm_kpts)
Exc = Exc.real
Exc /= 2.


Ek = -1. / Nk * np.einsum('kij,kji', Ko, dm_kpts) * 0.5
Ek /= 2.
Ek = Ek.real

Ej = 1. / Nk * np.einsum('kij,kji', Jo, dm_kpts)
Ej /= 2.
Ej = Ej.real

print('Ek (a.u.) is ', Ek)
print('Ej (a.u.) is ', Ej)
print('Exc (a.u.) is ', Exc)

print('Ehcore (a.u.) is ', ehcore)
print('Enuc (a.u.) is ', mf.energy_nuc().real)
print('Ecoul (a.u.) is ', Ek + Ej)
print('Etot (a.u.) is ',Ej + exc + ehcore + mf.energy_nuc().real)


mf.xc = 'hf'
e2 = mf.kernel()

dm_kpts = mf.make_rdm1()

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
print('Etot (a.u.) is ', Ek + Ej + ehcore + mf.energy_nuc().real)

print(e1, e2)

