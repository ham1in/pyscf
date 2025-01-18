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
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import numint as pbcnumint
from pyscf import dft
from pyscf.dft import numint
from pyscf.lib import chkfile


cwd = os.getcwd()
nthreads = 4
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
def build_diamond_cell(nk = (1,1,1),kecut=100,wrap_around=True,with_gamma_point=True):
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
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)    
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
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    #cell.ke_cutoff = 55.13
    cell.ke_cutoff = kecut
    cell.max_memory = 240000
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=wrap_around,with_gamma_point=with_gamma_point)
    return cell, kpts

def build_cBN_cell(nk = (1,1,1),kecut=100,with_gamma_point=True,wrap_around=True,a=6.83324967102632):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    pos = a/4
    A_1d = a/2

    cell.atom = [['B',(0, 0, 0)],
                 ['N',(pos, pos, pos)]]

    cell.a = [[0.0,A_1d,A_1d],
              [A_1d,0.0,A_1d],
              [A_1d,A_1d,0.0]]


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

nk = 3
nks = np.array([nk, nk, nk])
Nk = np.prod(nks)
cell, kpts= build_cBN_cell(nk=nks,kecut=150,with_gamma_point=False)
cell.dimension = 3
cell.build()

print('Kmesh:', nks)

# # HF
# mf = khf.KRHF(cell, exxdiv='ewald')
# df_type = df.GDF
# mf.with_df = df_type(cell, kpts).build()

# Nk = np.prod(nks)
# mf.exxdiv = 'ewald'
# e1 = mf.kernel()


# num_gaussians = 1
# force_centered = True
# force_isotropic = True
# fit_with_coul = True
# sigma_multiplier = 1.0

# N_local = [3,3,3]
# sigma = 0.7
# results = khf_ssng(mf, nks, num_gaussians=num_gaussians, force_centered=force_centered, force_isotropic=force_isotropic,
#                     fit_with_coul=fit_with_coul,N_local=N_local,sigma_multiplier=sigma_multiplier,sigma=sigma)


# DFT
# Setup DFT object
dft.numint.NumInt.libxc = dft.xcfun
xc = 'PBE0'
xc_pure = "PBE"
x = 'PBEx'
c = 'PBEc'

mf = pbcdft.KRKS(cell, kpts)
mf.xc = xc
mf.exxdiv = 'ewald'
df_type = df.GDF
mf.with_df = df_type(cell, kpts).build()

# # If high nk, save to chkfile
if nk > 8:
    filename = 'cBN-ss1g-nk'+str(nk)*3+f'-a{a:.3f}.chk'
    mf.chkfile = filename

# # Load from chkfile
# scf_result_dic = chkfile.load('chk-1227/'+filename, 'scf')
# mf.__dict__.update(scf_result_dic)
# print('E(HF) from chkfile = %s' % mf.e_tot)
# e1 = mf.e_tot

# Run and extract Density Matrix
e1 = mf.kernel()


# Extract density matriz
dm_kpts = mf.make_rdm1()

## Compute DFT energy components
ni = pbcnumint.KNumInt()
_,exc_pure, vxc_pure = pbcnumint.nr_rks(ni,cell, mf.grids, xc_pure, dm_kpts, kpts=kpts)
_,ex_pure, vx_pure = pbcnumint.nr_rks(ni,cell, mf.grids, x+',', dm_kpts, kpts=kpts)
_,ec, vc = pbcnumint.nr_rks(ni,cell, mf.grids, ","+c, dm_kpts, kpts=kpts)
_, _, hyb = ni.rsh_and_hybrid_coeff(xc, spin=cell.spin)

# Nuclear, core, and Hartree energies
h1e = mf.get_hcore()
ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm_kpts).real
enuc = mf.energy_nuc().real
Jo, Ko = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=mf.kpts, kpts_band=mf.kpts, with_j=True)

Ej = 1. / Nk * np.einsum('kij,kji', Jo, dm_kpts)
Ej /= 2.
Ej = Ej.real

## Regular Exchange energy
Ek = -1. / Nk * np.einsum('kij,kji', Ko, dm_kpts) * 0.5
Ek /= 2.
Ek = Ek.real

## SSNG exchange

num_gaussians = 1
force_centered = True
force_isotropic = True
fit_with_coul = True
sigma_multiplier = 1.0
fit_method = "scipy_least_squares"
qG_norm_cutoff = "auto"


N_local = [9,9,9]
results = khf_ssng(mf, nks, num_gaussians=num_gaussians, force_centered=force_centered, force_isotropic=force_isotropic,
            fit_with_coul=fit_with_coul,N_local=N_local,sigma_multiplier=sigma_multiplier,fit_method=fit_method,
            qG_norm_cutoff=qG_norm_cutoff)

Ek_ss_ng = results['Ek_ss_ng']


## Staggered mesh

ex = hyb * Ek + (1-hyb) * ex_pure
ex_ss = hyb * Ek_ss_ng + (1-hyb) * ex_pure
exc = ex + ec
exc_ss = ex_ss + ec

# Print results
print("== Computing DFT Energy Components (a.u.) == ")
print('Ehcore = {:.15f}'.format(ehcore))
print('Ej     = {:.15f}'.format(Ej))
print('Enuc   = {:.15f}'.format(enuc))
print('Exc    = {:.15f}'.format(exc))
print('    Ex calculation: {:.2f} * Ex_pure ({}) + {:.2f} * Ek_HF'.format(1-hyb,x,hyb))
print('    Ex       = {:.15f}'.format(ex))
print('        Ek_HF    = {:.15f}'.format(Ek))
print('        Ex_pure  = {:.15f}'.format(ex_pure))
print('    Ec       = {:.15f}'.format(ec))
print('Exc_ss = {:.15f}'.format(exc_ss))
print('    Ex_ss calculation: {:.2f} * Ex_pure ({}) + {:.2f} * Ek_ssng'.format(1-hyb,x,hyb))
print('    Ex_ss    = {:.15f}'.format(ex_ss))
print('        Ek_ss_ng = {:.15f}'.format(Ek_ss_ng))
print('        Ex_pure  = {:.15f}'.format(ex_pure))
print('    Ec       = {:.15f}'.format(ec))


Etot = exc + Ej + enuc + ehcore
Etot_ss = exc_ss + Ej + enuc + ehcore

print('Etot (a.u.) = {:.15f}'.format(Etot))
print('Etot_ss (a.u.) = {:.15f}'.format(Etot_ss))
print('mf.kernel energy (a.u.) is ', e1)

assert np.isclose(exc, ex + ec, atol=1e-7)