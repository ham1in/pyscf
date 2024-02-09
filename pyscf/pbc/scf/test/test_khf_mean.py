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
nthreads = 24
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)
os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)

lib.num_threads(os.getenv('OMP_NUM_THREADS'))


def nk_output_str(nk):
    return '-nk' + str(nk[0]) + str(nk[1]) + str(nk[2])


def kecut_output_str(kecut):
    return '-kecut' + str(kecut)


def build_carbon_chain_cell(nk=(1, 1, 1), kecut=100, bla_angstrom=0.128, a_angstrom=2.578, vac_angstrom=6.0, basis='gth-szv'):
    cell = pbcgto.Cell()
    cell.unit = 'angstrom'  # Not bohr!

    # smaller bond length
    bond_length = (a_angstrom - bla_angstrom) / 2.
    xpos = [(a_angstrom - bond_length) / 2., (a_angstrom + bond_length) / 2.]

    cell.atom = (['C', [xpos[0], 0, 0]],
                 ['C', [xpos[1], 0, 0]])

    cell.a = [[a_angstrom, 0, 0],
              [0, vac_angstrom, 0],
              [0, 0, vac_angstrom]]

    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    cell.basis = basis
    cell.precision = 1e-8
    cell.pseudo = 'gth-pbe'  # PAW-PBE used in Gruneis paper
    cell.ke_cutoff = kecut
    cell.max_memory = 30000

    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=False)
    return cell, kpts

# L = 4
# cell = pbcgto.Cell()
# A = np.eye(3) * 4,
#
# A = np.array([[4,0,0],
#               [0,4,0],
#               [0,0,4]])


kmesh = [8,1, 1]
bla = 0.125
cell, kpts= build_carbon_chain_cell(nk=kmesh,kecut=100,bla_angstrom=bla,basis='gth-dzv')

# Compute Mean Method Exchange

print('testing regular method')
mf = khf.KRHF(cell,exxdiv='ewald')
mf.with_df = df.MDF(cell,kpts).build()

# mf.kpts = cell.make_kpts(kmesh)
Nk = np.prod(kmesh)
mf.exxdiv = 'ewald'
e1 = mf.kernel()
dm_un = mf.make_rdm1()

#Regular exchange
# Jo, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un, kpts = mf.kpts, kpts_band = mf.kpts, with_j = True)
Jo, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un)

vhf_kpts = mf.get_veff(mf.cell, dm_un)
e_coul = 1./Nk * np.einsum('kij,kji', dm_un, vhf_kpts) * 0.5


Ek = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
Ek /=2
print('Ek (a.u.) is ')
print(Ek)


# Mean method
# mf.exxdiv = 'mean'
# cell.dimension = 2
# _, Ko = mf.get_jk(cell = mf.cell, dm_kpts = dm_un, kpts = mf.kpts, kpts_band = mf.kpts, with_j = False)
# Ek_mean = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
# Ek_mean /=2
#
# Ek_mean = Ek_mean.real
# print('Ek_mean (a.u.) is ')
# print(Ek_mean)

# Extract rhoG
# J_no_coulG = mf.with_df.get_electron_density(dm_kpts=dm_un, kpts=mf.kpts, kpts_band=mf.kpts)
# electron_density_l2 = 1. / Nk * np.einsum('kij,kji', J_no_coulG, dm_un)
#
# print('electron_density_l2 (a.u.) is ', electron_density_l2.real)
# print('Printing data for rho(q)')
#
# rhoG, rhoR = mf.with_df.get_rhoG(dm_kpts=dm_un, kpts=mf.kpts, kpts_band=mf.kpts)
# electron_density_l2 = 1. / Nk * np.einsum('kij,kji', J_no_coulG, dm_un)
#
# rhok = np.sum(rhoG,axis=1)
# rhoR = np.sum(rhoR,axis=0)
#
# mesh = cell.mesh
# grid = cell.gen_uniform_grids(mesh = mesh)
#
# x_only = (grid[:, 1] == 0) & (grid[:, 2] == 0)
# rhoR = rhoR[x_only]
# grid = grid[x_only]
#
# # rhok_G0 = rhoG[:,0]
# print('Saving data for rho(q)')
# print(rhok)
# ng = 1
# for ig in range(ng):
#     print('G index', ig)
#     rhok_G = rhoG[:, ig]
#     print(rhok_G)