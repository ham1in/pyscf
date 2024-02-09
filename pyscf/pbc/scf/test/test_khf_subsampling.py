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


def build_carbon_chain_cell(nk=(1, 1, 1), kecut=100, bla_angstrom=0.128, a_angstrom=2.578, vac_angstrom=6.0):
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
    cell.basis = 'gth-szv'
    cell.precision = 1e-8
    cell.pseudo = 'gth-pbe'  # PAW-PBE used in Gruneis paper
    cell.ke_cutoff = kecut
    cell.max_memory = 30000

    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=True)
    return cell, kpts

# L = 4
# cell = pbcgto.Cell()
# A = np.eye(3) * 4,
#
# A = np.array([[4,0,0],
#               [0,4,0],
#               [0,0,4]])


nkx = 8
kmesh = [nkx, 1, 1]
cell, kpts= build_carbon_chain_cell(nk=kmesh,kecut=56,bla_angstrom=0.125)

# f = open(cell.output, "a")
# f.write("Full kpoint test\n")
# f.write('testing mean method\n')
print('Kmesh:', kmesh)

mf = khf.KRHF(cell, exxdiv='ewald')
mf.with_df = df.FFTDF(cell, kpts).build()

Nk = np.prod(kmesh)
mf.exxdiv = 'ewald'
e1 = mf.kernel()
dm = mf.make_rdm1()

# Regular energy components

h1e = mf.get_hcore()
ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm).real

Jo, Ko = mf.get_jk(cell=mf.cell, dm_kpts=dm, kpts=mf.kpts, kpts_band=mf.kpts, with_j=True)
np.save('carbyne-Jo'+ str(nkx) + '.npy', Jo)
np.save('carbyne-Ko'+ str(nkx) + '.npy', Ko)
np.save('carbyne-h1e'+ str(nkx) + '.npy', h1e)


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
print(f'\nSubsampling to  ',Nk/2 ,' kpoints')

Ko_div1 = Ko[0::2]
Jo_div1 = Jo[0::2]
dm_div1 = dm[0::2]
h1e_div1 = h1e[0::2]

Nk /= 2

ehcore = 1. / Nk * np.einsum('kij,kji->', h1e_div1, dm_div1).real
Ek = -1. / Nk * np.einsum('kij,kji', Ko_div1, dm_div1) * 0.5
Ej = 1. / Nk * np.einsum('kij,kji', Jo_div1, dm_div1)
Ek /= 2.
Ek = Ek.real
Ej /= 2.
Ej = Ej.real

print('Ek (a.u.) is ', Ek)
print('Ej (a.u.) is ', Ej)
print('Ehcore (a.u.) is ', ehcore)
print('Enuc (a.u.) is ', mf.energy_nuc().real)
print('Ecoul (a.u.) is ', Ek + Ej)


# Subsample 4 kpts
print(f'\nSubsampling to  ',Nk/2 ,' kpoints')

Ko_div2 = Ko_div1[0::2]
Jo_div2 = Jo_div1[0::2]
dm_div2 = dm_div1[0::2]
h1e_div2 = h1e_div1[0::2]

Nk /= 2

ehcore = 1. / Nk * np.einsum('kij,kji->', h1e_div2, dm_div2).real
Ek = -1. / Nk * np.einsum('kij,kji', Ko_div2, dm_div2) * 0.5
Ej = 1. / Nk * np.einsum('kij,kji', Jo_div2, dm_div2)
Ek /= 2.
Ek = Ek.real
Ej /= 2.
Ej = Ej.real

print('Ek (a.u.) is ', Ek)
print('Ej (a.u.) is ', Ej)
print('Ehcore (a.u.) is ', ehcore)
print('Enuc (a.u.) is ', mf.energy_nuc().real)
print('Ecoul (a.u.) is ', Ek + Ej)



# J_no_coulG = mf.with_df.get_electron_density(dm_kpts=dm_un, kpts=mf.kpts, kpts_band=mf.kpts)
# electron_density_l2 = 1. / Nk * np.einsum('kij,kji', J_no_coulG, dm_un)
# print('electron_density_l2 (a.u.) is ', electron_density_l2.real)
#
# print('Saving data for rho(q)')
#
# kpts_wrap = cell.make_kpts(nks=kmesh, wrap=True)
# np.save('h2-kpts-nkx' + str(nkx) + '.npy', kpts_wrap)
# ig = 0
# rhoG0 = mf.with_df.get_rhoG(dm_kpts=dm_un, kpts=mf.kpts, kpts_band=mf.kpts)[:, ig]
#
# np.save('h2-rhoG0-nkx' + str(nkx) + '.npy', rhoG0)