#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

import sys

from functools import reduce
import numpy as np
import scipy.linalg
import h5py
from pyscf.pbc.scf import hf as pbchf
from pyscf import lib
from pyscf.scf import hf as mol_hf
from pyscf.lib import logger
from pyscf.pbc.gto import ecp
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile  # noqa
from pyscf.pbc import tools
from pyscf.pbc import df
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.pbc.lib.kpts import KPoints
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'pbc_scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'pbc_scf_analyze_pre_orth_method', 'ANO')
CHECK_COULOMB_IMAG = getattr(__config__, 'pbc_scf_check_coulomb_imag', True)


def get_ovlp(mf, cell=None, kpts=None):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    return pbchf.get_ovlp(cell, kpts)


def get_hcore(mf, cell=None, kpts=None):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    if cell.pseudo:
        nuc = lib.asarray(mf.with_df.get_pp(kpts))
    else:
        nuc = lib.asarray(mf.with_df.get_nuc(kpts))
    if len(cell._ecpbas) > 0:
        nuc += lib.asarray(ecp.ecp_int(cell, kpts))
    t = lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    return nuc + t


def get_j(mf, cell, dm_kpts, kpts, kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.  It needs to be Hermitian.

    Kwargs:
        kpts_band : (k,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    return df.FFTDF(cell).get_jk(dm_kpts, kpts, kpts_band, with_k=False)[0]


def get_jk(mf, cell, dm_kpts, kpts, kpts_band=None, with_j=True, with_k=True,
           omega=None, **kwargs):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point. It needs to be Hermitian.

    Kwargs:
        kpts_band : (3,) ndarray
            A list of arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    return df.FFTDF(cell).get_jk(dm_kpts, kpts, kpts_band, with_j, with_k,
                                 omega, exxdiv=mf.exxdiv)

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    f_kpts = h1e_kpts + vhf_kpts
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f_kpts

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s_kpts is None: s_kpts = mf.get_ovlp()
    if dm_kpts is None: dm_kpts = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f_kpts = [mol_hf.damping(s1e, dm_kpts[k] * 0.5, f_kpts[k], damp_factor)
                  for k, s1e in enumerate(s_kpts)]
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts = [mol_hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                  for k, s in enumerate(s_kpts)]
    return lib.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''Fermi level
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ

    # mo_energy_kpts and mo_occ_kpts are k-point RHF quantities
    assert (mo_energy_kpts[0].ndim == 1)
    assert (mo_occ_kpts[0].ndim == 1)

    # occ array in mo_occ_kpts may have different size. See issue #250
    nocc = sum(mo_occ.sum() for mo_occ in mo_occ_kpts) / 2
    # nocc may not be perfect integer when smearing is enabled
    nocc = int(nocc.round(3))
    fermi = np.sort(np.hstack(mo_energy_kpts))[nocc-1]

    for k, mo_e in enumerate(mo_energy_kpts):
        mo_occ = mo_occ_kpts[k]
        if mo_occ[mo_e > fermi].sum() > 1.:
            logger.warn(mf, 'Occupied band above Fermi level: \n'
                        'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    return fermi

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nkpts = len(mo_energy_kpts)
    nocc = mf.cell.tot_electrons(nkpts) // 2

    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(np.double) * 2)

    if nocc < mo_energy.size:
        logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
    else:
        logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]> 0]),
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]==0]))
        np.set_printoptions(threshold=1000)

    return mo_occ_kpts


def get_grad(mo_coeff_kpts, mo_occ_kpts, fock):
    '''
    returns 1D array of gradients, like non K-pt version
    note that occ and virt indices of different k pts now occur
    in sequential patches of the 1D array
    '''
    nkpts = len(mo_occ_kpts)
    grad_kpts = [mol_hf.get_grad(mo_coeff_kpts[k], mo_occ_kpts[k], fock[k])
                 for k in range(nkpts)]
    return np.hstack(grad_kpts)


def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    '''One particle density matrices for all k-points.

    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    '''
    nkpts = len(mo_occ_kpts)
    dm = [mol_hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k]) for k in range(nkpts)]
    return lib.tag_array(dm, mo_coeff=mo_coeff_kpts, mo_occ=mo_occ_kpts)


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(dm_kpts)
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts, h1e_kpts)
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts, vhf_kpts) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if CHECK_COULOMB_IMAG and abs(e_coul.imag > mf.cell.precision*10):
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real


def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment
    '''
    mf.dump_scf_summary(verbose)

    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        return mf.mulliken_meta(mf.cell, dm, s=ovlp_ao, verbose=verbose)
    else:
        raise NotImplementedError
        #return mf.mulliken_pop(mf.cell, dm, s=ovlp_ao, verbose=verbose)


def mulliken_meta(cell, dm_ao_kpts, verbose=logger.DEBUG,
                  pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''A modified Mulliken population analysis, based on meta-Lowdin AOs.

    Note this function only computes the Mulliken population for the gamma
    point density matrix.
    '''
    from pyscf.lo import orth
    if s is None:
        s = get_ovlp(cell)
    log = logger.new_logger(cell, verbose)
    log.note('Analyze output for *gamma point*')
    log.info('    To include the contributions from k-points, transform to a '
             'supercell then run the population analysis on the supercell\n'
             '        from pyscf.pbc.tools import k2gamma\n'
             '        k2gamma.k2gamma(mf).mulliken_meta()')
    log.note("KRHF mulliken_meta")
    dm_ao_gamma = dm_ao_kpts[0,:,:].real
    s_gamma = s[0,:,:].real
    orth_coeff = orth.orth_ao(cell, 'meta_lowdin', pre_orth_method, s=s_gamma)
    c_inv = np.dot(orth_coeff.T, s_gamma)
    dm = reduce(np.dot, (c_inv, dm_ao_gamma, c_inv.T.conj()))

    log.note(' ** Mulliken pop on meta-lowdin orthogonal AOs **')
    return mol_hf.mulliken_pop(cell, dm, np.eye(orth_coeff.shape[0]), log)


def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)
    mo_coeff = []
    mo_energy = []
    for k, mo in enumerate(mo_coeff_kpts):
        mo1 = np.empty_like(mo)
        mo_e = np.empty_like(mo_occ_kpts[k])
        occidx = mo_occ_kpts[k] == 2
        viridx = ~occidx
        for idx in (occidx, viridx):
            if np.count_nonzero(idx) > 0:
                orb = mo[:,idx]
                f1 = reduce(np.dot, (orb.T.conj(), fock[k], orb))
                e, c = scipy.linalg.eigh(f1)
                mo1[:,idx] = np.dot(orb, c)
                mo_e[idx] = e
        mo_coeff.append(mo1)
        mo_energy.append(mo_e)
    return mo_energy, mo_coeff

def _cast_mol_init_guess(fn):
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = fn(cell)
        nkpts = len(kpts)
        dm_kpts = np.asarray([dm] * nkpts)
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = [dm.mo_coeff] * nkpts
            mo_occ = [dm.mo_occ] * nkpts
            dm_kpts = lib.tag_array(dm_kpts, mo_coeff=mo_coeff, mo_occ=mo_occ)
        return dm_kpts
    fn_init_guess.__name__ = fn.__name__
    fn_init_guess.__doc__ = (
        'Generates initial guess density matrix and the orbitals of the initial '
        'guess DM ' + fn.__doc__)
    return fn_init_guess

def init_guess_by_minao(cell, kpts=None):
    '''Generates initial guess density matrix and the orbitals of the initial
    guess DM based on ANO basis.
    '''
    return KSCF(cell).init_guess_by_minao(cell, kpts)

def init_guess_by_atom(cell, kpts=None):
    '''Generates initial guess density matrix and the orbitals of the initial
    guess DM based on the superposition of atomic HF density matrix.
    '''
    return KSCF(cell).init_guess_by_atom(cell, kpts)

def init_guess_by_chkfile(cell, chkfile_name, project=None, kpts=None):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    from pyscf.pbc.scf import kuhf
    dm = kuhf.init_guess_by_chkfile(cell, chkfile_name, project, kpts)
    return dm[0] + dm[1]


def dip_moment(cell, dm_kpts, unit='Debye', verbose=logger.NOTE,
               grids=None, rho=None, kpts=np.zeros((1,3))):
    ''' Dipole moment in the cell (is it well defined)?

    Args:
         cell : an instance of :class:`Cell`

         dm_kpts (a list of ndarrays) : density matrices of k-points

    Return:
        A list: the dipole moment on x, y and z components
    '''
    from pyscf.pbc.dft import gen_grid
    from pyscf.pbc.dft import numint
    if grids is None:
        grids = gen_grid.UniformGrids(cell)
    if rho is None:
        rho = numint.KNumInt().get_rho(cell, dm_kpts, grids, kpts, cell.max_memory)
    return pbchf.dip_moment(cell, dm_kpts, unit, verbose, grids, rho, kpts)

def get_rho(mf, dm=None, grids=None, kpts=None):
    '''Compute density in real space
    '''
    from pyscf.pbc.dft import gen_grid
    from pyscf.pbc.dft import numint
    if dm is None:
        dm = mf.make_rdm1()
    if getattr(dm[0], 'ndim', None) != 2:  # KUHF
        dm = dm[0] + dm[1]
    if grids is None:
        grids = gen_grid.UniformGrids(mf.cell)
    if kpts is None:
        kpts = mf.kpts
    ni = numint.KNumInt()
    return ni.get_rho(mf.cell, dm, grids, kpts, mf.max_memory)

def as_scanner(mf):
    import copy
    if isinstance(mf, lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)

    class SCF_Scanner(mf.__class__, lib.SinglePointScanner):
        def __init__(self, mf_obj):
            self.__dict__.update(mf_obj.__dict__)

        def __call__(self, cell_or_geom, **kwargs):
            from pyscf.pbc import gto
            if isinstance(cell_or_geom, gto.Cell):
                cell = cell_or_geom
            else:
                cell = self.cell.set_geom_(cell_or_geom, inplace=False)

            # Cleanup intermediates associated to the pervious mol object
            self.reset(cell)

            if 'dm0' in kwargs:
                dm0 = kwargs.pop('dm0')
            elif self.mo_coeff is None:
                dm0 = None
            elif self.chkfile and h5py.is_hdf5(self.chkfile):
                dm0 = self.from_chk(self.chkfile)
            else:
                dm0 = self.make_rdm1()
                # dm0 form last calculation cannot be used in the current
                # calculation if a completely different system is given.
                # Obviously, the systems are very different if the number of
                # basis functions are different.
                # TODO: A robust check should include more comparison on
                # various attributes between current `mol` and the `mol` in
                # last calculation.
                if dm0.shape[-1] != cell.nao_nr():
                    #TODO:
                    #from pyscf.scf import addons
                    #if numpy.any(last_mol.atom_charges() != mol.atom_charges()):
                    #    dm0 = None
                    #elif non-relativistic:
                    #    addons.project_dm_nr2nr(last_mol, dm0, last_mol)
                    #else:
                    #    addons.project_dm_r2r(last_mol, dm0, last_mol)
                    dm0 = None
            self.mo_coeff = None  # To avoid last mo_coeff being used by SOSCF
            e_tot = self.kernel(dm0=dm0, **kwargs)
            return e_tot

    return SCF_Scanner(mf)


class KSCF(pbchf.SCF):
    '''SCF base class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points,
    e.g. mo_coeff is (nkpts, nao, nao) ndarray

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    conv_tol_grad = getattr(__config__, 'pbc_scf_KSCF_conv_tol_grad', None)
    direct_scf = getattr(__config__, 'pbc_scf_SCF_direct_scf', True)

    def __init__(self, cell, kpts=np.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        # Range separation JK builder
        self.rsjk = None

        self.exxdiv = exxdiv
        self.kpts = kpts
        self.conv_tol = max(cell.precision * 10, 1e-8)

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv', 'with_df', 'rsjk'])

    @property
    def kpts(self):
        if 'kpts' in self.__dict__:
            # To handle the attribute kpt loaded from chkfile
            self.kpt = self.__dict__.pop('kpts')
        return self.with_df.kpts

    @kpts.setter
    def kpts(self, x):
        self.with_df.kpts = np.reshape(x, (-1,3))
        if self.rsjk:
            self.rsjk.kpts = self.with_df.kpts

    @property
    def mo_energy_kpts(self):
        return self.mo_energy

    @property
    def mo_coeff_kpts(self):
        return self.mo_coeff

    @property
    def mo_occ_kpts(self):
        return self.mo_occ

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'N kpts = %d', len(self.kpts))
        logger.debug(self, 'kpts = %s', self.kpts)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        # "vcut_ws" precomputing is triggered by pbc.tools.pbc.get_coulG
        #if self.exxdiv == 'vcut_ws':
        #    if self.exx_built is False:
        #        self.precompute_exx()
        #    logger.info(self, 'WS alpha = %s', self.exx_alpha)
        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(cell, [self.kpts])
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            nkpts = len(self.kpts)
            # FIXME: consider the fractional num_electron or not? This maybe
            # relates to the charged system.
            nelectron = float(self.cell.tot_electrons(nkpts)) / nkpts
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*nelectron * -.5)
        if getattr(self, 'smearing_method', None) is not None:
            logger.info(self, 'Smearing method = %s', self.smearing_method)
        logger.info(self, 'DF object = %s', self.with_df)
        if not getattr(self.with_df, 'build', None):
            # .dump_flags() is called in pbc.df.build function
            self.with_df.dump_flags(verbose)
        return self

    def check_sanity(self):
        mol_hf.SCF.check_sanity(self)
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
            isinstance(self.with_df, df.df.DF)):
            logger.warn(self, 'exxdiv %s is not supported in DF or MDF',
                        self.exxdiv)
        return self

    def build(self, cell=None):
        if cell is None:
            cell = self.cell
        #if self.exxdiv == 'vcut_ws':
        #    self.precompute_exx()

        if 'kpts' in self.__dict__:
            # To handle the attribute kpts loaded from chkfile
            self.kpts = self.__dict__.pop('kpts')

        if self.rsjk:
            if not np.all(self.rsjk.kpts == self.kpts):
                self.rsjk = self.rsjk.__class__(cell, self.kpts)

        # Let df.build() be called by get_jk function later on needs.
        # DFT objects may need to initiailze df with different paramters.
        #if self.with_df:
        #    self.with_df.build()

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    get_init_guess = pbchf.SCF.get_init_guess

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return mol_hf.SCF.init_guess_by_1e(self, cell)

    init_guess_by_minao = _cast_mol_init_guess(mol_hf.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(mol_hf.init_guess_by_atom)

    get_hcore = get_hcore
    get_ovlp = get_ovlp
    get_fock = get_fock
    get_occ = get_occ
    energy_elec = energy_elec
    get_fermi = get_fermi

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpts=None,
              kpts_band=None, omega=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
                           with_k=False, omega=omega)[0]

    def get_k(self, cell=None, dm_kpts=None, hermi=1, kpts=None,
              kpts_band=None, omega=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
                           with_j=False, omega=omega)[1]

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            vj, vk = self.rsjk.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                      with_j, with_k, omega, self.exxdiv)
        else:
            vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                         with_j, with_k, omega, self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        return vj - vk * .5

    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, with_meta_lowdin, **kwargs)

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)
        return get_grad(mo_coeff_kpts, mo_occ_kpts, fock)

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        eig_kpts = []
        mo_coeff_kpts = []

        for k in range(nkpts):
            e, c = self._eigh(h_kpts[k], s_kpts[k])
            eig_kpts.append(e)
            mo_coeff_kpts.append(c)
        return eig_kpts, mo_coeff_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ

        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = (kpts_band.ndim == 1)
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_hcore(cell, kpts_band)
        fock = fock + self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        mo_energy, mo_coeff = self.eig(fock, s1e)
        if single_kpt_band:
            mo_energy = mo_energy[0]
            mo_coeff = mo_coeff[0]
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=None, kpts=None):
        if chk is None: chk = self.chkfile
        if kpts is None: kpts = self.kpts
        return init_guess_by_chkfile(self.cell, chk, project, kpts)
    def from_chk(self, chk=None, project=None, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        if self.chkfile:
            mol_hf.SCF.dump_chk(self, envs)
            with h5py.File(self.chkfile, 'a') as fh5:
                fh5['scf/kpts'] = self.kpts
        return self

    def mulliken_meta(self, cell=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta(cell, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    def mulliken_pop(self):
        raise NotImplementedError

    get_rho = get_rho

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, cell=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        rho = kwargs.pop('rho', None)
        if rho is None:
            rho = self.get_rho(dm)
        if cell is None:
            cell = self.cell
        return dip_moment(cell, dm, unit, verbose, rho=rho, kpts=self.kpts, **kwargs)

    canonicalize = canonicalize

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import df_jk
        return df_jk.density_fit(self, auxbasis, with_df=with_df)

    def rs_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import rsdf_jk
        return rsdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def mix_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import mdf_jk
        return mdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def jk_method(self, J='FFTDF', K=None):
        '''
        Set up the schemes to evaluate Coulomb and exchange matrix

        FFTDF: planewave density fitting using Fast Fourier Transform
        AFTDF: planewave density fitting using analytic Fourier Transform
        GDF: Gaussian density fitting
        MDF: Gaussian and planewave mix density fitting
        RS: range-separation JK builder
        RSDF: range-separation density fitting
        '''
        if K is None:
            K = J

        if J != K:
            raise NotImplementedError('J != K')

        if 'DF' in J or 'DF' in K:
            if 'DF' in J and 'DF' in K:
                assert J == K
            else:
                df_method = J if 'DF' in J else K
                self.with_df = getattr(df, df_method)(self.cell, self.kpts)

        if 'RS' in J or 'RS' in K:
            self.rsjk = RangeSeparatedJKBuilder(self.cell, self.kpts)
            self.rsjk.verbose = self.verbose

        # For nuclear attraction
        if J == 'RS' and K == 'RS' and not isinstance(self.with_df, df.GDF):
            self.with_df = df.GDF(self.cell, self.kpts)

        nuc = self.with_df.__class__.__name__
        logger.debug1(self, 'Apply %s for J, %s for K, %s for nuc', J, K, nuc)
        return self

    def stability(self,
                  internal=getattr(__config__, 'pbc_scf_KSCF_stability_internal', True),
                  external=getattr(__config__, 'pbc_scf_KSCF_stability_external', False),
                  verbose=None):
        from pyscf.pbc.scf.stability import rhf_stability
        return rhf_stability(self, internal, external, verbose)

    def newton(self):
        from pyscf.pbc.scf import newton_ah
        return newton_ah.newton(self)

    def sfx2c1e(self):
        from pyscf.pbc.x2c import sfx2c1e
        return sfx2c1e.sfx2c1e(self)
    x2c = x2c1e = sfx2c1e

    def to_rhf(self, mf=None):
        '''Convert the input mean-field object to a KRHF/KROHF/KRKS/KROKS object'''
        return addons.convert_to_rhf(self, mf)

    def to_uhf(self, mf=None):
        '''Convert the input mean-field object to a KUHF/KUKS object'''
        return addons.convert_to_uhf(self, mf)

    def to_ghf(self, mf=None):
        '''Convert the input mean-field object to a KGHF/KGKS object'''
        return addons.convert_to_ghf(self, mf)

    def to_khf(self):
        return self

    as_scanner = as_scanner


class KRHF(KSCF, pbchf.RHF):
    def check_sanity(self):
        cell = self.cell
        if isinstance(self.kpts, KPoints):
            nkpts = self.kpts.nkpts
        else:
            nkpts = len(self.kpts)
        if cell.spin != 0 and nkpts % 2 != 0:
            logger.warn(self, 'Problematic nelec %s and number of k-points %d '
                        'found in KRHF method.', cell.nelec, nkpts)
        return KSCF.check_sanity(self)

    def get_init_guess(self, cell=None, key='minao'):
        dm_kpts = pbchf.SCF.get_init_guess(self, cell, key)
        nkpts = len(self.kpts)
        if dm_kpts.ndim == 2:
            # dm[nao,nao] at gamma point -> dm_kpts[nkpts,nao,nao]
            dm_kpts = np.repeat(dm_kpts[None,:,:], nkpts, axis=0)

        ne = np.einsum('kij,kji->', dm_kpts, self.get_ovlp(cell)).real
        # FIXME: consider the fractional num_electron or not? This maybe
        # relate to the charged system.
        nelectron = float(self.cell.tot_electrons(nkpts))
        if abs(ne - nelectron) > 0.01*nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne/nkpts, nelectron/nkpts)
            dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts

    def convert_from_(self, mf):
        '''Convert given mean-field object to KRHF'''
        addons.convert_to_rhf(mf, self)
        return self

    def nuc_grad_method(self):
        from pyscf.pbc.grad import krhf
        return krhf.Gradients(self)

del (WITH_META_LOWDIN, PRE_ORTH_METHOD)

def khf_stagger(icell,ikpts, version = "Non_SCF", df_type = None, dm_kpts = None, kshift_rel = 0.5, fourinterp = False,N_local=7):
    from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
    from pyscf.pbc import gto,scf
    #To Do: Additional control arguments such as custom shift, scf control (cycles ..etc), ...
    #Cell formatting used in the built in Madelung code
    def set_cell(mf):
        import copy
        Nk = get_monkhorst_pack_size(mf.cell, mf.kpts)
        ecell = copy.copy(mf.cell)
        ecell._atm = np.array([[1, mf.cell._env.size, 0, 0, 0, 0]])
        ecell._env = np.append(mf.cell._env, [0., 0., 0.])
        ecell.unit = 'B'
        # ecell.verbose = 0
        ecell.a = np.einsum('xi,x->xi', mf.cell.lattice_vectors(), Nk)
        ecell.mesh = np.asarray(mf.cell.mesh) * Nk
        return ecell

    #Function for Madelung constant calculation following formula in Stephen's paper
    def staggered_Madelung(cell_input, shifted, ew_eta = None, ew_cut = None, dm_kpts = None):
        #Here, the only difference from overleaf is that eta here is defined as 4eta^2 = eta_paper
        from pyscf.pbc.gto.cell import get_Gv_weights
        nk = get_monkhorst_pack_size(icell, ikpts)
        if ew_eta is None or ew_cut is None:
            ew_eta, ew_cut = cell_input.get_ewald_params(cell_input.precision, cell_input.mesh)
        chargs = cell_input.atom_charges()
        log_precision = np.log(cell_input.precision / (chargs.sum() * 16 * np.pi ** 2))
        ke_cutoff = -2 * ew_eta ** 2 * log_precision
        #Get FFT mesh from cutoff value
        mesh = cell_input.cutoff_to_mesh(ke_cutoff)
        # if cell_input.dimension <= 2:
        #     mesh[2] = 1
        # if cell_input.dimension == 1:
        #     mesh[1] = 1
        #Get grid
        Gv, Gvbase, weights = cell_input.get_Gv_weights(mesh = mesh)
        #Get q+G points
        G_combined = Gv + shifted
        absG2 = np.einsum('gi,gi->g', G_combined, G_combined)


        if cell_input.dimension ==3:
            # Calculate |q+G|^2 values of the shifted points
            qG2 = np.einsum('gi,gi->g', G_combined, G_combined)
            # Note: Stephen - remove those points where q+G = 0
            qG2[qG2 == 0] = 1e200
            # Now putting the ingredients together
            component = 4 * np.pi / qG2 * np.exp(-qG2 / (4 * ew_eta ** 2))
            #First term
            sum_term = weights*np.einsum('i->',component).real
            #Second Term
            sub_term = 2*ew_eta/np.sqrt(np.pi)
            return sum_term - sub_term

        elif cell_input.dimension == 2:  # Truncated Coulomb
            from scipy.special import erfc, erf
            # The following 2D ewald summation is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            def fn(eta, Gnorm, z):
                Gnorm_z = Gnorm * z
                large_idx = Gnorm_z > 20.0
                ret = np.zeros_like(Gnorm_z)
                x = Gnorm / 2. / eta + eta * z
                with np.errstate(over='ignore'):
                    erfcx = erfc(x)
                    ret[~large_idx] = np.exp(Gnorm_z[~large_idx]) * erfcx[~large_idx]
                    ret[large_idx] = np.exp((Gnorm * z - x ** 2)[large_idx]) * erfcx[large_idx]
                return ret

            def gn(eta, Gnorm, z):
                return np.pi / Gnorm * (fn(eta, Gnorm, z) + fn(eta, Gnorm, -z))

            def gn0(eta, z):
                return -2 * np.pi * (z * erf(eta * z) + np.exp(-(eta * z) ** 2) / eta / np.sqrt(np.pi))

            b = cell_input.reciprocal_vectors()
            inv_area = np.linalg.norm(np.cross(b[0], b[1])) / (2 * np.pi) ** 2
            # Perform the reciprocal space summation over  all reciprocal vectors
            # within the x,y plane.
            planarG2_idx = np.logical_and(Gv[:, 2] == 0, absG2 > 0.0)

            G_combined = G_combined[planarG2_idx]
            absG2 = absG2[planarG2_idx]
            absG = absG2 ** (0.5)
            # Performing the G != 0 summation.
            coords = np.array([[0,0,0]])
            rij = coords[:, None, :] - coords[None, :, :] # should be just the zero vector for correction.
            Gdotr = np.einsum('ijx,gx->ijg', rij, G_combined)
            ewg = np.einsum('i,j,ijg,ijg->', chargs, chargs, np.cos(Gdotr),
                            gn(ew_eta, absG, rij[:, :, 2:3]))
            # Performing the G == 0 summation.
            # ewg += np.einsum('i,j,ij->', chargs, chargs, gn0(ew_eta, rij[:, :, 2]))

            ewg *= inv_area # * 0.5

            ewg_analytical = 2 * ew_eta / np.sqrt(np.pi)
            return ewg - ewg_analytical


    if df_type is None:
        if icell.dimension <=2:
            df_type = df.GDF
        else:
            df_type = df.FFTDF

    if fourinterp:
        assert(version == "Non_SCF", "Fourier interpolation only available for Non-SCF version")

    if version == "One_shot":
        nks = get_monkhorst_pack_size(icell, ikpts)
        shift = icell.get_abs_kpts([kshift_rel / n for n in nks])
        if icell.dimension <=2:
            shift[2] =  0
        elif icell.dimension == 1:
            shift[1] = 0
        Nk = np.prod(nks) * 2
        print("Shift is: " + str(shift))
        kmesh_shifted = ikpts + shift
        combined = np.concatenate((ikpts,kmesh_shifted),axis=0)
        print(combined)

        mf2 = scf.KHF(icell, combined)
        mf2.with_df = df_type(icell, combined).build() #For 2d,1d, df_type cannot be FFTDF

        print(mf2.kernel())
        d_m = mf2.make_rdm1()
        #Get dm at kpoints in unshifted mesh
        dm2 = d_m[:Nk//2,:,:]
        #Get dm at kpoints in shifted mesh
        dm_shift = d_m[Nk//2:,:,:]
        #K matrix on shifted mesh with potential defined by dm on unshifted mesh
        _, Kmat = mf2.get_jk(cell=mf2.cell, dm_kpts= dm2, kpts=ikpts, kpts_band = kmesh_shifted)
        E_stagger = -1. / Nk * np.einsum('kij,kji', dm_shift, Kmat)
        E_stagger /= 2

        #Madelung constant computation
        count_iter = 1
        mf2.kpts = ikpts
        ecell = set_cell(mf2)
        ew_eta, ew_cut = ecell.get_ewald_params(mf2.cell.precision, mf2.cell.mesh)
        prev = 0
        conv_Madelung = 0
        while True and icell.dimension !=1:
            Madelung = staggered_Madelung(cell_input=ecell, shifted=shift, ew_eta=ew_eta, ew_cut=ew_cut)
            print("Iteration number " + str(count_iter))
            print("Madelung:" + str(Madelung))
            print("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(Madelung - prev) < 1e-8:
                conv_Madelung = Madelung
                break
            if count_iter > 30:
                print("Error. Madelung constant not converged")
                break
            ew_eta *= 2
            count_iter += 1
            prev = Madelung

        nocc = mf2.cell.tot_electrons() // 2
        E_stagger_M = E_stagger + nocc * conv_Madelung
        print("One Shot")

    elif version == "Two_shot":
        #Regular scf calculation
        mfs = scf.KHF(icell, ikpts)
        print(mfs.kernel())
        nks = get_monkhorst_pack_size(mfs.cell, mfs.kpts)
        shift = mfs.cell.get_abs_kpts([kshift_rel / n for n in nks])
        kmesh_shifted = mfs.kpts + shift
        #Calculation on shifted mesh
        mf2 = scf.KHF(icell, kmesh_shifted)
        mf2.with_df = df_type(icell, ikpts).build()  # For 2d,1d, df_type cannot be FFTDF
        print(mf2.kernel())
        dm_2 = mf2.make_rdm1()
        #Get K matrix on shifted kpts, dm from unshifted mesh
        _, Kmat = mf2.get_jk(cell = mf2.cell, dm_kpts = mfs.make_rdm1(), kpts = mfs.kpts, kpts_band = mf2.kpts)
        Nk = np.prod(nks)
        E_stagger = -1. / Nk * np.einsum('kij,kji', dm_2, Kmat) * 0.5
        E_stagger/=2

        #Madelung calculation
        count_iter = 1
        ecell = set_cell(mf2)
        ew_eta, ew_cut = ecell.get_ewald_params(mf2.cell.precision, mf2.cell.mesh)
        prev = 0
        conv_Madelung = 0
        while True and icell.dimension !=1 :
            Madelung = staggered_Madelung(cell_input=ecell, shifted=shift, ew_eta=ew_eta, ew_cut=ew_cut)
            print("Iteration number " + str(count_iter))
            print("Madelung:" + str(Madelung))
            print("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(Madelung - prev) < 1e-8:
                conv_Madelung = Madelung
                break
            if count_iter > 30:
                print("Error. Madelung constant not converged")
                break
            ew_eta *= 2
            count_iter += 1
            prev = Madelung

        nocc = mf2.cell.tot_electrons() // 2
        E_stagger_M = E_stagger + nocc * conv_Madelung

        print("Two Shot")
    else: # Non-SCF
        mf2 = scf.KHF(icell,ikpts, exxdiv='ewald')
        mf2.with_df = df_type(icell, ikpts).build()
        if dm_kpts is None:
            print(mf2.kernel())
            # Get converged density matrix
            dm_un = mf2.make_rdm1()
        else:
            dm_un = dm_kpts

        #Defining size and making shifted mesh
        nks = get_monkhorst_pack_size(mf2.cell, mf2.kpts)
        shift = mf2.cell.get_abs_kpts([kshift_rel/n for n in nks])
        if icell.dimension <=2:
            shift[2] =  0
        elif icell.dimension == 1:
            shift[1] = 0
        kmesh_shifted = mf2.kpts + shift
        print(mf2.kpts)
        print(kmesh_shifted)

        print("\n")
        if dm_kpts is None:
            print("Converged Density Matrix")
        else:
            print("Input density matrix")

        for i in range(0,dm_un.shape[0]):
            print("kpt: " + str(mf2.kpts[i]) + "\n")
            mat = dm_un[i,:,:]
            for j in mat:
                print(' '.join(str(np.real(el)) for el in j))

        #Construct the Fock Matrix
        h1e = get_hcore(mf2, cell = mf2.cell, kpts = kmesh_shifted)
        Jmat, Kmat = mf2.get_jk(cell = mf2.cell, dm_kpts = dm_un, kpts = mf2.kpts, kpts_band = kmesh_shifted,exxdiv='ewald')
        #Veff = Jmat - Kmat/2
        Veff = mf2.get_veff(cell = mf2.cell, dm_kpts = dm_un, kpts = mf2.kpts, kpts_band = kmesh_shifted)
        F_shift = h1e + Veff
        s1e = get_ovlp(mf2, cell = mf2.cell, kpts = kmesh_shifted)
        mo_energy_shift, mo_coeff_shift = mf2.eig(F_shift, s1e)
        mo_occ_shift = mf2.get_occ(mo_energy_kpts=mo_energy_shift, mo_coeff_kpts=mo_coeff_shift)
        dm_shift = mf2.make_rdm1(mo_coeff_kpts=mo_occ_shift,mo_occ_kpts = mo_occ_shift)


        if fourinterp:
            # Extract uKpts from each set of kpts
            
            E_madelung, E_madelung, uKpts1, qGrid, kGrid = make_ss_inputs(mf2,mf2.kpts,dm_un, mf2.mo_coeff_kpts())
            E_madelung, E_madelung, uKpts2, qGrid, kGrid = make_ss_inputs(mf2,kmesh_shifted,dm_shift, mo_coeff_shift)

            # Set some parameters
            nkpts = np.prod(nks)
            nbands = nocc
            NsCell = mf2.cell.mesh
            nG = np.prod(NsCell)
            
            Lvec_real = mf2.cell.lattice_vectors()
            L_delta = Lvec_real / NsCell[:, None]
            dvol = np.abs(np.linalg.det(L_delta))

            # Evaluate wavefunction on all real space grid points
            # Establishing real space grid (Generalized for arbitary volume defined by 3 vectors)
            xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
            mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
            rptGrid3D = mesh_idx @ L_delta
            #   Step 1.4: compute the pair product
            Lvec_recip = cell.reciprocal_vectors()
            Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
            Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
            Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
            Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
            GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip

            # aoval = kmf.cell.pbc_eval_gto("GTO
            SqG = np.zeros((nkpts, nG), dtype=np.float64)
            print("MEM USAGE IS:", SqG.nbytes)
            for q in range(nkpts):
                for k in range(nkpts):
                    temp_SqG_k = np.zeros(nG, dtype=np.float64)  # Temporary storage for sums over m, n for the current k and q

                    kpt1 = kGrid[k, :]
                    qpt = qGrid[q, :]
                    kpt2 = kpt1 + qpt

                    kpt2_BZ = minimum_image(mf2.cell, kpt2)
                    idx_kpt2 = np.where(np.sum((kGrid - kpt2_BZ[None, :]) ** 2, axis=1) < 1e-8)[0]
                    if len(idx_kpt2) != 1:
                        raise TypeError("Cannot locate (k+q) in the kmesh.")
                    idx_kpt2 = idx_kpt2[0]
                    kGdiff = kpt2 - kpt2_BZ


                    for n in range(nbands):
                        for m in range(nbands):
                            u1 = uKpts1[k, n, :]
                            u2 = np.squeeze(np.exp(-1j * (rptGrid3D @ np.reshape(kGdiff, (-1, 1))))) * uKpts2[idx_kpt2, m, :]
                            rho12 = np.reshape(np.conj(u1) * u2, (NsCell[0], NsCell[1], NsCell[2]))
                            temp_fft = np.fft.fftn((rho12 * dvol))
                            # Compute sums on the fly instead of storing in rho (For mem. reasons, rho doesn't too large for >5x5x5 in some systems)
                            temp_SqG_k += np.abs(temp_fft.reshape(-1)) ** 2

                    SqG[q, :] += temp_SqG_k / nkpts
            #SqG = np.sum(np.abs(rhokqmnG) ** 2, axis=(0, 2, 3)) / nkpts
            SqG = SqG - nocc  # remove the zero order approximate nocc
            assert (np.abs(SqG[0, 0]) < 1e-4)

            #   Exchange energy can be formulated as
            #   Ex = prefactor_ex * bz_dvol * sum_{q} (\sum_G S(q+G) * 4*pi/|q+G|^2)
            prefactor_ex = -1 / (8 * np.pi ** 3)
            bz_dvol = np.abs(np.linalg.det(Lvec_recip)) / nkpts

            #   Step 3.1: define the local domain as multiple of BZ
            Lvec_recip = cell.reciprocal_vectors()

            LsCell_bz_local = N_local * Lvec_recip
            LsCell_bz_local_norms = np.linalg.norm(LsCell_bz_local, axis=1)

            #   localizer for the local domain
            r1 = np.min(LsCell_bz_local_norms) / 2
            r1_prefactor = 1.0
            r1 = r1_prefactor * r1
            from pyscf.pbc.scf import ss_localizers
            H = lambda q: ss_localizers.localizer_step(q,r1)

            #   reciprocal lattice within the local domain
            Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
            Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, [0], indexing='ij')
            GptGrid3D_local = np.hstack(
                (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip

            #   location/index of GptGrid3D_local within 'GptGrid3D'
            idx_GptGrid3D_local = []
            for Gl in GptGrid3D_local:
                idx_tmp = np.where(np.linalg.norm(Gl[None, :] - GptGrid3D, axis=1) < 1e-8)[0]
                if len(idx_tmp) != 1:
                    raise TypeError("Cannot locate local G vector in the reciprocal lattice.")
                else:
                    idx_GptGrid3D_local.append(idx_tmp[0])
            idx_GptGrid3D_local = np.array(idx_GptGrid3D_local)

            #   focus on S(q + G) with q in qGrid and G in GptGrid3D_local
            SqG_local = SqG[:, idx_GptGrid3D_local]

            #   Step 3.2: compute the Fourier transform of 1/|q|^2
            nqG_local = N_local * nks  # lattice size along each dimension in the real-space (equal to q + G size)
            Lvec_real_local = Lvec_real / N_local  # dual real cell of local domain LsCell_bz_local

            Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
            Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
            Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
            Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
            RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local

            #   Kernel from Fourier Interpolation
            from scipy.special import sici
            normR = np.linalg.norm(RptGrid3D_local, axis=1)
            CoulR = 4 * np.pi / normR * sici(normR * r1)[0]
            CoulR[normR < 1e-8] = 4 * np.pi * r1

            #   Integral with Fourier Approximation
            Ex_stagger_fourier = 0.0
            for iq, qpt in enumerate(qGrid):
                qG = qpt[None, :] + GptGrid3D_local
                exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
                tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
                tmp = SqG_local[iq, :].T * H(qG) * tmp
                Ex_stagger_fourier += np.real(np.sum(tmp)) * bz_dvol
            Ex_stagger_fourier *= 4 * np.pi 
            return np.real(Ex_stagger_fourier), 0.0, np.real(E_madelung)

        else: # regular stagger

            #Computing the Staggered mesh energy
            Nk = np.prod(nks)
            E_stagger = -1./Nk * np.einsum('kij,kji', dm_shift,Kmat ) * 0.5
            E_stagger/=2

            count_iter = 1
            ecell = set_cell(mf2)
            ew_eta, ew_cut = ecell.get_ewald_params(mf2.cell.precision, mf2.cell.mesh)
            prev = 0
            conv_Madelung = 0
            while True and icell.dimension !=1:
                Madelung = staggered_Madelung( cell_input = ecell,  shifted = shift ,  ew_eta = ew_eta, ew_cut = ew_cut)
                print("Iteration number " + str(count_iter))
                print("Madelung:" + str(Madelung))
                print("Eta:" + str(ew_eta))
                if count_iter>1 and abs(Madelung-prev)<1e-8:
                    conv_Madelung = Madelung
                    break
                if count_iter>30:
                    print("Error. Madelung constant not converged")
                    break
                ew_eta*=2
                count_iter+=1
                prev = Madelung

            nocc = mf2.cell.tot_electrons()//2
            E_stagger_M = E_stagger + nocc*conv_Madelung
            print("Non SCF")
        # Standard Exchange energy
        Jo, Ko = mf2.get_jk(cell=mf2.cell, dm_kpts=dm_un, kpts=mf2.kpts, kpts_band=mf2.kpts,exxdiv='ewald')
        E_madelung = -1. / Nk * np.einsum('kij,kji', dm_un, Ko) * 0.5
        E_madelung /= 2

        return np.real(E_stagger_M), np.real(E_stagger), np.real(E_madelung)


def minimum_image(cell, kpts):
    """
    Compute the minimum image of k-points in 'kpts' in the first Brillouin zone

    Arguments:
        cell -- a cell instance
        kpts -- a list of k-points

    Returns:
        kpts_bz -- a list of k-point in the first Brillouin zone
    """
    tmp_kpt = cell.get_scaled_kpts(kpts)
    tmp_kpt = tmp_kpt - np.floor(tmp_kpt)
    tmp_kpt[tmp_kpt > 0.5 - 1e-8] -= 1
    kpts_bz = cell.get_abs_kpts(tmp_kpt)
    return kpts_bz

def khf_ss_3d(kmf, nks, uKpts, ex_madelung, N_local=7, debug=False, localizer=None, r1_prefactor=1.0,fourier_only=False):
    """
    Perform Singularity Subtraction for Fock Exchange (3D) calculation.

    Args:
        kmf (object): A Kohn-Sham mean-field object.
        nks (array-like): A 1D array-like object containing the number of k-points along each dimension.
        uKpts (ndarray): A 3D array containing the wavefunction evaluated on all grid points. From mak
        ex_madelung (float): The initial exchange energy with the madelung correction
        N_local (int, optional): Number of BZs to include for the support along 1 dimension. Defaults to 7.
        debug (bool, optional): Whether to print debug information. Defaults to False.
        localizer (callable, optional): Localizer function. Must take in q, r1 as arguments. Defaults the polynomial localizer.
        r1_prefactor (float, optional): A prefactor for the localizer radius. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the exchange energy with singularity subtraction (e_ex_ss) and an alternative exchange energy calculation (e_ex_ss2).
    """
    # Function implementation goes here
    #Xin's version - using for test/benchmarking
    print("Singularity Subtraction for Fock Exchange (3D) requested")
    from scipy.special import sici
    
    if localizer is None:
        import pyscf.pbc.scf.ss_localizers as ss_localizers
        localizer = lambda q, r1: ss_localizers.localizer_poly(q, r1, 4)

    #   basic info
    cell = kmf.cell
    kpts = kmf.kpts
    nks = np.array(nks)
    nocc = cell.tot_electrons() // 2
    nkpts = np.prod(nks)

    #   compute standard exchange energy without any correction
    # kmf.exxdiv = None
    # if dm_kpts is None:
    #     dm_kpts = kmf.make_rdm1()
    # vk_kpts = kmf.get_k(kmf.cell, dm_kpts)
    # e_ex = 1. / nkpts * np.einsum('kij,kji', dm_kpts, -0.5 * vk_kpts) * 0.5
    # print(f"Exchange energy without any correction: {e_ex.real}")

    #   compute the ewald correction
    # xi = madelung(kmf.cell, kmf.kpts)
    # e_ex_madelung = e_ex - nocc * xi
    # print(f"Exchange energy with Madelung correction: {e_ex_madelung.real}")

    #   compute the singularity subtraction correction

    #   Step 1: compute the pair product in reciproal space

    #   Step 1.1: evaluate AO on a real fine mesh in unit cell
    Lvec_real = kmf.cell.lattice_vectors()
    NsCell = kmf.cell.mesh
    L_delta = Lvec_real / NsCell[:, None]
    dvol = np.abs(np.linalg.det(L_delta))
    #Evaluate wavefunction on all real space grid points
    # # Establishing real space grid (Generalized for arbitary volume defined by 3 vectors)
    xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
    mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
    rptGrid3D = mesh_idx @ L_delta
    # aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kmf.kpts)

    #   Step 1.2: map q-mesh and k-mesh to BZ
    qGrid = minimum_image(cell, kpts - kpts[0, :])
    kGrid = minimum_image(cell, kpts)

    #   Step 1.3: evaluate MO periodic component on a real fine mesh in unit cell
    nbands = nocc
    nG = np.prod(NsCell)
    # uKpts = np.zeros((nkpts, nbands, nG), dtype=complex)
    # for k in range(nkpts):
    #     for n in range(nbands):
    #         #   mo_coeff_kpts is of dimension (nkpts, nbasis, nband)
    #         utmp = aoval[k] @ np.reshape(mo_coeff_kpts[k][:, n], (-1, 1))
    #         exp_part = np.exp(-1j * (rptGrid3D @ np.reshape(kGrid[k], (-1, 1))))
    #         uKpts[k, n, :] = np.squeeze(exp_part * utmp)

            #   Step 1.4: compute the pair product
    Lvec_recip = cell.reciprocal_vectors()
    Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
    Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
    Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
    Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
    GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip

    SqG = np.zeros((nkpts, nG), dtype=np.float64)
    print("MEM USAGE IS:", SqG.nbytes)
    for q in range(nkpts):
        for k in range(nkpts):
            temp_SqG_k = np.zeros(nG, dtype=np.float64)  # Temporary storage for sums over m, n for the current k and q

            kpt1 = kGrid[k, :]
            qpt = qGrid[q, :]
            kpt2 = kpt1 + qpt

            kpt2_BZ = minimum_image(kmf.cell, kpt2)
            idx_kpt2 = np.where(np.sum((kGrid - kpt2_BZ[None, :]) ** 2, axis=1) < 1e-8)[0]
            if len(idx_kpt2) != 1:
                raise TypeError("Cannot locate (k+q) in the kmesh.")
            idx_kpt2 = idx_kpt2[0]
            kGdiff = kpt2 - kpt2_BZ

            for n in range(nbands):
                for m in range(nbands):
                    u1 = uKpts[k, n, :]
                    u2 = np.squeeze(np.exp(-1j * (rptGrid3D @ np.reshape(kGdiff, (-1, 1))))) * uKpts[idx_kpt2, m, :]
                    rho12 = np.reshape(np.conj(u1) * u2, (NsCell[0], NsCell[1], NsCell[2]))
                    temp_fft = np.fft.fftn((rho12 * dvol))
                    # Compute sums on the fly instead of storing in rho (For mem. reasons, rho doesn't too large for >5x5x5 in some systems)
                    temp_SqG_k += np.abs(temp_fft.reshape(-1)) ** 2

            SqG[q, :] += temp_SqG_k / nkpts

    #SqG = np.sum(np.abs(rhokqmnG) ** 2, axis=(0, 2, 3)) / nkpts
    SqG = SqG - nocc  # remove the zero order approximate nocc
    assert (np.abs(SqG[0, 0]) < 1e-4)

    #   Exchange energy can be formulated as
    #   Ex = prefactor_ex * bz_dvol * sum_{q} (\sum_G S(q+G) * 4*pi/|q+G|^2)
    prefactor_ex = -1 / (8 * np.pi ** 3)
    bz_dvol = np.abs(np.linalg.det(Lvec_recip)) / nkpts

    #   Side Step: double check the validity of SqG by computing the exchange energy
    # if False:
    #     CoulG = np.zeros_like(SqG)
    #     for iq, qpt in enumerate(qGrid):
    #         qG = qpt[None, :] + GptGrid3D
    #         norm2_qG = np.sum(qG ** 2, axis=1)
    #         CoulG[iq, :] = 4 * np.pi / norm2_qG
    #         CoulG[iq, norm2_qG < 1e-8] = 0
    #     Ex = prefactor_ex * bz_dvol * np.sum((SqG + nocc) * CoulG)
    #     print(f'Ex = {Ex} = {e_ex.real}')

        #   Step 3: construct Fouier Approximation of S(q+G)h(q+G)

    #   Step 3.1: define the local domain as multiple of BZ
    LsCell_bz_local = N_local * Lvec_recip
    LsCell_bz_local_norms = np.linalg.norm(LsCell_bz_local, axis=1)

    #   localizer for the local domain
    r1 = np.min(LsCell_bz_local_norms) / 2
    r1 = r1_prefactor * r1
    H = lambda q: localizer(q,r1)

    #   reciprocal lattice within the local domain
    Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
    Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, [0], indexing='ij')
    GptGrid3D_local = np.hstack(
        (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip

    #   location/index of GptGrid3D_local within 'GptGrid3D'
    idx_GptGrid3D_local = []
    for Gl in GptGrid3D_local:
        idx_tmp = np.where(np.linalg.norm(Gl[None, :] - GptGrid3D, axis=1) < 1e-8)[0]
        if len(idx_tmp) != 1:
            raise TypeError("Cannot locate local G vector in the reciprocal lattice.")
        else:
            idx_GptGrid3D_local.append(idx_tmp[0])
    idx_GptGrid3D_local = np.array(idx_GptGrid3D_local)

    #   focus on S(q + G) with q in qGrid and G in GptGrid3D_local
    SqG_local = SqG[:, idx_GptGrid3D_local]

    #   Step 3.2: compute the Fourier transform of 1/|q|^2
    nqG_local = N_local * nks  # lattice size along each dimension in the real-space (equal to q + G size)
    Lvec_real_local = Lvec_real / N_local  # dual real cell of local domain LsCell_bz_local

    Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
    Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
    Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
    Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
    RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local

      #   Kernel from Fourier Interpolation
    normR = np.linalg.norm(RptGrid3D_local, axis=1)
    CoulR = 4 * np.pi / normR * sici(normR * r1)[0]
    CoulR[normR < 1e-8] = 4 * np.pi * r1

    #   Step 4: Compute the correction

    ss_correction = 0
    #   Integral with Fourier Approximation
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGrid3D_local
        exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
        tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
        tmp = SqG_local[iq, :].T * H(qG) * tmp
        ss_correction += np.real(np.sum(tmp)) * bz_dvol
    int_terms = ss_correction
    if fourier_only:
        print("Returning integral term with Fourier interpolation only. Please double check that the step localizer is used.")
        return  4*np.pi*int_terms
    
    #   Quadrature with Coulomb kernel
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGrid3D_local
        tmp = SqG_local[iq, :].T * H(qG) / np.sum(qG ** 2, axis=1)
        tmp[np.isinf(tmp) | np.isnan(tmp)] = 0
        ss_correction -= np.sum(tmp) * bz_dvol
    quad_terms = ss_correction - int_terms


    ss_correction = 4 * np.pi * ss_correction  # Coulomb kernel = 4 pi / |q|^2
    


    #   Step 5: apply the correction
    e_ex_ss = np.real(ex_madelung + prefactor_ex * ss_correction)

    #   Step 6: Lin's new idea
    e_ex_ss2 = 0
    #   Integral with Fourier Approximation
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGrid3D_local
        exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
        tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
        tmp = (SqG_local[iq, :].T + nocc) * tmp
        e_ex_ss2 += np.real(np.sum(tmp)) * bz_dvol
    e_ex_ss2 = np.real(prefactor_ex * 4 * np.pi * e_ex_ss2)

    return e_ex_ss, e_ex_ss2, int_terms, quad_terms


def khf_ss_2d(kmf, nks, uKpts, ex, N_local=5, debug=False, localizer=None, r1_prefactor=1.0):
    """
    Perform Singularity Subtraction for Fock Exchange (2D) calculation.

    Args:
        kmf (object): A Kohn-Sham mean-field object.
        nks (array-like): Number of k-points in each direction.
        uKpts (array-like): Kohn-Sham orbitals generated from make_ss_inputs.
        ex (float): Exchange energy (no Madelung correction).
        N_local (int, optional): Number of BZs for the support. Defaults to 5.
        debug (bool, optional): Enable printing of SqG, HqG, VqG mat files.
        localizer (function, optional): Localizer function. Defaults to the 2D Polynomial Localizer, d=4.
        r1_prefactor (float, optional): Scaling prefactor for the localizer radius. Defaults to 1.0.

    Returns:
        float: The singularity subtraction correction.

    Raises:
        TypeError: If the (k+q) point cannot be located in the kmesh.

    """
    from scipy.special import sici
    from scipy.special import iv
    
    print("Singularity Subtraction for Fock Exchange (2D) requested")
    def minimum_image(cell, kpts):
        """
        Compute the minimum image of k-points in 'kpts' in the first Brillouin zone

        Arguments:
            cell -- a cell instance
            kpts -- a list of k-points

        Returns:
            kpts_bz -- a list of k-point in the first Brillouin zone
        """
        tmp_kpt = cell.get_scaled_kpts(kpts)
        tmp_kpt = tmp_kpt - np.floor(tmp_kpt)
        tmp_kpt[tmp_kpt > 0.5 - 1e-8] -= 1
        kpts_bz = cell.get_abs_kpts(tmp_kpt)
        return kpts_bz

    if localizer is None:
        import pyscf.pbc.scf.ss_localizers as ss_localizers
        localizer = lambda q, r1: ss_localizers.localizer_poly_2d(q, r1, 4)

    #   basic info
    cell = kmf.cell
    kpts = kmf.kpts
    nks = np.array(nks)
    nocc = cell.tot_electrons() // 2
    nkpts = np.prod(nks)

    Lvec_real = kmf.cell.lattice_vectors()
    NsCell = kmf.cell.mesh
    L_delta = Lvec_real / NsCell[:, None]
    dvol = np.abs(np.linalg.det(L_delta))

    xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
    mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
    rptGrid3D = mesh_idx @ L_delta

    qGrid = minimum_image(cell, kpts - kpts[0, :])
    kGrid = minimum_image(cell, kpts)

    #Extract vacuum size information
    vac_size = np.linalg.norm(Lvec_real[2])
    nbands = nocc
    nG = np.prod(NsCell)

    #   Step 1.4: compute the pair product
    # Sample Unit cell in Reciprocal Space
    Lvec_recip = cell.reciprocal_vectors()
    Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
    Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
    Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
    Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
    GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip


    #Pick out unique Gz in L^*
    z_tmp = GptGrid3D[:,2]
    z_coords = np.unique(z_tmp)
    z_size = len(z_coords)
    z_coord_arr = np.zeros((z_size,3))
    z_coord_arr[:,2] = z_coords

    vac_size_bz = np.linalg.norm(Lvec_recip[2])

    #Compute pair products via Fourier transform, define structure factor
    SqG = np.zeros((nkpts, nG), dtype=np.float64)
    print("MEM USAGE IS:", SqG.nbytes)
    for q in range(nkpts):
        for k in range(nkpts):
            temp_SqG_k = np.zeros(nG, dtype=np.float64)  # Temporary storage for sums over m, n for the current k and q

            kpt1 = kGrid[k, :]
            qpt = qGrid[q, :]
            kpt2 = kpt1 + qpt

            kpt2_BZ = minimum_image(kmf.cell, kpt2)
            idx_kpt2 = np.where(np.sum((kGrid - kpt2_BZ[None, :]) ** 2, axis=1) < 1e-8)[0]
            if len(idx_kpt2) != 1:
                raise TypeError("Cannot locate (k+q) in the kmesh.")
            idx_kpt2 = idx_kpt2[0]
            kGdiff = kpt2 - kpt2_BZ

            for n in range(nbands):
                for m in range(nbands):
                    u1 = uKpts[k, n, :]
                    u2 = np.squeeze(np.exp(-1j * (rptGrid3D @ np.reshape(kGdiff, (-1, 1))))) * uKpts[idx_kpt2, m, :]
                    rho12 = np.reshape(np.conj(u1) * u2, (NsCell[0], NsCell[1], NsCell[2]))
                    temp_fft = np.fft.fftn((rho12 * dvol))
                    temp_SqG_k += np.abs(temp_fft.reshape(-1)) ** 2

            SqG[q, :] += temp_SqG_k / nkpts

    #SqG = SqG - nocc  # remove the zero order approximate nocc (turned off)

    #Attach pre-constant immediately
    vol_Bz = abs(np.linalg.det(Lvec_recip))
    area_Bz = np.linalg.norm(np.cross(Lvec_recip[0], Lvec_recip[1]))
    SqG = SqG * 1/(8*np.pi**3 * vol_Bz)
    SqG = SqG * area_Bz

    #   Step 3.1: define the local domain as multiple of BZ
    #LsCell_bz_local = N_local * Lvec_recip
    LsCell_bz_local = [N_local * Lvec_recip[0], N_local * Lvec_recip[1], Lvec_recip[2]]
    LsCell_bz_local_norms = np.linalg.norm(LsCell_bz_local, axis=1)

    #   localizer for the local domain
    r1 = np.min(LsCell_bz_local_norms[0:2]) / 2
    r1 = r1_prefactor*r1
    H = lambda q: localizer(q,r1)

    #   reciprocal lattice within the local domain
    #   Needs modification for 2D
    Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
    Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, [0] , indexing='ij')
    GptGrid3D_local = np.hstack(
        (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip

    # Make combined localizer grid
    GptGridz_localizer = GptGrid3D_local + z_coord_arr[0,:]
    for i in range(1,len(z_coords)):
        tmp = GptGrid3D_local + z_coord_arr[i,:]
        GptGridz_localizer = np.vstack((GptGridz_localizer, tmp))


    # SJQ Manually load SQG and all GptGrid3D lol
    load_from_mat = False
    if load_from_mat:
        SqG = scipy.io.loadmat("SqG.mat")["SqG"]
        GptGrid3D_local = scipy.io.loadmat("all_grids.mat")["GptGrid_Localizer"]
        GptGrid3D = scipy.io.loadmat("all_grids.mat")["GptGrid_UnitCell"]

    #   location/index of GptGrid3D_local within 'GptGrid3D'
    idx_GptGridz_local = []
    for Gl in GptGridz_localizer:
        idx_tmp = np.where(np.linalg.norm(Gl[None, :] - GptGrid3D, axis=1) < 1e-8)[0]
        if len(idx_tmp) != 1:
            raise TypeError("Cannot locate local G vector in the reciprocal lattice.")
        else:
            idx_GptGridz_local.append(idx_tmp[0])
    idx_GptGridz_local = np.array(idx_GptGridz_local)

    #   focus on S(q + G) with q in qGrid and G in GptGrid3D_local
    SqG_local = SqG[:, idx_GptGridz_local]

    #   Step 3.2: compute the Fourier transform of 1/|q|^2
    nqG_local = [N_local * nks[0], N_local * nks[1], 1]  # lattice size along each dimension in the real-space (equal to q + G size)
    Lvec_real_local = [Lvec_real[0]/N_local, Lvec_real[1]/N_local, Lvec_real[2]]  # dual real cell of local domain LsCell_bz_local
    Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
    Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
    Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
    Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
    RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local
    # SJQ Manually load all RptGrid3D lol

    if load_from_mat:
        RptGrid3D_local = scipy.io.loadmat("all_grids.mat")["RptGrid_Fourier"]

    from scipy.integrate import quad
    from scipy.special import i0
    normR = np.linalg.norm(RptGrid3D_local, axis=1)
    #np.pi * vac_size if x ==0 else
    def kernel_func(R, Gz, vac_size, a,b):
        func = lambda x: 8*np.pi* np.pi * x * (1- np.cos(vac_size/2 * Gz)* np.exp(-vac_size/2 * x)) * iv(0,-1j * x * R) / (x**2 + Gz**2)
        integral = quad(func, a, b, limit = 50)[0]
        return integral

    #trunc_int = [kernel_func(R, vac_size, 0, r1) for R in normR]

    CoulG = np.array([kernel_func(R, z_coords[0], vac_size, 0, r1) for R in normR]).reshape(-1,1)
    for Gz in z_coords[1:]:
        Coul_Gz = np.array([kernel_func(R, Gz, vac_size, 0, r1) for R in normR]).reshape(-1,1)
        CoulG = np.hstack((CoulG, Coul_Gz))

    # Exact expression when |R| = 0
    #CoulR[normR < 1e-8] = 8*np.pi**2 * (np.log(vac_size * r1) + gamma(0,r1 * vac_size) * gammaincc(0, r1 * vac_size) + np.euler_gamma)

    #   Step 4: Compute the correction

    ss_correction = 0
    #   Integral with Fourier Approximation
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGridz_localizer
        qGxy = qpt[None, :] + GptGrid3D_local
        exp_mat = np.exp(1j * (qGxy @ RptGrid3D_local.T))
        tmp = (exp_mat @ CoulG) / (N_local ** 2)
        tmp = tmp.reshape(-1, 1, order='F')
        tmp = tmp.flatten()
        prod = SqG_local[iq, :].T * H(qG) * tmp
        ss_correction -= np.real(np.sum(prod)) / nkpts

    int_term = ss_correction
    if debug:
        nqG_local = N_local**2*nkpts
        qG_full = np.zeros([nqG_local,3])
        HqG_local_full = np.zeros([nqG_local])
        SqG_local_full = np.zeros([nqG_local])
        VqG_local_full = np.zeros([nqG_local])
    #   Quadrature with Coulomb kernel
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGridz_localizer
        qG_no_z = qG[:, 0:2]
        tmp = SqG_local[iq, :].T * H(qG) / np.sum(qG ** 2,axis=1)  # (1 - np.exp(-vac_size/2 * np.sum(qG **2, axis =1))) * np.cos(vac_size/2 * qG[2])
        coul = np.exp(-vac_size / 2 * np.sqrt(np.sum(qG_no_z ** 2, axis=1))) * np.cos(vac_size / 2 * qG[:, 2])
        prod = tmp * (1 - coul) * 4 * np.pi
        qG0 = np.all(qG == 0, axis=1)
        indices = np.where(qG0)[0]
        if indices.size > 0:
            prod[indices] = SqG_local[iq, indices] * -np.pi * vac_size ** 2 / 2
        ss_correction += np.sum(prod) / nkpts * area_Bz
        if debug:
            qGz0 =qG[qG[:,2]==0]
            SqGz0 = SqG_local[iq, :].T[qG[:, 2] == 0]

            qG_full[iq*N_local**2:(iq+1)*N_local**2] = qGz0
            SqG_local_full[iq*N_local**2:(iq+1)*N_local**2]=SqGz0
            HqG_local_full[iq*N_local**2:(iq+1)*N_local**2]=H(qGz0)
            VqG_local_full[iq*N_local**2:(iq+1)*N_local**2]=(1 - coul[qG[:,2]==0])/ np.sum(qGz0 ** 2,axis=1)
    if debug:
        print('Saving qG mat files requested')
        scipy.io.savemat('qG_full_nk'+str(nks[0])+str(nks[1])+'1.mat', {"qG_full":qG_full})
        scipy.io.savemat('HqG_local_full_nk'+str(nks[0])+str(nks[1])+'1.mat', {"HqG_local_full":HqG_local_full})
        scipy.io.savemat('VqG_local_full_nk'+str(nks[0])+str(nks[1])+'1.mat', {"VqG_local_full":VqG_local_full})
        scipy.io.savemat('SqG_local_full_nk'+str(nks[0])+str(nks[1])+'1.mat', {"SqG_local_full":SqG_local_full})


    #ss_correction = 4 * np.pi * ss_correction/ np.linalg.det(Lvec_real)/np.linalg.norm(Lvec_recip[2])  # Coulomb kernel = 4 pi / |q|^2

    quad_term = ss_correction-int_term
    ss_correction =  ss_correction *vac_size_bz**2
    quad_term = quad_term *vac_size_bz**2
    int_term = int_term * vac_size_bz ** 2
    #   Step 5: apply the correction
    e_ex_ss = np.real(ex + ss_correction)

    #   Step 6: Lin's new idea
    # e_ex_ss2 = 0
    # #   Integral with Fourier Approximation
    # for iq, qpt in enumerate(qGrid):
    #     qG = qpt[None, :] + GptGrid3D_local
    #     exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
    #     tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
    #     tmp = (SqG_local[iq, :].T + nocc) * tmp
    #     e_ex_ss2 += np.real(np.sum(tmp)) * bz_dvol
    # e_ex_ss2 = prefactor_ex * 4 * np.pi * e_ex_ss2

    return e_ex_ss, int_term, quad_term



def make_ss_inputs(kmf,kpts,dm_kpts, mo_coeff_kpts,shiftFac=np.zeros(3)):
    from pyscf.pbc.tools import madelung,get_monkhorst_pack_size
    Madelung = madelung(kmf.cell, kpts)
    nocc = kmf.cell.tot_electrons() // 2
    nk = get_monkhorst_pack_size(kmf.cell, kpts)
    Nk = np.prod(nk)
    # dm_kpts = kmf.make_rdm1() ## make input
    _, K = kmf.get_jk(cell=kmf.cell, dm_kpts=dm_kpts, kpts=kpts, kpts_band=kpts)
    E_standard = -1. / Nk * np.einsum('kij,kji', dm_kpts, K) * 0.5
    E_standard /= 2
    E_madelung = E_standard - nocc * Madelung
    print(E_madelung)

    # Saving the wavefunction data (Strange MKL error just feeding mo_coeff...)
    # mo_coeff_kpts = kmf.mo_coeff_kpts # make input as well
    Lvec_real = kmf.cell.lattice_vectors()
    NsCell = kmf.cell.mesh
    L_delta = Lvec_real / NsCell[:, None]
    dvol = np.abs(np.linalg.det(L_delta))
    xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
    mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
    rptGrid3D = mesh_idx @ L_delta
    aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kpts)
    
    kshift_abs = mf.cell.reciprocal_vectors()*shiftFac / nk

    qGrid = minimum_image(kmf.cell, kshift_abs - kpts)
    kGrid = minimum_image(kmf.cell, kpts)

    nbands = nocc
    nG = np.prod(NsCell)
    uKpts = np.zeros((Nk, nbands, nG), dtype=complex)
    for k in range(Nk):
        for n in range(nbands):
            utmp = aoval[k] @ np.reshape(mo_coeff_kpts[k][:, n], (-1, 1))
            exp_part = np.exp(-1j * (rptGrid3D @ np.reshape(kGrid[k], (-1, 1))))
            uKpts[k, n, :] = np.squeeze(exp_part * utmp)
    return np.real(E_standard), np.real(E_madelung), uKpts, qGrid, kGrid

if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 3
    cell.mesh = [11] * 3
    cell.verbose = 5
    cell.build()
    mf = KRHF(cell, [2,1,1])
    mf.kernel()
    mf.analyze()
