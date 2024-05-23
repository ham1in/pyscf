import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools
from pyscf.lib import logger
import copy


def subsample_kpts(mf, dim, div_vector, dm_kpts=None, stagger_type=None, df_type=None, exxdiv='ewald',
                   wrap_around=False, singularity_subtraction=False, ss_nlocal=7, ss_localizer=None, ss_debug=False,
                   ss_r1_prefactor=1.0):
    """

    Args:
        mf: mean-field object
        dim: Dimension of kpoint subsampling routine
        div_vector: Array of integers to consecutively divide nk along 1 dimension. e.g. you can do [2,2,3] for nk_1d=24
        dm_kpts: Density matrix from mf.kernel
        stagger_type: Use subsampling with this version of staggered mesh
        df_type: Density fitting type (df.GDF, df.FFTDF, etc.)
        singularity_subtraction: Set to true to do subsampling with singularity subtraction
        exxdiv: Procedure for handling exchange divergence
        wrap_around: Set to true to keep kpoints within FBZ

    Returns:
        nk_list: Array of total nk for each subsampling iteration
        nks_list: Array of nk split into their dimensions
        Ej_list: Hartree term for each subsampling iteration
        Ek_list: Exchange term for each subsampling iteration
    """
    nks = pbc_tools.get_monkhorst_pack_size(cell=mf.cell, kpts=mf.kpts)
    nk = np.prod(nks)
    assert(nk % (np.prod(div_vector) ** dim) == 0, "Div vector must divide nk")
    assert(dim == mf.cell.dimension, "Dimension must match cell dimension")
    # Sanity run
    if mf.cell.output is not None:
        f = open(mf.cell.output, "a")
    else:
        f = None
    print('Recomputing jk', file=f)
    print('Sampling ', nk, 'k-points', file=f)
    mo_coeff_kpts = np.array(mf.mo_coeff_kpts)

    if dm_kpts is None:
        dm_kpts = mf.make_rdm1()
    mf.exxdiv = exxdiv
    J, K = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=mf.kpts, kpts_band=mf.kpts, with_j=True)

    Ek = -1. / nk * np.einsum('kij,kji', dm_kpts, K) * 0.5
    Ek /= 2.

    Ej = 1. / nk * np.einsum('kij,kji', dm_kpts, J)
    Ej /= 2.

    kpts_div_old = mf.cell.make_kpts(nks, wrap_around=wrap_around)

    print('Ej (a.u.) = ', Ej, file=f)
    print('Ek (a.u.) = ', Ek, file=f)

    results = {
        "Ek_list": [],
        "Ek_uncorr_list": [],
        "Ej_list": [],
        "nk_list": [],
        "nks_list": [],
        "Ek_stagger_list": [],
        "Ek_ss_list": [],
        "int_terms": [],
        "quad_terms": [],
        "Ek_ss_2_list": []
    }

    if stagger_type is not None:
        print('Warning, no J term computed', file=f)

    # for div in div_vector:
    for j in range(-1, len(div_vector)):
        if j == -1:
            div = 1
            for i in range(dim):
                nks[i] = nks[i] / div
            nk_div = np.prod(nks)
            print('Initial Sanity run. Dividing by ', div ** dim, ', subsampling ', nk_div, 'k-points', file=f)
        else:
            div = div_vector[j]
            for i in range(dim):
                nks[i] = nks[i] / div
            nk_div = np.prod(nks)
            print('Dividing by ', div ** dim, ', subsampling ', nk_div, 'k-points', file=f)

        kpts_div = mf.cell.make_kpts(nks, wrap_around=wrap_around)

        subsample_indices = []
        for ik in range(nk_div):
            diff_mat = kpts_div_old - kpts_div[ik]
            diff_norm = np.einsum('ij,ij->i', diff_mat, diff_mat)
            diff0 = np.where(diff_norm == 0)
            assert (len(diff0) == 1)
            assert (len(diff0[0]) == 1)
            diff0 = diff0[0][0]
            subsample_indices.append(diff0)

        dm_kpts = dm_kpts[subsample_indices]
        mo_coeff_kpts = mo_coeff_kpts[subsample_indices]

        if singularity_subtraction:
            from pyscf.pbc.scf.khf import make_ss_inputs, khf_ss_2d, khf_ss_3d
            mf.kpts = kpts_div
            mf.exxdiv = None  #so that standard energy is computed without madelung
            E_standard, E_madelung, uKpts, qGrid, kGrid = make_ss_inputs(kmf=mf, kpts=kpts_div, dm_kpts=dm_kpts,
                                                           mo_coeff_kpts=mo_coeff_kpts)
            if mf.cell.dimension ==3:
                e_ss, ex_ss_2, int_term, quad_term = khf_ss_3d(mf, nks, uKpts, E_madelung, N_local=ss_nlocal, debug=ss_debug,
                                                localizer=ss_localizer, r1_prefactor=ss_r1_prefactor)
                results["Ek_ss_2_list"].append(ex_ss_2)

            elif mf.cell.dimension ==2:
                e_ss, int_term, quad_term = khf_ss_2d(mf, nks, uKpts, E_standard, N_local=ss_nlocal, debug=ss_debug,
                                                localizer=ss_localizer, r1_prefactor=ss_r1_prefactor)
                
            print('Ek (Madelung) (a.u.) = ', E_madelung, file=f)
            print('Ek (SS) (a.u.) = ', e_ss, file=f)

            results["Ek_ss_list"].append(e_ss)
            results["Ek_list"].append(E_madelung)
            results["nk_list"].append(nk_div)
            results["nks_list"].append(copy.copy(nks))
            results["int_terms"].append(int_term)
            results["quad_terms"].append(quad_term)
            results["Ek_uncorr_list"].append(E_standard)
        elif stagger_type != None:
            from pyscf.pbc.scf.khf import khf_stagger

            Ek_stagger_M, Ek_stagger, Ek_standard = khf_stagger(icell=mf.cell, ikpts=kpts_div, version=stagger_type,
                                                                df_type=df_type, dm_kpts=dm_kpts)

            print('Ek (a.u.) = ', Ek_stagger_M, file=f)
            results["Ek_stagger_list"].append(Ek_stagger_M)
            results["Ek_list"].append(Ek_standard)
            results["nk_list"].append(nk_div)
            results["nks_list"].append(copy.copy(nks))

        else:

            J, K = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=kpts_div, kpts_band=kpts_div, with_j=True)
            Ek = -1. / nk_div * np.einsum('kij,kji', dm_kpts, K) * 0.5
            Ek /= 2.

            Ej = 1. / nk_div * np.einsum('kij,kji', dm_kpts, J)
            Ej /= 2.

            Ek = Ek.real
            Ej = Ej.real

            print('Ej (a.u.) = ', Ej, file=f)
            print('Ek (a.u.) = ', Ek, file=f)
            results["Ej_list"].append(Ej)
            results["Ek_list"].append(Ek)
            results["nk_list"].append(nk_div)
            results["nks_list"].append(copy.copy(nks))

        kpts_div_old = kpts_div

    print('=== Kpoint Subsampling Results === ')
    for key, value in results.items():
        print('\n' + key)
        print(value)
    return results
