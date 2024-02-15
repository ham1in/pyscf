
import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools
from pyscf.lib import logger
import copy

def subsample_kpts(mf, dim, div_vector, dm_kpts = None, stagger_type = None, df_type = None, singularity_subtraction=False, exxdiv ='ewald'):
    nks = pbc_tools.get_monkhorst_pack_size(cell=mf.cell,kpts = mf.kpts)
    nk = np.prod(nks)
    assert(nk % (np.prod(div_vector)**dim) == 0, "Div vector must divide nk")

    # Sanity run
    if mf.cell.output is not None:
        f = open(mf.cell.output, "a")
    else:
        f = None
    print('Recomputing jk', file=f)
    print('Sampling ', nk, 'k-points',file=f)
    mo_coeff_kpts = np.array(mf.mo_coeff_kpts)

    if dm_kpts is None:
        dm_kpts = mf.make_rdm1()
    mf.exxdiv = exxdiv
    J, K = mf.get_jk(cell = mf.cell, dm_kpts = dm_kpts, kpts = mf.kpts, kpts_band = mf.kpts, with_j = True)

    Ek = -1. / nk * np.einsum('kij,kji', dm_kpts, K) * 0.5
    Ek /= 2.

    Ej = 1. / nk * np.einsum('kij,kji', dm_kpts, J)
    Ej /= 2.

    kpts_div_old = mf.cell.make_kpts(nks, wrap_around=True)

    print('Ej (a.u.) = ', Ej, file=f)
    print('Ek (a.u.) = ', Ek, file=f)

    Ek_list = []
    Ej_list = []

    nk_list = []
    nks_list = []

    if stagger_type is not None:
        print('Warning, no J term computed', file=f)

    # for div in div_vector:
    for j in range(-1,len(div_vector)):
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

        kpts_div = mf.cell.make_kpts(nks, wrap_around=True)

        subsample_indices = []
        for ik in range(nk_div):
            diff_mat = kpts_div_old - kpts_div[ik]
            diff_norm = np.einsum('ij,ij->i', diff_mat, diff_mat)
            diff0 = np.where(diff_norm == 0)
            assert(len(diff0) == 1)
            assert(len(diff0[0]) == 1)
            diff0 = diff0[0][0]
            subsample_indices.append(diff0)


        dm_kpts = dm_kpts[subsample_indices]
        mo_coeff_kpts = mo_coeff_kpts[subsample_indices]

        if singularity_subtraction:
            from pyscf.pbc.scf.khf import make_ss_inputs, khf_2d
            mf.kpts = kpts_div
            mf.exxdiv = None #so that standard energy is computed without madelung
            E_standard, E_madelung, uKpts = make_ss_inputs(kmf=mf, kpts=kpts_div, dm_kpts=dm_kpts,
                                                           mo_coeff_kpts=mo_coeff_kpts)
            e_ss = khf_2d(mf, nks, uKpts, E_madelung, N_local=15,localizer_degree=6)
            print('Ek (standard) (a.u.) = ', E_madelung, file=f)
            print('Ek (SS) (a.u.) = ', e_ss, file=f)

            Ek_list.append(e_ss)
            nk_list.append(nk_div)
            nks_list.append(copy.copy(nks))
        elif stagger_type !=None:
            from pyscf.pbc.scf.khf import khf_stagger

            Ek_stagger_M, Ek_stagger, Ek_standard = khf_stagger(icell=mf.cell, ikpts=kpts_div, version=stagger_type, df_type=df_type, dm_kpts=dm_kpts)



            kpts_div_old = kpts_div


            print('Ek (a.u.) = ', Ek_stagger_M, file=f)
            Ek_list.append(Ek_stagger_M)
            nk_list.append(nk_div)
            nks_list.append(copy.copy(nks))

        else:

            J, K = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=kpts_div, kpts_band=kpts_div, with_j=True)
            Ek = -1. / nk_div * np.einsum('kij,kji', dm_kpts, K) * 0.5
            Ek /= 2.

            Ej = 1. / nk_div * np.einsum('kij,kji', dm_kpts, J)
            Ej /= 2.

            kpts_div_old  = kpts_div

            Ek = Ek.real
            Ej = Ej.real

            print('Ej (a.u.) = ', Ej, file = f)
            print('Ek (a.u.) = ', Ek, file = f)
            Ej_list.append(Ej)
            Ek_list.append(Ek)
            nk_list.append(nk_div)
            nks_list.append(copy.copy(nks))





    return nk_list, nks_list, Ej_list, Ek_list



