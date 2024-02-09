
import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools
import copy

def subsample_kpts(mf, dim, div_vector):
    nks = pbc_tools.get_monkhorst_pack_size(cell=mf.cell,kpts = mf.kpts)
    nk = np.prod(nks)
    assert(nk % (np.prod(div_vector)**dim) == 0, "Div vector must divide nk")

    # Sanity run
    print('Initial sanity run. Ssampling ', nk, 'k-points',file=mf.cell.output)

    dm = mf.make_rdm1()
    J, K = mf.get_jk(cell = mf.cell, dm_kpts = dm, kpts = mf.kpts, kpts_band = mf.kpts, with_j = True)

    Ek = -1. / nk * np.einsum('kij,kji', dm, K) * 0.5
    Ek /= 2.

    Ej = 1. / nk * np.einsum('kij,kji', dm, J)
    Ej /= 2.

    kpts_div_old = mf.cell.make_kpts(nks, wrap_around=True)
    print('Ej (a.u.) = ', Ej, file=mf.cell.output)
    print('Ek (a.u.) = ', Ek, file=mf.cell.output)

    Ek_list = [Ek.real]
    Ej_list = [Ej.real]
    nk_list = [nk]
    nks_list = [copy.copy(nks)]
    for div in div_vector:

        for i in range(dim):
            nks[i] = nks[i]/div

        kpts_div = mf.cell.make_kpts(nks, wrap_around=True)
        nk_div = np.prod(nks)
        print('Dividing by ', div**dim, ', subsampling ',nk_div , 'k-points', file=mf.cell.output)
        subsample_indices = []
        for ik in range(nk_div):
            diff_mat = kpts_div_old - kpts_div[ik]
            diff_norm = np.einsum('ij,ij->i', diff_mat, diff_mat)
            diff0 = np.where(diff_norm == 0)
            assert(len(diff0) == 1)
            assert(len(diff0[0]) == 1)
            diff0 = diff0[0][0]
            subsample_indices.append(diff0)


        dm = dm[subsample_indices]
        J, K = mf.get_jk(cell=mf.cell, dm_kpts=dm, kpts=kpts_div, kpts_band=kpts_div, with_j=True)

        Ek = -1. / nk_div * np.einsum('kij,kji', dm, K) * 0.5
        Ek /= 2.

        Ej = 1. / nk_div * np.einsum('kij,kji', dm, J)
        Ej /= 2.

        kpts_div_old  = kpts_div


        Ek = Ek.real
        Ej = Ej.real
        print('Ej (a.u.) = ', Ej, file = mf.cell.output)
        print('Ek (a.u.) = ', Ek, file = mf.cell.output)
        Ej_list.append(Ej)
        Ek_list.append(Ek)
        nk_list.append(nk_div)
        nks_list.append(copy.copy(nks))




    return nk_list, nks_list, Ej_list, Ek_list



