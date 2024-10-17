import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools
from pyscf.lib import logger
import copy


# def subsample_kpts(mf, dim, div_vector, dm_kpts=None, khf_routine="standard", df_type=None, exxdiv='ewald',
#                    wrap_around=False, ss_nlocal=7, ss_localizer=None, ss_debug=False,ss_r1_prefactor=1.0,
#                    ss_subtract_nocc=False,ss_use_sqG_anisotropy=False,ss_nufft_gl=False,ss_n_fft=400):
def subsample_kpts(mf, dim, div_vector, dm_kpts=None, mo_coeff_kpts=None, khf_routine="standard", df_type=None, exxdiv='ewald',
                   wrap_around=False, sanity_run=False, ss_params=None):
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
    assert nk % (np.prod(div_vector) ** dim) == 0, "Div vector must divide nk"
    assert dim == mf.cell.dimension, "Dimension must match cell dimension"
    # assert(nk / (np.prod(div_vector) ** dim) !=0, "Div vector has more divisions than what is possible.")

    # Sanity run


    if mf.cell.output is not None:
        f = open(mf.cell.output, "a")
    else:
        f = None
    print('Recomputing jk', file=f)
    print('Sampling ', nk, 'k-points', file=f)
    if mo_coeff_kpts is None:
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

    stagger_routine_to_type= {
        "stagger_nonscf": "Non-SCF",
        "stagger": "Regular",
        "stagger_splitscf" : "Split-SCF",
        "stagger_nonscf_fourier": "Non-SCF",

    }
    khf_routines_stagger = [
        "stagger_nonscf",
        "stagger_splitscf",
        "stagger",
        "singularity_subtraction",
        "fourier",
        "stagger_nonscf_fourier",
    ]
    khf_routines_ss = [
        "singularity_subtraction",
        "fourier",
    ]
    khf_routines_all = [
        "standard",
    ]

    khf_routines_all.extend(khf_routines_ss)
    khf_routines_all.extend(khf_routines_stagger)

    if khf_routine not in khf_routines_all:
        raise ValueError(f'khf_routine must be one of {khf_routines_all}')
    else:
        print('khf_routine = ', khf_routine, file=f)

    if khf_routine in khf_routines_stagger:
        print('Warning, no J term computed', file=f)

    if khf_routine in khf_routines_ss:
        assert(ss_params)
        # Unpack params
        ss_localizer = ss_params['localizer']
        ss_localizer_M = lambda q, r1: ss_localizer(q, r1, M)
        ss_nlocal = ss_params.get('nlocal', 3)
        ss_r1_prefactor = ss_params.get('r1_prefactor', 1.0)
        ss_H_use_unscaled = ss_params.get('H_use_unscaled', False)  
        ss_SqG_filenames = ss_params.get('SqG_filenames', [None]*len(div_vector))
        M = np.array([1,1,1])

        if ss_params['use_sqG_anisotropy']:
            assert (khf_routine in khf_routines_ss)
            assert (mf.cell.dimension == 3)
            from pyscf.pbc.scf.khf import compute_SqG_anisotropy
            if 'M' in ss_params.keys():
                M = ss_params['M']
            else:
                print('Computing SqG anisotropy', file=f)
                M = compute_SqG_anisotropy(cell=mf.cell, nk=nks, N_local=7)

    # for div in div_vector:
    start_ind = 0
    if sanity_run:
        start_ind = -1
    k=0
    for j in range(start_ind, len(div_vector)):
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

        if khf_routine in khf_routines_ss:
            from pyscf.pbc.scf.khf import make_ss_inputs, khf_ss_2d, khf_ss_3d
            mf.kpts = kpts_div
            mf.exxdiv = None  #so that standard energy is computed without madelung
            E_standard, E_madelung, uKpts, qGrid, kGrid = make_ss_inputs(kmf=mf, kpts=kpts_div, dm_kpts=dm_kpts,
                                                           mo_coeff_kpts=mo_coeff_kpts)

            fourier_only = (khf_routine == "fourier")

            from pyscf.pbc.scf.khf import closest_fbz_distance
            r1, (b1,b2) = closest_fbz_distance(mf.cell.reciprocal_vectors(),ss_nlocal)

            # Find normalized perpendicular vector to closest plane vectors
            reciprocal_vectors = mf.cell.reciprocal_vectors()
            normal_vector = np.cross(reciprocal_vectors[b1], reciprocal_vectors[b2])
            normal_vector /= np.linalg.norm(normal_vector)

            # Find anisotropy in the direction of normal vector components
            M_normal = np.dot(M, normal_vector)

            # M = np.array([1,1,1])

            if ss_params['r1_prefactor'] == "precompute":
                if ss_H_use_unscaled:
                    from pyscf.pbc.scf.khf import precompute_r1_prefactor
                    gamma = 1e-4
                    delta = 0.5
                    power_law_exponent = -1
                    print('Using power law exponent {0} for r1_prefactor '.format(power_law_exponent), file=f,flush=True)
                    nk_1d = nks[0]
                    normal_vector = np.array([1,0,0])
                    M = np.array([1,1,1])
                    r1 = ss_nlocal/2. # now working in the basis of reciprocal lattice vectors
                    ss_r1_prefactor = precompute_r1_prefactor(power_law_exponent,nk_1d,delta,gamma,M,r1,normal_vector)
                    print('Precomputed r1_prefactor = ', ss_r1_prefactor, file=f,flush=True)
                else:
                    from pyscf.pbc.scf.khf import precompute_r1_prefactor
                    gamma = 1e-4
                    delta = 0.5
                    power_law_exponent = -1
                    print('Using power law exponent {0} for r1_prefactor '.format(power_law_exponent), file=f,flush=True)
                    nk_1d = nks[0]
                    ss_r1_prefactor = precompute_r1_prefactor(power_law_exponent,nk_1d,delta,gamma,M,r1,normal_vector)
                    print('Precomputed r1_prefactor = ', ss_r1_prefactor, file=f,flush=True)
                    # # Override
                    # ss_params['r1_prefactor'] = ss_r1_prefactor

            if mf.cell.dimension ==3:
                e_ss, ex_ss_2, int_term, quad_term = khf_ss_3d(mf, nks, uKpts, E_standard, E_madelung, 
                                                               N_local=ss_params['nlocal'], debug=ss_params['debug'],
                                                               localizer=ss_localizer_M, r1_prefactor=ss_r1_prefactor, 
                                                               fourier_only=fourier_only, subtract_nocc=ss_params['subtract_nocc'], 
                                                               nufft_gl=ss_params['nufft_gl'], n_fft=ss_params['n_fft'],
                                                               vhR_symm=ss_params['vhR_symm'],H_use_unscaled=ss_H_use_unscaled,
                                                               SqG_filename=ss_SqG_filenames[k])

                results["Ek_ss_2_list"].append(ex_ss_2)

            elif mf.cell.dimension ==2:
                e_ss, int_term, quad_term = khf_ss_2d(mf, nks, uKpts, E_standard, N_local=ss_params['nlocal'], debug=ss_params['debug'],
                                                localizer=ss_localizer_M, r1_prefactor=ss_params['r1_prefactor'],
                                                subtract_nocc=ss_params['subtract_nocc'])

            print('Ek (Madelung) (a.u.) = ', E_madelung, file=f)
            print('Ek (SS) (a.u.) = ',   e_ss, file=f)

            results["Ek_ss_list"].append(e_ss)
            results["Ek_list"].append(E_madelung)
            results["nk_list"].append(nk_div)
            results["nks_list"].append(copy.copy(nks))
            results["int_terms"].append(int_term)
            results["quad_terms"].append(quad_term)
            results["Ek_uncorr_list"].append(E_standard)
        elif khf_routine in khf_routines_stagger:
            from pyscf.pbc.scf.khf import khf_stagger
            stagger_type = stagger_routine_to_type[khf_routine]
            fourinterp = (khf_routine == "stagger_nonscf_fourier")
            Ek_stagger_M, Ek_stagger, Ek_madelung = khf_stagger(icell=mf.cell, ikpts=kpts_div, version=stagger_type,
                                                                df_type=df_type, dm_kpts=dm_kpts,
                                                                mo_coeff_kpts=mo_coeff_kpts, fourinterp=fourinterp)

            print('Ek (a.u.) = ', Ek_stagger_M, file=f)
            results["Ek_stagger_list"].append(Ek_stagger_M)
            results["Ek_list"].append(Ek_madelung)
            results["nk_list"].append(nk_div)
            results["nks_list"].append(copy.copy(nks))

        else: # standard exchange

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
        k = k + 1

    print('=== Kpoint Subsampling Results === ')
    for key, value in results.items():
        print('\n' + key)
        print(value)
    return results
