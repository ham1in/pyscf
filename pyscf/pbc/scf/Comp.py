import numpy as np
from scipy.special import sici
from pyscf.pbc.tools import madelung


#   Minimum images of k-points to the first BZ
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


#   polynomial localizer
def poly_localizer(x, r1, d):
    x = np.asarray(x)
    x = x / r1
    r = np.linalg.norm(x, axis=1) if x.ndim > 1 else np.linalg.norm(x)
    val = (1 - r ** d) ** d
    if x.ndim > 1:
        val[r > 1] = 0
    elif r > 1:
        val = 0
    return val


def khf_exchange_ss(kmf, nks, mo_coeff, made, N_local=5):
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
    mo_coeff_kpts = mo_coeff
    Lvec_real = kmf.cell.lattice_vectors()
    NsCell = kmf.cell.mesh
    L_delta = Lvec_real / NsCell[:, None]
    dvol = np.abs(np.linalg.det(L_delta))
    #Evaluate wavefunction on all real space grid points
    # # Establishing real space grid (Generalized for arbitary volume defined by 3 vectors)
    xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
    mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
    rptGrid3D = mesh_idx @ L_delta
    aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kmf.kpts)

    #   Step 1.2: map q-mesh and k-mesh to BZ
    qGrid = minimum_image(cell, kpts - kpts[0, :])
    kGrid = minimum_image(cell, kpts)

    #   Step 1.3: evaluate MO periodic component on a real fine mesh in unit cell
    nbands = nocc
    nG = np.prod(NsCell)
    uKpts = np.zeros((nkpts, nbands, nG), dtype=complex)
    for k in range(nkpts):
        for n in range(nbands):
            #   mo_coeff_kpts is of dimension (nkpts, nbasis, nband)
            utmp = aoval[k] @ np.reshape(mo_coeff_kpts[k][:, n], (-1, 1))
            exp_part = np.exp(-1j * (rptGrid3D @ np.reshape(kGrid[k], (-1, 1))))
            uKpts[k, n, :] = np.squeeze(exp_part * utmp)

            #   Step 1.4: compute the pair product
    Lvec_recip = cell.reciprocal_vectors()
    Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
    Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
    Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
    Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
    GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip

    rhokqmnG = np.zeros((nkpts, nkpts, nbands, nbands, nG), dtype=complex)
    for k in range(nkpts):
        for q in range(nkpts):
            kpt1 = kGrid[k, :]
            qpt = qGrid[q, :]
            kpt2 = kpt1 + qpt

            #   locate uk with k = kpt2
            kpt2_BZ = minimum_image(kmf.cell, kpt2)
            idx_kpt2 = np.where(np.sum((kGrid - kpt2_BZ[None, :]) ** 2, axis=1) < 1e-8)[0]
            if len(idx_kpt2) != 1:
                raise TypeError("Cannot locate (k+q) in the kmesh.")
            else:
                idx_kpt2 = idx_kpt2[0]
            kGdiff = kpt2 - kpt2_BZ

            #   compute rho_{nk1, m(k1+q)}(G)}
            for n in range(nbands):
                for m in range(nbands):
                    u1 = uKpts[k, n, :]
                    u2 = np.squeeze(np.exp(-1j * (rptGrid3D @ np.reshape(kGdiff, (-1, 1))))) * uKpts[idx_kpt2, m, :]
                    rho12 = np.reshape(np.conj(u1) * u2, (NsCell[0], NsCell[1], NsCell[2]))
                    temp_fft = np.fft.fftn((rho12 * dvol))
                    rhokqmnG[k, q, n, m, :] = temp_fft.reshape(-1)

    #   Step 2: Construct the structure factor
    SqG = np.sum(np.abs(rhokqmnG) ** 2, axis=(0, 2, 3)) / nkpts
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
    H = lambda q: poly_localizer(q, r1, 6)

    #   reciprocal lattice within the local domain
    Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
    Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, Grid_1D, indexing='ij')
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
    #   Quadrature with Coulomb kernel
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGrid3D_local
        tmp = SqG_local[iq, :].T * H(qG) / np.sum(qG ** 2, axis=1)
        tmp[np.isinf(tmp) | np.isnan(tmp)] = 0
        ss_correction -= np.sum(tmp) * bz_dvol

    #   Integral with Fourier Approximation
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGrid3D_local
        exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
        tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
        tmp = SqG_local[iq, :].T * H(qG) * tmp
        ss_correction += np.real(np.sum(tmp)) * bz_dvol

    ss_correction = 4 * np.pi * ss_correction  # Coulomb kernel = 4 pi / |q|^2

    #   Step 5: apply the correction
    e_ex_ss = made + prefactor_ex * ss_correction

    #   Step 6: Lin's new idea
    e_ex_ss2 = 0
    #   Integral with Fourier Approximation
    for iq, qpt in enumerate(qGrid):
        qG = qpt[None, :] + GptGrid3D_local
        exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
        tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
        tmp = (SqG_local[iq, :].T + nocc) * tmp
        e_ex_ss2 += np.real(np.sum(tmp)) * bz_dvol
    e_ex_ss2 = prefactor_ex * 4 * np.pi * e_ex_ss2

    return e_ex_ss, e_ex_ss2






