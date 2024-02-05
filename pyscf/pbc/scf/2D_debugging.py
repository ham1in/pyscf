from scipy.integrate import quad
from scipy.special import iv
import numpy as np
import scipy.io
#normR = [0,0.1111,0.2222,0.3333,0.4444,0.5556,0.6667,0.7778,0.8889,1,0.8889,0.7778,0.6667,0.5556,0.4444,0.3333,0.2222,0.1111,0.1111,0.1571,0.2485,0.3514,0.4581,0.5666,0.6759,0.7857,0.8958,1.0062,0.8958,0.7857,0.6759,0.5666,0.4581,0.3514,0.2485,0.1571,0.2222,0.2485,0.3143,0.4006,0.4969,0.5984,0.7027,0.8089,0.9162,1.0244,0.9162,0.8089,0.7027,0.5984,0.4969,0.4006,0.3143,0.2485,0.3333,0.3514,0.4006,0.4714,0.5556,0.6479,0.7454,0.8462,0.9493,1.0541,0.9493,0.8462,0.7454,0.6479,0.5556,0.4714,0.4006,0.3514,0.4444,0.4581,0.4969,0.5556,0.6285,0.7115,0.8012,0.8958,0.9938,1.0943,0.9938,0.8958,0.8012,0.7115,0.6285,0.5556,0.4969,0.4581,0.5556,0.5666,0.5984,0.6479,0.7115,0.7857,0.8678,0.9558,1.0482,1.144,1.0482,0.9558,0.8678,0.7857,0.7115,0.6479,0.5984,0.5666,0.6667,0.6759,0.7027,0.7454,0.8012,0.8678,0.9428,1.0244,1.1111,1.2019,1.1111,1.0244,0.9428,0.8678,0.8012,0.7454,0.7027,0.6759,0.7778,0.7857,0.8089,0.8462,0.8958,0.9558,1.0244,1.0999,1.1811,1.2669,1.1811,1.0999,1.0244,0.9558,0.8958,0.8462,0.8089,0.7857,0.8889,0.8958,0.9162,0.9493,0.9938,1.0482,1.1111,1.1811,1.2571,1.338,1.2571,1.1811,1.1111,1.0482,0.9938,0.9493,0.9162,0.8958,1,1.0062,1.0244,1.0541,1.0943,1.144,1.2019,1.2669,1.338,1.4142,1.338,1.2669,1.2019,1.144,1.0943,1.0541,1.0244,1.0062,0.8889,0.8958,0.9162,0.9493,0.9938,1.0482,1.1111,1.1811,1.2571,1.338,1.2571,1.1811,1.1111,1.0482,0.9938,0.9493,0.9162,0.8958,0.7778,0.7857,0.8089,0.8462,0.8958,0.9558,1.0244,1.0999,1.1811,1.2669,1.1811,1.0999,1.0244,0.9558,0.8958,0.8462,0.8089,0.7857,0.6667,0.6759,0.7027,0.7454,0.8012,0.8678,0.9428,1.0244,1.1111,1.2019,1.1111,1.0244,0.9428,0.8678,0.8012,0.7454,0.7027,0.6759,0.5556,0.5666,0.5984,0.6479,0.7115,0.7857,0.8678,0.9558,1.0482,1.144,1.0482,0.9558,0.8678,0.7857,0.7115,0.6479,0.5984,0.5666,0.4444,0.4581,0.4969,0.5556,0.6285,0.7115,0.8012,0.8958,0.9938,1.0943,0.9938,0.8958,0.8012,0.7115,0.6285,0.5556,0.4969,0.4581,0.3333,0.3514,0.4006,0.4714,0.5556,0.6479,0.7454,0.8462,0.9493,1.0541,0.9493,0.8462,0.7454,0.6479,0.5556,0.4714,0.4006,0.3514,0.2222,0.2485,0.3143,0.4006,0.4969,0.5984,0.7027,0.8089,0.9162,1.0244,0.9162,0.8089,0.7027,0.5984,0.4969,0.4006,0.3143,0.2485,0.1111,0.1571,0.2485,0.3514,0.4581,0.5666,0.6759,0.7857,0.8958,1.0062,0.8958,0.7857,0.6759,0.5666,0.4581,0.3514,0.2485,0.1571]
normR = [0]

u_data = scipy.io.loadmat('ss221.mat')
uKpts = u_data['uKpt']

k_data = scipy.io.loadmat('kpt221.mat')
kpts = k_data['kptGrid3D']

from scipy.special import iv


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


#   basic info
Lvec_real = np.array([[1,0,0], [0,1,0], [0,0,6]])
Lvec_recip = np.array([[2*np.pi,0,0],[0,(2*np.pi),0],[0,0,(2*np.pi)/6]])
nkpts = 4
nocc = 1

NsCell = [10,10,60]
L_delta = [[0.1,0,0],[0,0.1,0],[0,0,0.1]]
dvol = np.abs(np.linalg.det(L_delta))

xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
rptGrid3D = mesh_idx @ L_delta
#   Step 2: Construct the structure factor
SqDat = scipy.io.loadmat('SqG221.mat')
SqG = SqDat['SqG']

Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip

cell_bz_1 = np.concatenate((np.arange(0, NsCell[0] // 2 + 1), np.arange(-NsCell[0] // 2 + 1, 0)))
cell_bz_2 = np.concatenate((np.arange(0, NsCell[1] // 2 + 1), np.arange(-NsCell[1] // 2 + 1, 0)))
cell_bz_3 = np.concatenate((np.arange(0, NsCell[2] // 2 + 1), np.arange(-NsCell[2] // 2 + 1, 0)))
Zbz, Ybz, Xbz = np.meshgrid(cell_bz_3,cell_bz_2,cell_bz_1, indexing = 'ij')
GptGrid3D = (Xbz.flatten()[:, np.newaxis] * Lvec_recip[0] + Ybz.flatten()[:, np.newaxis] * Lvec_recip[1] + Zbz.flatten()[:, np.newaxis] * Lvec_recip[2])

N_local = 7
#   Exchange energy can be formulated as
#   Ex = prefactor_ex * bz_dvol * sum_{q} (\sum_G S(q+G) * 4*pi/|q+G|^2)
# prefactor_ex = -1 / (8 * np.pi ** 3)
# # Area
# bz_dvol = np.abs(np.linalg.det(Lvec_recip))/vac_size_bz / nkpts

#   Side Step: double check the validity of SqG by computing the exchange energy
# if False:
#     CoulG = np.zeros_like(SqG)
#     for iq, qpt in enumerate(qGrid):
#         qG = qpt[None, :] + GptGrid3D
#         norm2_qG = np.sum(qG ** 2, axis=1)
#         CoulG[iq, :] = 4 * np.pi / norm2_qG
#         CoulG[iq, norm2_qG < 1e-8] = 0
#     Ex = prefactor_ex * bz_dvol * np.sum((SqG + nocc) * CoulG)
#     print(f'Ex = {Ex}')

#   Step 3: construct Fourier Approximation of S(q+G)h(q+G)

#   Step 3.1: define the local domain as multiple of BZ
LsCell_bz_local = N_local * Lvec_recip
LsCell_bz_local_norms = np.linalg.norm(LsCell_bz_local, axis=1)
#   localizer for the local domain
r1 = np.min(LsCell_bz_local_norms[0:2]) / 2

H = lambda q: poly_localizer(q, r1, 4)
q_dat = scipy.io.loadmat('qGrid.mat')
qGrid = q_dat['qGrid']

#   reciprocal lattice within the local domain
#   Needs modification for 2D
# Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
# Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, [0], indexing='ij')
# GptGrid3D_local = np.hstack(
#     (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip
Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local + 1) // 2 + 1, 0)))
Yl, Xl, Zl = np.meshgrid(Grid_1D, Grid_1D, [0], indexing='ij')
GptGrid3D_local = (Zl.flatten()[:,np.newaxis] *Lvec_recip[2] + Yl.flatten()[:, np.newaxis] * Lvec_recip[1] + Xl.flatten()[:, np.newaxis] * Lvec_recip[0])

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
nqG_local = [N_local * 2, N_local * 2,
             1]  # lattice size along each dimension in the real-space (equal to q + G size)
Lvec_real_local = Lvec_real / N_local  # dual real cell of local domain LsCell_bz_local

Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local

from scipy.integrate import quad
from scipy.special import gammaincc
from scipy.special import gamma

normR = np.linalg.norm(RptGrid3D_local, axis=1)

def kernel_func(R, vac_size, a, b):
    func = lambda x: np.pi * vac_size if x == 0 else 2 * np.pi * (1 - np.exp(-vac_size / 2 * x)) / x * iv(0,
                                                                                                          -1j * x * R)
    integral = quad(func, a, b, limit=50)[0]
    return integral


trunc_int = [kernel_func(R, 6, 0, r1) for R in normR]

CoulR = trunc_int

# Exact expression when |R| = 0
# CoulR[normR < 1e-8] = 8*np.pi**2 * (np.log(vac_size * r1) + gamma(0,r1 * vac_size) * gammaincc(0, r1 * vac_size) + np.euler_gamma)

#   Step 4: Compute the correction

ss_correction = 0
#   Quadrature with Coulomb kernel
for iq, qpt in enumerate(qGrid):
    qG = qpt[None, :] + GptGrid3D_local
    tmp = SqG_local[iq, :].T * H(qG) / np.sum(qG ** 2,axis=1)
    coul = np.exp(-6 / 2 * np.sqrt(np.sum(qG ** 2, axis=1)))
    tmp = tmp *(1- coul * np.cos(6 / 2 * qG[:, 2]))
    tmp[np.isinf(tmp) | np.isnan(tmp)] = 0
    ss_correction += np.sum(tmp) / nkpts

#   Integral with Fourier Approximation
for iq, qpt in enumerate(qGrid):
    qG = qpt[None, :] + GptGrid3D_local
    exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
    tmp = (exp_mat @ CoulR) / (np.linalg.norm(np.cross(LsCell_bz_local[0], LsCell_bz_local[1])))
    tmp = SqG_local[iq, :].T * H(qG) * tmp
    ss_correction -= np.real(np.sum(tmp)) / nkpts

ss_correction = 4 * np.pi * ss_correction / np.linalg.det(Lvec_real) / np.linalg.norm(
    Lvec_recip[2])  # Coulomb kernel = 4 pi / |q|^2

print("CORRECTION IS:")
print(ss_correction)