from scipy.integrate import quad
from scipy.special import iv
import numpy as np
import scipy.io

u_data = scipy.io.loadmat('uKpt.mat')
uKpts = u_data['uKpt']
nks = [2,2,1]
k_data = scipy.io.loadmat('kptGrid3D.mat')
kGrid = k_data['kptGrid3D']
q_data = scipy.io.loadmat('qGrid3D.mat')
qGrid = q_data['qGrid']
N_local = 3
from scipy.special import iv


def poly_localizer(x, r1, d):
    x = np.asarray(x)
    x = x[:,:2] / r1
    r = np.linalg.norm(x, axis=1) if x.ndim > 1 else np.linalg.norm(x)
    val = (1 - r ** d) ** d
    if x.ndim > 1:
        val[r > 1] = 0
    elif r > 1:
        val = 0
    return val


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


# Extract vacuum size information
non_per_dim = np.where(nks == 1)
if len(non_per_dim) > 1:
    raise TypeError("More than one non-periodic direction found.")
non_per_dim = non_per_dim[0]
vac_size = np.linalg.norm(Lvec_real[2])

nbands = nocc
nG = np.prod(NsCell)

#   Step 1.4: compute the pair product
# Sample Unit cell in Reciprocal Space (Unchanged)
Ggrid = scipy.io.loadmat('Ggrid.mat')
GptGrid3D = Ggrid['GptGrid_UnitCell']

# Pick out unique Gz in L^*
z_tmp = GptGrid3D[:, 2]
z_coords = np.unique(z_tmp)
z_size = len(z_coords)
z_coord_arr = np.zeros((z_size, 3))
z_coord_arr[:, 2] = z_coords

vac_size_bz = np.linalg.norm(Lvec_recip[2])

SqG_dat = scipy.io.loadmat('SqG.mat')
SqG = SqG_dat['SqG']

# Attach pre-constant immediately
vol_Bz = abs(np.linalg.det(Lvec_recip))
area_Bz = np.linalg.norm(np.cross(Lvec_recip[0], Lvec_recip[1]))
SqG = SqG * 1 / (8 * np.pi ** 3 * vol_Bz)
SqG = SqG * area_Bz  # nkpts already in the sum above

#   Step 3.1: define the local domain as multiple of BZ
# LsCell_bz_local = N_local * Lvec_recip
LsCell_bz_local = [N_local * Lvec_recip[0], N_local * Lvec_recip[1], Lvec_recip[2]]
LsCell_bz_local_norms = np.linalg.norm(LsCell_bz_local, axis=1)

#   localizer for the local domain
r1 = np.min(LsCell_bz_local_norms[0:2]) / 2
H = lambda q: poly_localizer(q, r1, 4)

#   reciprocal lattice within the local domain
#   Needs modification for 2D
Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, [0], indexing='ij')
GptGrid3D_local = np.hstack((Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip

loc_grid = scipy.io.loadmat('loc_grid.mat')
GptGrid3D_local = loc_grid["GptGrid_Localizer"]

# Make combined localizer grid
GptGridz_localizer = GptGrid3D_local + z_coord_arr[0, :]
for i in range(1, len(z_coords)):
    tmp = GptGrid3D_local + z_coord_arr[i, :]
    GptGridz_localizer = np.vstack((GptGridz_localizer, tmp))
locz = scipy.io.loadmat('locz.mat')
GptGridz_localizer = locz['GptGridz_Localizer']
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
nqG_local = [N_local * nks[0], N_local * nks[1],1]  # lattice size along each dimension in the real-space (equal to q + G size)
Lvec_real_local = [Lvec_real[0] / N_local, Lvec_real[1] / N_local, Lvec_real[2]]  # dual real cell of local domain LsCell_bz_local
Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local

from scipy.integrate import quad
from scipy.special import i0

normR = np.linalg.norm(RptGrid3D_local, axis=1)

# np.pi * vac_size if x ==0 else
def kernel_func(R, Gz, vac_size, a, b):
    func = lambda x: 8 * np.pi * np.pi * x * (1 - np.cos(vac_size / 2 * Gz) * np.exp(-vac_size / 2 * x)) * iv(0,-1j * x * R) / (x ** 2 + Gz ** 2)
    integral = quad(func, a, b, limit=50)[0]
    return integral

# trunc_int = [kernel_func(R, vac_size, 0, r1) for R in normR]
CoulG = np.array([kernel_func(R, z_coords[0], vac_size, 0, r1) for R in normR]).reshape(-1, 1)
for Gz in z_coords[1:]:
    Coul_Gz = np.array([kernel_func(R, Gz, vac_size, 0, r1) for R in normR]).reshape(-1, 1)
    CoulG = np.hstack((CoulG, Coul_Gz))

# Exact expression when |R| = 0
# CoulR[normR < 1e-8] = 8*np.pi**2 * (np.log(vac_size * r1) + gamma(0,r1 * vac_size) * gammaincc(0, r1 * vac_size) + np.euler_gamma)

#   Step 4: Compute the correction

ss_correction = 0

#   Integral with Fourier Approximation
for iq, qpt in enumerate(qGrid):
    qG = qpt[None, :] + GptGridz_localizer
    qGxy = qpt[None, :] + GptGrid3D_local
    exp_mat = np.exp(1j * (qGxy @ RptGrid3D_local.T))
    tmp = (exp_mat @ CoulG) / (N_local ** 2)
    tmp = tmp.reshape(-1, 1, order = 'F')
    tmp = tmp.flatten()
    prod = SqG_local[iq, :].T * H(qG) * tmp
    ss_correction -= np.real(np.sum(prod)) / nkpts

print(ss_correction)
#   Quadrature with Coulomb kernel
for iq, qpt in enumerate(qGrid):
    qG = qpt[None, :] + GptGridz_localizer
    qG_no_z = qG[:, 0:2]
    tmp = SqG_local[iq, :].T * H(qG) / np.sum(qG ** 2,axis=1)  # (1 - np.exp(-vac_size/2 * np.sum(qG **2, axis =1))) * np.cos(vac_size/2 * qG[2])
    coul = np.exp(-vac_size / 2 * np.sqrt(np.sum(qG_no_z ** 2, axis=1))) * np.cos(vac_size / 2 * qG[:,2])
    prod = tmp * (1 - coul) * 4 * np.pi
    qG0 = np.all(qG == 0, axis=1)
    indices = np.where(qG0)[0]
    if indices.size > 0:
        prod[indices] = SqG_local[iq, indices] * -np.pi * vac_size ** 2 / 2
    ss_correction += np.sum(prod) / nkpts * area_Bz

# ss_correction = 4 * np.pi * ss_correction/ np.linalg.det(Lvec_real)/np.linalg.norm(Lvec_recip[2])  # Coulomb kernel = 4 pi / |q|^2
print(ss_correction)
print(ss_correction * vac_size_bz **2)
#