import scipy.io
import numpy as np

u_data = scipy.io.loadmat('ukpt.mat')
uKpts = u_data['uKpt']
print(np.shape(uKpts))
LsCell = np.array([[1,0,0],[0,1,0],[0,0,1]])
NsCell = np.array([12,12,12])
nocc = 1


kpts = np.array([[0,0,0], [np.pi,0,0],[0,np.pi,0],[np.pi,np.pi,0],[0,0,np.pi], [np.pi,0,np.pi],[0,np.pi,np.pi],[np.pi,np.pi,np.pi]])
recip_vec = np.linalg.inv(LsCell.T)*2*np.pi
L_incre = LsCell/NsCell[:,np.newaxis]
dvol = np.linalg.det(L_incre)
Nk = 27
#CHECK UP TO THIS POINT
#NOW, GRID IS EXACTLY THE SAME. FLIP Z AND X POSITIONS
Z, Y, X = np.meshgrid(np.arange(0, NsCell[2]), np.arange(0, NsCell[1]), np.arange(0, NsCell[0]), indexing='ij')
rptGrid3D = (X.flatten()[:, np.newaxis] * L_incre[0] + Y.flatten()[:, np.newaxis] * L_incre[1] + Z.flatten()[:,np.newaxis] * L_incre[2])
LsCell_bz = np.linalg.inv(LsCell.T)*2*np.pi

#GRID IS THE SAME, BUT DIFFERENT ORDER
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


def localizer(x, r0, r1):
    r = np.sqrt(np.sum(x ** 2, axis=1))
    val = np.exp(-1.0 / (r1 - r)) / (np.exp(-1.0 / (r - r0)) + np.exp(-1.0 / (r1 - r)))
    val[r <= r0] = 1
    val[r >= r1] = 0
    return val

def pair_product_recip_exchange(uKpt, kptGrid3D, rptGrid3D, NsCell, dvol, cell, nbands):
    '''
    NsCell will be the number of plane waves in each direction
    LsCell will be the length of the lattice vectors
    kptGrid3D will be the k-point grid used in the calculation
    ukpt** will be the wavefunction evaluated on all grid points.
    nocc will be an occupation number
    '''

    qGrid = kptGrid3D - kptGrid3D[0,:]
    kGrid = kptGrid3D
    nkpt = kGrid.shape[0]
    LsCell_bz = np.linalg.inv(LsCell.T)*2*np.pi
    print("RECIPROCAL LATTICE VECTORS")
    print(LsCell_bz)
    print("QGRID")
    #Subtract out copies of reciprocal lattice vectors? Construct grid in First Brillouin zone.
    for q in range(nkpt):
        qpt = qGrid[q, :]
        #Define qpt in the basis of the reciprocal lattice vectors
        qpt_trans = np.linalg.inv(LsCell_bz.T).dot(qpt)
        # Get rid of multiples of recip vectors
        #for i in range(len(LsCell_bz)):
            #qpt = qpt - np.floor((np.dot(qpt + 1e-12, LsCell_bz[i])) / np.dot(LsCell_bz[i], LsCell_bz[i])) * LsCell_bz[i]
        for i in range(3):
            if abs(qpt_trans[i])>1e-8:
                qpt_trans[i] = round(qpt_trans[i],6)%1
            qpt_trans[i] = qpt_trans[i]%1
        # Bring into first Brillouin zone
        #qpt = qpt - np.where(np.dot(qpt, LsCell_bz.T) / (np.linalg.norm(LsCell_bz, axis=1) ** 2) >= 0.5, 1,
                             #0) @ LsCell_bz
        for i in range(3):
            if qpt_trans[i]>=0.5:
                qpt_trans[i] -=1
        #Transform back into cartesian coordinates
        qpt = np.dot(LsCell_bz.T, qpt_trans)
        # Update qGrid
        qGrid[q, :] = qpt

    #print(qGrid)
    nG = uKpts.shape[0]
    rhokqmnG = np.zeros((nkpt,nkpt,nbands,nbands, nG),dtype = complex)

    print("KGRID IS")
    print(kGrid)

    print("QGRID IS")
    print(qGrid)
    for k in range(nkpt):
        for q in range(nkpt):
            kpt1 = kGrid[k,:]
            qpt = qGrid[q,:]
            kpt2 = kpt1 + qpt
            #Bring back into the cell
            trans_point = np.linalg.inv(LsCell_bz.T).dot(kpt2)
            for i in range(3):
                if abs(trans_point[i])>1e-6:
                    trans_point[i] = round(trans_point[i],6)%1

            kpt2 = np.dot(LsCell_bz.T,trans_point)

            d2 = np.sum((kGrid - kpt2) ** 2, axis=1)

            idx_kpt2 = np.where(d2 < 1e-10)[0]
            if len(idx_kpt2) != 1:
                raise TypeError("Cannot locate (k+q) in the kmesh.")
            else:
                idx_kpt2 = idx_kpt2[0]
            kGdiff = (kpt1 + qpt) - kpt2

            for n in range(nbands):
                for m in range(nbands):
                    u1 = uKpt[:,n,k]
                    u2 = np.exp(-1j * (np.dot(rptGrid3D, kGdiff.T))) * uKpt[:,m,idx_kpt2]
                    rho12 = np.conj(u1) * u2
                    rho12 = np.reshape(rho12 , (NsCell[0], NsCell[1], NsCell[2]) , order = 'F')
                    temp_fft = np.fft.fftn((np.array(rho12)*dvol))
                    rhokqmnG[k,q,n,m,:] = temp_fft.reshape(-1,order = 'F')


    return rhokqmnG, kGrid, qGrid

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

def localizer(x, r0, r1):
    r = np.sqrt(np.sum(x**2, axis=1))
    val = np.exp(-1.0 / (r1 - r)) / (np.exp(-1.0 / (r - r0)) + np.exp(-1.0 / (r1 - r)))
    val[r <= r0] = 1
    val[r >= r1] = 0
    return val

rho_kqijG, kGrid, qGrid = pair_product_recip_exchange(uKpt = uKpts, kptGrid3D= kpts, rptGrid3D = rptGrid3D, NsCell = NsCell, dvol = dvol , cell = None, nbands= nocc)
sum_res = np.sum(np.abs(rho_kqijG)**2 , axis = (0,2,3),keepdims=True)
sum_res = np.reshape(sum_res, (rho_kqijG.shape[1],rho_kqijG.shape[4]),order = 'F')
SqG = 1/Nk * sum_res
SqG-=nocc

#LsCell_bz_incre = LsCell_bz/NsCell[:,np.newaxis]
cell_bz_1 = np.concatenate((np.arange(0, NsCell[0] // 2 + 1), np.arange(-NsCell[0] // 2 + 1, 0)))
cell_bz_2 = np.concatenate((np.arange(0, NsCell[1] // 2 + 1), np.arange(-NsCell[1] // 2 + 1, 0)))
cell_bz_3 = np.concatenate((np.arange(0, NsCell[2] // 2 + 1), np.arange(-NsCell[2] // 2 + 1, 0)))
Zbz, Ybz, Xbz = np.meshgrid(cell_bz_3,cell_bz_2,cell_bz_1, indexing = 'ij')
cell_grid_bz = (Xbz.flatten()[:, np.newaxis] * LsCell_bz[0] + Ybz.flatten()[:, np.newaxis] * LsCell_bz[1] + Zbz.flatten()[:, np.newaxis] * LsCell_bz[2])

print("Recip lattice unit cell")
print(cell_grid_bz)
#Localizer support setting
N_local = 5
#Establish Localizer grid
LsCell_bz_local = N_local * LsCell_bz
Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local + 1) // 2 + 1, 0)))
Zl, Yl, Xl = np.meshgrid(Grid_1D, Grid_1D, Grid_1D, indexing='ij')
loc_grid = (Zl.flatten()[:,np.newaxis] *LsCell_bz[2] + Yl.flatten()[:, np.newaxis] * LsCell_bz[1] + Xl.flatten()[:, np.newaxis] * LsCell_bz[0])

LsCell_bz_local_norms = [sum(c** 2 for c in v) ** 0.5 for v in LsCell_bz_local]

r0 = 0 * np.min(LsCell_bz_local_norms) /2
r1 = 1 * np.min(LsCell_bz_local_norms) / 2

#H = lambda q: localizer(q, r0, r1)
H = lambda q: poly_localizer(q,r1,4)

#Index locator
idxG_localizer = np.zeros((np.shape(loc_grid)[0],1), dtype = int)
for i in range(np.shape(loc_grid)[0]):
    dist = np.sum((cell_grid_bz - loc_grid[i,:]) ** 2, axis=1)
    idxG_localizer[i] = np.where(dist < 1e-12)[0][0]

idxG_localizer = idxG_localizer.flatten()
SqG = SqG[:,idxG_localizer]

#ATTENTION NEEDED: What to do in the generalized case where the reciprocal cell vectors aren't along the axes?
#MISTAKE: Use LsCellBZ, not the local extended version. This may be leading to the blow up behavior...
LsCell_bz_local_norm2 = [sum(c ** 2 for c in v) for v in LsCell_bz_local]
inv_LsCell_bz_local = np.array(LsCell_bz_local)/LsCell_bz_local_norm2
nk = np.array([2,2,2])
N = N_local * np.array(nk)
#Testing
#N = 13 * np.array(nk)
if N[0]%2 == 0:
    G_1 = 2 * np.pi * np.concatenate((np.arange(0, N[0] // 2 +1), np.arange(-N[0] // 2 + 1, 0)))
else:
    G_1 = 2 * np.pi * np.concatenate((np.arange(0, (N[0] - 1) // 2 + 1), np.arange(-(N[0] + 1) // 2 + 1, 0)))
if N[1]%2 == 0:
    G_2 = 2 * np.pi * np.concatenate((np.arange(0, N[1] // 2 + 1), np.arange(-N[1] // 2 + 1, 0)))
else:
    G_2 = 2 * np.pi * np.concatenate((np.arange(0, (N[1] - 1) // 2 + 1), np.arange(-(N[1] + 1) // 2 + 1, 0)))
if N[2]%2 == 0:
    G_3 = 2 * np.pi * np.concatenate((np.arange(0, N[2] // 2 + 1), np.arange(-N[2] // 2 + 1, 0)))
else:
    G_3 = 2 * np.pi * np.concatenate((np.arange(0, (N[2] - 1) // 2 + 1), np.arange(-(N[2] + 1) // 2 + 1, 0)))

Zf, Yf, Xf = np.meshgrid(G_3, G_2, G_1, indexing='ij')
gptGrid_fourier = (Xf.flatten()[:, np.newaxis] * inv_LsCell_bz_local[0] + Yf.flatten()[:, np.newaxis] * inv_LsCell_bz_local[1] + Zf.flatten()[:, np.newaxis] * inv_LsCell_bz_local[2])
normG = np.sqrt(np.sum(gptGrid_fourier ** 2, axis=1))
from scipy.special import sici
coulG = 4 * np.pi / normG * sici(normG * r1)[0]
coulG[normG < 1e-12] = 4 * np.pi * r1

#Implementing the correction
correction = 0
#quadrature
for iq in range(np.shape(qGrid)[0]):
    qG = qGrid[iq,:] + loc_grid
    print(SqG[iq,:])
    tmp = SqG[iq,:] * H(qG)/ np.sum(qG**2, axis=1)
    tmp[np.isinf(tmp)] = 0
    tmp[np.isnan(tmp)] = 0
    print(tmp)
    correction += np.sum(tmp)/Nk
    print(correction)
print(correction)
#Integral
print("QGRID")
print(qGrid)
for iq in range(np.shape(qGrid)[0]):
    qG = qGrid[iq,:] + loc_grid
    exp_mat = np.exp(1j * np.dot(qG, gptGrid_fourier.T))
    tmp = np.dot(exp_mat, coulG.reshape(-1, order = 'F')) * 1/np.abs(np.linalg.det(LsCell_bz_local))
    tmp = SqG[iq,:].T * H(qG) * tmp
    correction -= 1/Nk * np.real(np.sum(tmp))

print(correction)