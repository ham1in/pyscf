import pyscf
import numpy
from pyscf import gto
from pyscf import scf


### Test System
def build_h2_cell(nk = (1,1,1),kecut=100):
    cell = pbcgto.Cell()
    cell.unit = 'Bohr'
    cell.atom='''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    cell.verbose = 7
    cell.spin = 0
    cell.charge = 0
    
    
    cell.basis = {'H':'gth-dzvp'}
    cell.pseudo = 'gth-pbe'
    
    cell.ke_cutoff = kecut
    cell.output = cwd + '/h2' + nk_output_str(nk) + kecut_output_str(kecut) +'.out'
    cell.max_memory = 1000
    cell.precision = 1e-8
    #for i in range(len(cell.atom)):
    #   cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
    cell.build()
    kpts = cell.make_kpts(nk, wrap_around=False)    
    return cell, kpts



'''
Trying to implement Staggered Mesh for Hartree Fock. Study Stephen's paper and Xin's implementation
of Staggered Mesh for MP2, which should take a similar form.

Author: Hamlin Wu
'''

#Starting with Non-Consistent SCF implementation, which seems to be the easiest.
#Start with the implemented SCF calculation

class HF_stagger_nsc(khf.KHF):
    def __init__(self, mf, frozen=None, flag_submesh=False):
        

       self.cell = mf.cell
       self._scf = mf
       self.verbose = self.cell.verbose
       self.stdout = self.cell.stdout
       self.max_memory = self.cell.max_memory

       #Staggered mesh energy
       self.e_stagger = None
       
       #These seem to be needed (From Xin's code)
       #nocc and nmo variables
       self._nocc = None
       self._nmo = None

       #Get orbitals and orbital energies on staggered mesh. Use Non-SCF approach here.
       nks = get_monkhorst_pack_size(mf.cell,mf.kpts)

