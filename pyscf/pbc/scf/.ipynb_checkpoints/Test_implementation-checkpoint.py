{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f4f103a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyscf\n",
    "import numpy\n",
    "from pyscf import gto\n",
    "from pyscf import scf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9e448e61",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Test System\n",
    "def build_h2_cell(nk = (1,1,1),kecut=100):\n",
    "    cell = pbcgto.Cell()\n",
    "    cell.unit = 'Bohr'\n",
    "    cell.atom='''\n",
    "        H 3.00   3.00   2.10\n",
    "        H 3.00   3.00   3.90\n",
    "        '''\n",
    "    cell.a = '''\n",
    "        6.0   0.0   0.0\n",
    "        0.0   6.0   0.0\n",
    "        0.0   0.0   6.0\n",
    "        '''\n",
    "    cell.verbose = 7\n",
    "    cell.spin = 0\n",
    "    cell.charge = 0\n",
    "    \n",
    "    \n",
    "    cell.basis = {'H':'gth-dzvp'}\n",
    "    cell.pseudo = 'gth-pbe'\n",
    "    \n",
    "    cell.ke_cutoff = kecut\n",
    "    cell.output = cwd + '/h2' + nk_output_str(nk) + kecut_output_str(kecut) +'.out'\n",
    "    cell.max_memory = 1000\n",
    "    cell.precision = 1e-8\n",
    "    #for i in range(len(cell.atom)):\n",
    "    #   cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))\n",
    "    cell.build()\n",
    "    kpts = cell.make_kpts(nk, wrap_around=False)    \n",
    "    return cell, kpts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6b77433",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Trying to implement Staggered Mesh for Hartree Fock. Study Stephen's paper and Xin's implementation\n",
    "of Staggered Mesh for MP2, which should take a similar form.\n",
    "\n",
    "Author: Hamlin Wu\n",
    "'''\n",
    "\n",
    "#Starting with Non-Consistent SCF implementation, which seems to be the easiest.\n",
    "#Start with the implemented SCF calculation\n",
    "\n",
    "class HF_stagger_nsc()\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
