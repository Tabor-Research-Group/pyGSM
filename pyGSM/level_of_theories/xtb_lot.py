
# third party
import numpy as np

from xtb.interface import Calculator
from xtb.utils import get_method, get_solvent
from xtb.interface import Environment
from xtb.libxtb import VERBOSITY_FULL

# local application imports
from ..utilities import manage_xyz, units, elements
from .base_lot import Lot


class xTB_lot(Lot):
    _default_options = None
    @classmethod
    def default_options(cls):
        if cls._default_options is not None:
            return cls._default_options.copy()

        opt = super().default_options()

        opt.add_option(
                key='xTB_Hamiltonian',
                value='GFN2-xTB',
                required=False,
                allowed_types=[str],
                doc='xTB hamiltonian'
                )

        opt.add_option(
                key='xTB_accuracy',
                value=1.0,
                required=False,
                allowed_types=[float],
                doc='xTB accuracy'
                )

        opt.add_option(
                key='xTB_electronic_temperature',
                value=300,
                required=False,
                allowed_types=[float],
                doc='xTB electronic_temperature'
                )

        opt.add_option(
            key='solvent',
            value=None,
            required=False,
            allowed_types=[str],
            doc='xTB solvent'
        )

        cls._default_options = opt
        return cls._default_options.copy()

    def __init__(self,
                 *,
                 xTB_Hamiltonian='GFN2-xTB',
                 xTB_accuracy=1.0,
                 xTB_electronic_temperature=300,
                 **kwargs
                 ):
        super().__init__(kwargs)

        self.xTB_Hamiltonian = xTB_Hamiltonian
        self.xTB_accuracy = xTB_accuracy
        self.xTB_electronic_temperature = xTB_electronic_temperature

    def run(self, geom, multiplicity, state, verbose=False):

        # no reason
        numbers = []
        E = elements.ElementData()
        for a in manage_xyz.get_atoms(geom):
            elem = E.from_symbol(a)
            numbers.append(elem.atomic_num)
        coords = manage_xyz.xyz_to_np(geom)

        # convert to bohr
        positions = coords * units.ANGSTROM_TO_AU
        calc = Calculator(get_method(self.xTB_Hamiltonian), numbers, positions, charge=self.charge)

        calc.set_accuracy(self.xTB_accuracy)
        calc.set_electronic_temperature(self.xTB_electronic_temperature)

        if self.solvent is not None:
            calc.set_solvent(get_solvent(self.solvent))

        calc.set_output('lot_jobs_{}.txt'.format(self.node_id))
        res = calc.singlepoint()  # energy printed is only the electronic part
        calc.release_output()

        # energy in hartree
        self._Energies[(multiplicity, state)] = self.Energy(res.get_energy(), 'Hartree')

        # grad in Hatree/Bohr
        self._Gradients[(multiplicity, state)] = self.Gradient(res.get_gradient(), 'Hartree/Bohr')

        # write E to scratch
        self.write_E_to_file()

        return res


if __name__ == "__main__":

    geom = manage_xyz.read_xyz('../../data/ethylene.xyz')
    # geoms=manage_xyz.read_xyzs('../../data/diels_alder.xyz')
    # geom = geoms[0]
    # geom=manage_xyz.read_xyz('xtbopt.xyz')
    xyz = manage_xyz.xyz_to_np(geom)
    # xyz *= units.ANGSTROM_TO_AU

    lot = xTB_lot.from_options(states=[(1, 0)], gradient_states=[(1, 0)], geom=geom, node_id=0)

    E = lot.get_energy(xyz, 1, 0)
    print(E)

    g = lot.get_gradient(xyz, 1, 0)
    print(g)
