# standard library imports
import abc
from collections import namedtuple
import os

# third party
import numpy as np
import tempfile as tmp

# local application imports
from ..utilities import manage_xyz, elements, units, Devutils as dev
from .file_options import File_Options

ELEMENT_TABLE = elements.ElementData()

class LoTError(Exception):
    pass

Energy = namedtuple('Energy','value unit')
Gradient = namedtuple('Gradient','value unit')
Coupling = namedtuple('Coupling','value unit')

class LoT(metaclass=abc.ABCMeta):
    """ Lot object for level of theory calculators """

    energy_units = "Hartree"
    distance_units = "BohrRadius"

    default_states = [(0,1)]
    def __init__(self,
                 states=None,
                 gradient_states=None,
                 coupling_states=None,
                 charge=0,
                 calc_grad=None,
                 do_coupling=None,
                 atoms=None,
                 numbers=None,
                 logger=None
                 ):
        """ Constructor """
        # self.options = options
        self.logger = dev.Logger.lookup(logger)

        if states is None: states = self.default_states
        self.charge = charge
        self.atoms = atoms

        if calc_grad is None:
            calc_grad = gradient_states is not None
        if do_coupling is None:
            do_coupling = coupling_states is not None
        self.states, self.gradient_states, self.coupling_states = self._resolve_state_data(
            states, gradient_states, coupling_states,
            calc_grad=calc_grad,
            do_coupling=do_coupling,
        )
        if numbers is None and atoms is not None:
            numbers = [
                ELEMENT_TABLE.from_symbol(atom).atomic_num
                    if isinstance(atom, str) else
                atom.atomic_num
                    if hasattr(atom, 'atomic_num') else
                atom
                for atom in atoms
            ]
        self.numbers = numbers
        if self.numbers is not None:
            self.check_num_electrons_and_mults(
                self.numbers,
                self.charge,
                [m for m,_ in self.states]
            )
        self.calc_grad = calc_grad
        self.do_coupling = calc_grad

    def _resolve_state_data(self, states, gradient_states, coupling_states,
                            *,
                            calc_grad,
                            do_coupling,
                            max_multiplicity=5,
                            complete_state_data=True,
                            ):

        if gradient_states is None and calc_grad:
            gradient_states = states

        if coupling_states is None and do_coupling:
            coupling_states = states

        if complete_state_data:
            # count number of states
            tuple_search_data = self.search_tuple(states)

            state_vecs = [
                tuple_search_data.get(i+1, [])
                for i in range(max_multiplicity)
            ]
            state_lens = [
                0
                    if len(s) == 0 else
                max([j for m,j in s])
                for s in state_vecs
            ]

            states = [
                (m, j)
                for m,s in zip(range(max_multiplicity), state_lens)
                for j in range(s)
            ]

        return states, gradient_states, coupling_states

    def _setup_job_dat(self):
        # package  specific implementation
        # TODO MOVE to specific package !!!
        # tc cloud
        self.options['job_data']['orbfile'] = self.options['job_data'].get('orbfile', '')
        # pytc? TODO
        self.options['job_data']['lot'] = self.options['job_data'].get('lot', None)

        print(" making folder scratch/{:03}/{}".format(self.ID, self.node_id))
        os.system('mkdir -p scratch/{:03}/{}'.format(self.ID, self.node_id))

    def get_state_dict(self):
        return dict(
            states=self.states,
            gradient_states=self.gradient_states,
            coupling_states=self.coupling_states,
            charge=self.charge,
            atoms=self.atoms,
            numbers=self.numbers
        )
    def copy(self):
        return type(self)(**self.get_state_dict())

    @classmethod
    def check_multiplicity(cls, n_electrons, multiplicity):
        if multiplicity > n_electrons + 1:
            raise ValueError("Spin multiplicity too high.")

    @classmethod
    def check_num_electrons_and_mults(cls, atomic_nums, charge, multiplicitities):
        n_electrons = sum(atomic_nums) - charge
        if n_electrons < 0:
            raise ValueError("Molecule has fewer than 0 electrons!!!")
        for m in multiplicitities:
            cls.check_multiplicity(n_electrons, m)
        return n_electrons

    @abc.abstractmethod
    def run(self, coords, mult, ad_idx, *, runtypes):
        ...

    def energy_obj(self, eng):
        return Energy(eng, self.energy_units)
    def grad_obj(self, grad):
        return Gradient(grad, self.energy_units + "/" + self.distance_units)
    def coupling_obj(self, cup):
        return Coupling(cup, self.energy_units + "/" + self.distance_units)

    def runall(self, coords, *, runtype="energy"):
        results = {}
        if isinstance(runtype, str):
            runtype = [runtype]
        for state in self.states:
            mult, ad_idx = state
            runtypes = set(runtype)
            if state in self.gradient_states:
                runtypes.add('energy')
            if state in self.coupling_states:
                runtypes.add('coupling')
            results[(mult, ad_idx)] = self.run(coords, mult, ad_idx, runtypes=runtypes)
        return LoTResults(results,
                          energy_units=self.energy_units,
                          distance_units=self.distance_units,
                          )
    def get_energy(self, coords):
        return self.runall(coords, runtype={"energy"})
    def get_gradient(self, coords):
        return self.runall(coords, runtype={"energy", "gradient"})

    def search_PES_tuple(self, tups, multiplicity, state):
        '''returns tuple in list of tuples that matches multiplicity and state'''
        return [tup for tup in tups if multiplicity == tup[0] and state == tup[1]]

    def search_tuple(self, tups, multiplicity=None):
        if multiplicity is None:
            tup_data = {}
            for tup in tup_data:
                mult = tup[0]
                if mult not in tup_data: tup_data[mult] = []
                tup_data[mult].append(tup)
            return tup_data
        else:
            return [tup for tup in tups if multiplicity == tup[0]]

    @classmethod
    def rmsd(cls, geom1, geom2):
        total = 0
        flat_geom1 = np.array(geom1).flatten()
        flat_geom2 = np.array(geom2).flatten()
        for i in range(len(flat_geom1)):
            total += (flat_geom1[i] - flat_geom2[i]) ** 2
        return total

    def pick_best_orb_from_lots(self, lots):
        '''
        The idea is that this would take a list of lots and pick the best one for a node
        Untested!
        '''
        rmsds = []
        xyz1 = manage_xyz.xyz_to_np(self.geom)
        for lot in lots:
            rmsds.append(lot.rmsd(xyz1, lot.self.currentCoords))
        minnode = rmsds.index(min(rmsds))
        self.lot = self.lot.copy(lots[minnode])

        return

class LoTResults:
    def __init__(self, result_data):
        self.results = result_data

    def list_energies(self):
        if Energy.unit == "Hartree":
            return Energy.value * units.KCAL_MOL_PER_AU
        elif Energy.unit == 'kcal/mol':
            return Energy.value
        elif Energy.unit is None:
            return Energy.value
        return Energy

    def get_energy(self, multiplicity, state):
        eng = self.results[(multiplicity,state)]["energy"]
        if eng.unit is None or eng.unit == 'kcal/mol':
            return eng.value
        elif eng.unit == "Hartree":
            return eng.value * units.KCAL_MOL_PER_AU
        else:
            raise NotImplementedError(f"unhandled conversion between {eng.unit} and kcal/mol")

    def get_gradient(self, multiplicity, state, frozen_atoms=None):
        grad = self.results[(multiplicity,state)]["gradient"]
        if grad.value is not None:
            if frozen_atoms is not None:
                for a in frozen_atoms:
                    grad.value[a, :] = 0.
            if grad.unit is None or grad.unit == "Hartree/Angstrom":
                return grad.value
            else:
                e_unit, d_unit = grad.unit.rsplit("/", 1)
                val = grad.value
                if e_unit == "kcal/mol":
                    val = val * units.KCAL_MOL_TO_AU
                elif e_unit != "Hartree":
                    raise NotImplementedError(f"unhandled conversion between {d_unit} and Angstrom")

                if d_unit == "BohrRadius":
                    val = val * units.ANGSTROM_TO_AU
                elif d_unit != "Angstrom":
                    raise NotImplementedError(f"unhandled conversion between {e_unit} and Angstrom")

                return val
        else:
            return None

    def get_coupling(self, multiplicity, state1, state2, frozen_atoms=None):
        cup = self.results[(state1, state2)]["coupling"]
        if cup.value is not None:
            if frozen_atoms is not None:
                for a in [3*i for i in frozen_atoms]:
                    cup.value[a:a+3, 0] = 0.
            if cup.unit is None or cup.unit == "Hartree/Angstrom":
                return cup.value
            else:
                e_unit, d_unit = cup.unit.rsplit("/", 1)
                val = cup.value
                if e_unit == "kcal/mol":
                    val = val * units.KCAL_MOL_TO_AU
                elif e_unit != "Hartree":
                    raise NotImplementedError(f"unhandled conversion between {d_unit} and Angstrom")

                if d_unit == "BohrRadius":
                    val = val * units.ANGSTROM_TO_AU
                elif d_unit != "Angstrom":
                    raise NotImplementedError(f"unhandled conversion between {e_unit} and Angstrom")

                return val
        else:
            return None

    def get_energy_file(self,
                        id_format=None,
                        node_id_format=None,
                        base_file_format='E_{node_id_fmt}.txt'):
        if id_format is None:
            id_format = self.id_format
        if node_id_format is None:
            node_id_format = self.node_id_format

        return os.path.join(
            self.scratch_dir,
            id_format.format(self.ID),
            base_file_format.format(
                node_id_format.format(self.node_id)
            )
        )
    # def write_E_to_file(self, file=None, energy_format='{mult} {state} {eng:9.7f} Hartree'):
    #     if file is None:
    #         file = self.get_energy_file()
    #     if hasattr(file, 'write'):
    #         for (mult, state), Energy in self.Energies.items():
    #             file.write(energy_format.format(mult=mult, state=state, eng=Energy.value) + "\n")
    #     else:
    #         with open(file, 'w') as f:
    #             self.write_E_to_file(f)
    #     return file

class FileBasedLoT(LoT):
    _default_options = None
    @classmethod
    def default_options(cls):
        """ Lot default options. """
        if cls._default_options is not None:
            return cls._default_options.copy()

        opt = super().default_options()

        opt.add_option(
            key="lot_inp_file",
            required=False,
            value=None,
            doc='file name storing LOT input section. Used for custom basis sets,\
                         custom convergence criteria, etc. Will override nproc, basis and\
                         functional. Do not specify charge or spin in this file. Charge \
                         and spin should be specified in charge and states options.\
                         for QChem, include $molecule line. For ORCA, do not include *xyz\
                         line.'
        )

        opt.add_option(
            key='file_options',
            value=None,
            allowed_types=[File_Options],
            doc='A specialized dictionary containing lot specific options from file\
                            including checks on dependencies and clashes. Not all packages\
                            require'
        )

        cls._default_options = opt
        return cls._default_options.copy()

    def __init__(self,
                 *,
                 lot_inp_file,
                 file_options=None,
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.lot_inp_file = lot_inp_file
        if file_options is None:
            file_options = File_Options(lot_inp_file)
        self.file_options = file_options