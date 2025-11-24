"""
Level Of Theory for ASE calculators
https://gitlab.com/ase/ase

Written by Tamas K. Stenczel in 2021
"""
import importlib

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase import units
from .base_lot import LoT, LoTError

class Constraint_custom_forces:
    def __init__(self, a, direction):
        self.a = a
        self.dir = direction

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):
        forces[self.a] = forces[self.a] + self.dir

class ASELoT(LoT):
    """
    Warning:
        multiplicity is not implemented, the calculator ignores it
    """
    # energy_units = "Hartrees"
    distance_units = "Angstrom"

    def __init__(self,
                 calculator: Calculator,
                 constraints_forces=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.ase_calculator = calculator
        self.constraints_forces = constraints_forces

    def get_state_dict(self):
        return dict(super().get_state_dict(),
                    calculator=self.ase_calculator,
                    constraints_forces=self.constraints_forces)

    @classmethod
    def from_calculator_string(cls, calculator_import: str, calculator_kwargs: dict = None, **kwargs):
        # this imports the calculator
        module_name = ".".join(calculator_import.split(".")[:-1])
        class_name = calculator_import.split(".")[-1]

        # import the module of the calculator
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise LoTError(
                "ASE-calculator's module is not found: {}".format(class_name))

        # class of the calculator
        if hasattr(module, class_name):
            calc_class = getattr(module, class_name)
            assert issubclass(calc_class, Calculator)
        else:
            raise LoTError(
                "ASE-calculator's class ({}) not found in module {}".format(class_name, module_name))

        # make sure there is no calculator in the options
        _ = kwargs.pop("calculator", None)

        # construct from the constructor
        if calculator_kwargs is None:
            calculator_kwargs = {}
        return cls(calc_class(**calculator_kwargs), **kwargs)

    @classmethod
    def read_constraints_file(cls, constraints_file):
        with open(constraints_file) as f:
            tmp = filter(None, (line.rstrip() for line in f))
            lines = []
            for line in tmp:
                lines.append(line)
        constraints = []
        for line in lines:
            idx1 = int(line.split()[0])
            idx2 = int(line.split()[1])
            value = float(line.split()[2])
            constraints.append([idx1, idx2, value])
        return constraints

    def run_raw(self, coords, mult, ad_idx, *, runtypes):
        # run ASE
        atoms = Atoms(numbers=self.numbers, positions=coords)
        return self.run_ase_atoms(atoms, mult, ad_idx, runtypes)

    def run_ase_atoms(self, atoms: Atoms, mult, ad_idx, runtypes):
        # set the calculator
        for runtype in runtypes:
            if runtype not in {'energy', 'gradient'}:
                raise NotImplementedError(f"Run type {runtype} is not implemented in the ASE calculator interface")

        atoms.calc = self.ase_calculator
        if self.constraints_forces is not None and self.constraints_forces[0][0] != None:
            atom_indices = [x for x in range(len(atoms))]
            constraint = Constraint_custom_forces(atom_indices, self.constraints_forces)
            atoms.set_constraint(constraint)

        res = {}
        if 'gradient' in runtypes:
            res['gradient'] = self.grad_obj(-atoms.get_forces().flatten() / units.Ha)
        if 'energy' in runtypes:
            res['energy'] = self.energy_obj(atoms.get_potential_energy()[0] / units.Ha)

        return res