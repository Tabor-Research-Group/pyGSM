"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""

from time import time
import numpy as np
import os
# from collections import Counter

from ..coordinate_systems import InternalCoordinates, CartesianCoordinates
from .. import coordinate_systems as coord_ops
from ..level_of_theories.base_lot import LoT
from ..level_of_theories.conveniences import construct_lot
from ..utilities import manage_xyz, elements, block_matrix, Devutils as dev, units

ELEMENT_TABLE = elements.ElementData()

class Molecule:

    def __init__(self,
                 coord_obj:InternalCoordinates,
                 *,
                 charge=0,
                 comment="",
                 frozen_atoms=None,
                 logger=None,
                 energy=None,
                 gradient=None,
                 primitive_gradient=None,
                 derivative_coupling=None,
                 primitive_coupling=None,
                 energy_evaluator=None,
                 hessian=None,
                 primitive_hessian=None,
                 energy_evaluator_options=None
                 ):

        self.coord_obj = coord_obj
        self.using_dlcs = coord_ops.is_dlc(self.coord_obj)
        atoms = coord_obj.atoms
        if isinstance(atoms[0], str):
            atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        self.charge = charge

        self.logger = dev.Logger.lookup(logger)
        if self.xyz.shape[-1] != 3 or len(atoms) != self.xyz.shape[-2]:
            raise ValueError(f"expected `xyz` to be {len(atoms)}x3, got {self.xyz.shape}")

        # create a dictionary from atoms
        # atoms contain info you need to know about the atoms
        self.atoms = atoms
        self.frozen_atoms = frozen_atoms

        if energy_evaluator is not None and not hasattr(energy_evaluator, 'get_gradient'):
            # just gonna duck type this, all we need really
            if energy_evaluator_options is None:
                energy_evaluator_options = {}
            energy_evaluator = construct_lot(energy_evaluator, atoms=self.atoms, **energy_evaluator_options)
        self.evaluator = energy_evaluator

        self.comment = comment

        #TODO
        self.gradrms = 0.
        self.isTSnode = False
        self.bdist = 0.
        self.newHess = 5

        self._hessian = hessian
        self._primitive_hessian = primitive_hessian
        self._prev_eval = None
        self._energy = energy
        self._gradient = gradient
        self._primitive_gradient = primitive_gradient
        self._derivative_coupling = derivative_coupling
        self._primitive_coupling = primitive_coupling

    def get_state_dict(self):
        return dict(
            coord_obj=self.coord_obj,
            charge=self.charge,
            energy_evaluator=self.evaluator,
            comment=self.comment,
            frozen_atoms=self.frozen_atoms,
            logger=self.logger,
            energy=self._energy,
            gradient=self._gradient,
            primitive_gradient=self._primitive_gradient,
            derivative_coupling=self._derivative_coupling,
            primitive_coupling=self._primitive_coupling,
            hessian=self._hessian,
            primitive_hessian=self._primitive_hessian
        )
    def modify(self, coord_obj=dev.default, energy_evaluator=dev.default, **changes):
        new_state = dict(self.get_state_dict(), **changes)
        no_cache = False
        if not dev.is_default(coord_obj, allow_None=False):
            no_cache = True
            new_state['coord_obj'] = coord_obj
        if not dev.is_default(energy_evaluator, allow_None=False):
            no_cache = True
            new_state['energy_evaluator'] = energy_evaluator
        new =  type(self)(**new_state)
        if no_cache:
            new.invalidate_cache()
        return new
    def copy(self):
        return self.modify()
    def modify_coordinate_system(self, **changes):
        return self.modify(
            coord_obj=self.coord_obj.modify(**changes)
        )
    def attach_evaluator(self, evaluator):
        return self.modify(energy_evaluator=evaluator)

    # @classmethod
    # def construct(cls, coord_sys, **opts):
    #     full_dict = cls.default_options()
    #     full_dict.update(opts)
    #     return cls(coord_sys, **full_dict)

    @classmethod
    def from_xyz(cls,
                 atoms, xyz,
                 bonds='auto',
                 primitives=None,
                 internals=None,
                 coordinate_type=None,
                 coordinate_system_options=None,
                 logger=None,
                 **opts):
        if coordinate_system_options is None:
            coordinate_system_options = {}
        logger = dev.Logger.lookup(logger)
        coord_sys = coord_ops.construct_coordinate_system(atoms, xyz,
                                                          bonds=bonds,
                                                          primitives=primitives,
                                                          internals=internals,
                                                          coordinate_type=coordinate_type,
                                                          logger=logger,
                                                          **coordinate_system_options
                                                          )
        return cls(coord_sys, logger=logger, **opts)

    @classmethod
    def from_file(cls, filename, **opts):
        atoms, xyz = cls.load_coordinates(
            filename=filename
        )
        return cls.from_xyz(atoms, xyz, **opts)

    @property
    def Hessian(self):
        if self._hessian is None:
            if self.Primitive_Hessian is not None:
                self.logger.log_print(" forming Hessian in basis")
                self._hessian = self.form_Hessian_in_basis()
            else:
                self._hessian = self.form_Hessian()
                self.newHess = 5 #TODO: figure out why
        return self._hessian
    @Hessian.setter
    def Hessian(self, hess):
        self._hessian = hess

    @classmethod
    def load_coordinates(cls, *, atoms=None, xyz=None, geom=None, filename=None, file_type=None):
        if atoms is None:
            if xyz is not None: raise ValueError("if `xyz` is supplied, `atoms` must be too")
            if geom is not None:
                atoms = manage_xyz.get_atoms(geom)
                xyz = manage_xyz.xyz_to_np(geom)
            elif filename is not None:
                if not os.path.exists(filename):
                    raise IOError(f"file {filename} doesn't exist")
                if file_type is None:
                    file_type = os.path.splitext(filename)[1][1:]
                if file_type != "xyz":
                    raise ValueError(f"currently only support xyz files (got `{file_type}`)")
                geom = manage_xyz.read_xyzs(filename, scale=1.)[0]
                xyz = manage_xyz.xyz_to_np(geom)
                atoms = manage_xyz.get_atoms(geom)
            else:
                raise ValueError("`geom` or `filename` is required")
        elif xyz is None:
            if xyz is not None: raise ValueError("if `atoms` are supplied, `xyz` must be too")

        return np.asanyarray(atoms), xyz

    # @property
    # def PES(self):
    #     if self._PES is None:
    #         if self._base_pes is not None:
    #             self._PES = self.resolve_PES(self._base_pes, **self.pes_options)
    #     return self._PES
    #
    # @classmethod
    # def resolve_PES(cls):
    #     type(self._base_pes).create_pes_from(
    #         PES=self._base_pes,
    #     )

    def center(self, center_mass=False):
        """ Move geometric center to the origin. """
        if center_mass:
            com = self.center_of_mass
            self.xyz -= com
        else:
            self.xyz -= self.xyz.mean(0)

    @property
    def atomic_mass(self):
        return np.array([units.AMU_TO_AU * ele.mass_amu for ele in self.atoms])

    @property
    def mass_amu(self):
        return np.array([ele.mass_amu for ele in self.atoms])

    @property
    def mass_amu_triples(self):
        return np.array([[ele.mass_amu, ele.mass_amu, ele.mass_amu] for ele in self.atoms])

    @property
    def atomic_num(self):
        return [ele.atomic_num for ele in self.atoms]

    @property
    def total_mass_au(self):
        """Returns the total mass of the molecule"""
        return np.sum(self.atomic_mass)

    @property
    def total_mass_amu(self):
        """Returns the total mass of the molecule"""
        return np.sum(self.mass_amu)

    @property
    def natoms(self):
        """The number of atoms in the molecule"""
        return len(self.atoms)

    #def atom_data(self):
    #    uniques = list(set(M.atoms))
    #    for a in uniques:
    #        nifty.printcool_dictionary(a._asdict())

    @property
    def center_of_mass(self):
        M = self.total_mass_au
        atomic_masses = self.atomic_mass
        return np.sum([self.xyz[i, :]*atomic_masses[i]/M for i in range(self.natoms)], axis=0)

    @property
    def mass_weighted_cartesians(self):
        M = self.total_mass_au
        return np.asarray([self.xyz[i, :]*self.atomic_mass[i]/M for i in range(self.natoms)])

    @property
    def radius_of_gyration(self):
        com = self.center_of_mass
        M = self.total_mass_au
        xyz1 = self.xyz.copy()
        xyz1 -= com
        return np.sum(self.atomic_mass[i]*np.dot(x, x) for i, x in enumerate(xyz1))/M

    # @property
    # def geometry(self):
    #     return manage_xyz.combine_atom_xyz(self.atom_symbols, self.xyz)

    @property
    def atom_symbols(self):
        return [a.symbol for a in self.atoms]

    @classmethod
    def construct_lot(cls, base_lot, **lot_opts):
        if isinstance(base_lot, LoT): return base_lot

    @property
    def _cached_calc(self):
        #TODO: migrate all of this stuff up to the GSM process/optimizer where it belongs...
        #      there's no reason this object should need to _guess_ at what a different operation
        #      needs cached
        if self._prev_eval is None:
            self._prev_eval = self.evaluator.get_all(self.xyz)
        return self._prev_eval

    @property
    def energy(self):
        if self._energy is None:
            self._energy = self._cached_calc["energy"]
        return self._energy

    @property
    def gradx(self):
        if self._primitive_gradient is None:
            self._energy = self._cached_calc["gradient"]
            # self._primitive_gradient = self.evaluator.get_gradient(self.xyz, frozen_atoms=self.frozen_atoms)
        return np.reshape(self._primitive_gradient, (-1, 3))

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self.coord_obj.calcGrad(self.xyz, self.gradx.flatten())
        return self._gradient
        gradx = self.evaluator.get_gradient(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, gradx)  # CartesianCoordinate just returns gradx

    # for PES seams
    # @property
    # def avg_gradient(self):
    #     gradx = self.PES.get_avg_gradient(self.xyz, frozen_atoms=self.frozen_atoms)
    #     return self.coord_obj.calcGrad(self.xyz, gradx)  # CartesianCoordinate just returns gradx

    @property
    def cupx(self):
        if self._primitive_coupling is None:
            self._primitive_coupling = self._cached_calc["coupling"]
        return np.reshape(self._primitive_coupling, (-1, 3))
    @property
    def derivative_coupling(self):
        if self._derivative_coupling is None:
            self._derivative_coupling = self.coord_obj.calcGrad(self.xyz, self.cupx.flatten())
        return self._derivative_coupling

    @property
    def difference_gradient(self):
        dgradx = self.PES.get_dgrad(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, dgradx)

    # @property
    # def difference_energy(self):
    #     self.energy # this is embarrassingly bad
    #     return self.PES.dE

    @property
    def Primitive_Hessian(self):
        if self._primitive_hessian is None:
            if not coord_ops.is_cartesian(self.coord_obj):
                self.logger.log_print(" making primitive Hessian")
                start = time()
                self._primitive_hessian = self.form_Primitive_Hessian()
                self.newHess = 10 #WTF is this...
                stop = time()
                self.logger.log_print(" Time to build Prim Hessian {e:.3f}s", e=stop - start)
        return self._primitive_hessian
    @Primitive_Hessian.setter
    def Primitive_Hessian(self, prim):
        self._primitive_hessian = prim

    def form_Primitive_Hessian(self):
        if coord_ops.is_dlc(self.coord_obj):
            prim = self.coord_obj.Prims.guess_hessian(self.xyz)
        else:
            prim = self.coord_obj.guess_hessian(self.xyz)
        return prim

    def update_Primitive_Hessian(self, change=None):
        print(" updating prim hess")
        if change is not None:
            self.Primitive_Hessian += change
        return self.Primitive_Hessian

    def form_Hessian(self):
        hess = self.coord_obj.guess_hessian(self.xyz)
        return hess

    def update_Hessian(self, change=None):
        #print " in update Hessian"
        if change is not None:
            self.Hessian += change
        return self.Hessian

    def form_Hessian_in_basis(self):
        # print " forming Hessian in current basis"
        return block_matrix.dot(block_matrix.dot(block_matrix.transpose(self.coord_basis), self._primitive_hessian), self.coord_basis)

    def invalidate_cache(self):
        self._prev_eval = None
        self._energy = None
        self._gradient = None
        self._primitive_gradient = None
        self._primitive_hessian = None
        self._hessian = None
        self._coupling = None
        self._primitive_coupling = None

    @property
    def xyz(self):
        return self.coord_obj.xyz
    @xyz.setter
    def xyz(self, newxyz):
        if newxyz is not None:
            #TODO: check whether or not I should actually invalidate the cache
            self.coord_obj.xyz = newxyz
            self.invalidate_cache()

    @property
    def bond_graph(self):
        return self.coord_obj.topology

    @property
    def num_frozen_atoms(self):
        if self.frozen_atoms is not None:
            return len(self.frozen_atoms)
        else:
            return 0

    def update_xyz(self, dq=None, verbose=True):
        if dq is not None:
            self.xyz = self.coord_obj.newCartesian(self.xyz, dq, frozen_atoms=self.frozen_atoms, verbose=verbose)
        return self.xyz

    def update_and_move(self, tan, deltadq, verbose=True):
        self.update_coordinate_basis(tan)
        dq = deltadq * self.constraint[:, 0]
        self.update_xyz(dq, verbose=verbose)

    def update_MW_xyz(self, mass, dq, verbose=True):
        self.xyz = self.coord_obj.massweighted_newCartesian(self.xyz, dq, mass, verbose)
        return self.xyz

    @property
    def primitive_internal_coordinates(self):
        if self.using_dlcs:
            return self.coord_obj.Prims.Internals
        else:
            raise ValueError(f"coord_obj {self.coord_obj} has not primitive internals")

    @property
    def num_primitives(self):
        return len(self.primitive_internal_coordinates)

    @property
    def num_bonds(self):
        count = 0
        for ic in self.coord_obj.Prims.Internals:
            if type(ic) == "Distance":
                count += 1
        return count

    @property
    def primitive_internal_values(self):
        ans = self.coord_obj.Prims.calculate(self.xyz)
        return np.asarray(ans)

    @property
    def coord_basis(self):
        return self.coord_obj.Vecs

    @coord_basis.setter
    def coord_basis(self, value):
        self.coord_obj.Vecs = value

    def update_coordinate_basis(self, constraints=None):
        if isinstance(self.coord_obj, CartesianCoordinates):
            return None
        # if constraints is not None:
        #     assert constraints.shape[0] == self.coord_basis.shape[0], '{} does not equal {} dimensions'.format(constraints.shape[0],self.coord_basis.shape[0])

        print(" updating coord basis")
        self.coord_obj.clearCache()
        self.coord_obj.build_dlc(self.xyz, constraints)
        return self.coord_basis

    @property
    def constraints(self):
        return self.coord_obj.Vecs.cnorms

    @property
    def coordinates(self):
        return np.reshape(self.coord_obj.calculate(self.xyz), (-1, 1))

    def mult_bm(self, left, right):
        return block_matrix.dot(left, right)

    @property
    def prim_CMatrix(self):
        return self.coord_obj.Prims.second_derivatives(self.xyz)
        #Der = self.coord_obj.Prims.second_derivatives(self.xyz)
        #return Der.reshape(Der.shape[0],3*self.xyz.shape[0],3*self.xyz.shape[0])

    @property
    def CMatrix(self):
        return self.coord_obj.second_derivatives(self.xyz)
        # Der = self.coord_obj.second_derivatives(self.xyz)
        # return Der.reshape(Der.shape[0],3*self.xyz.shape[0],3*self.xyz.shape[0])
        # return Der

        # Answer = []
        # for i in range(Der.shape[0]):
        #    Answer.append(Der[i].flatten())
        # return np.array(Answer)

    @property
    def BMatrix(self):
        return self.coord_obj.Prims.wilsonB(self.xyz)

    @property
    def WilsonB(self):
        return self.coord_obj.wilsonB(self.xyz)

    @property
    def num_coordinates(self):
        return len(self.coordinates)


if __name__ == '__main__':
    from level_of_theories import Molpro
    filepath = '../../data/ethylene.xyz'
    molpro = Molpro.from_options(states=[(1, 0)], fnm=filepath, lot_inp_file='../../data/ethylene_molpro.com')

    pes = PES.from_options(lot=molpro, ad_idx=0, multiplicity=1)

    reactant = Molecule.from_options(fnm=filepath, PES=pes, coordinate_type="TRIC", Form_Hessian=False)

    print(reactant.coord_basis)
