"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""

from time import time
import numpy as np
import os
# from collections import Counter

from ..coordinate_systems import InternalCoordinates, construct_coordinate_system, CartesianCoordinates
from ..potential_energy_surfaces import PES, Penalty_PES, Avg_PES
from ..utilities import manage_xyz, elements, options, block_matrix, Devutils as dev, units

ELEMENT_TABLE = elements.ElementData()

# TOC:
# constructors
# methods
# properties

"""Specify a molecule by its atom composition, coordinates, charge
and spin multiplicities.
"""


class Molecule:
    _default_options = None
    @classmethod
    def default_options(cls):
        if cls._default_options is not None:
            return cls._default_options.copy()

        opt = options.Options()

        opt.add_option(
            key='coord_obj',
            required=False,
            value=None,
            #allowed_types=[DelocalizedInternalCoordinates,CartesianCoordinates],
                doc='A coordinate object.'
        )

        opt.add_option(
            key='geom',
            required=False,
            allowed_types=[list],
            doc='geometry including atomic symbols'
        )

        opt.add_option(
            key='xyz',
            required=False,
            allowed_types=[np.ndarray],
            doc='The Cartesian coordinates in Angstrom'
        )

        opt.add_option(
            key='Primitive_Hessian',
            value=None,
            required=False,
            doc='Primitive hessian save file for doing optimization.'
        )

        opt.add_option(
            key='Hessian',
            value=None,
            required=False,
            doc='Hessian save file in the basis of coordinate_type.'
        )

        opt.add_option(
            key='Form_Hessian',
            value=True,
            doc='Form the Hessian in the current basis -- takes time for large molecules.'
        )

        opt.add_option(
            key="top_settings",
            value={},
            doc='some extra kwargs for forming coordinate object.'
        )

        opt.add_option(
            key='comment',
            required=False,
            value='',
            doc='A string that is saved on the molecule, used for descriptive purposes'
        )

        opt.add_option(
            key='node_id',
            required=False,
            value=0,
            doc='used to specify level of theory node identification',
        )

        opt.add_option(
            key='frozen_atoms',
            required=False,
            value=None,
            doc='frozen atoms',
        )
        cls._default_options = opt
        return cls._default_options.copy()

    @classmethod
    def from_options(cls, options):
        return cls(**options)

    @staticmethod
    def copy_from_options(MoleculeA, xyz=None, fnm=None, new_node_id=1, copy_wavefunction=True):
        """Create a copy of MoleculeA"""
        print(" Copying from MoleculA {}".format(MoleculeA.node_id))
        PES = type(MoleculeA.PES).create_pes_from(PES=MoleculeA.PES, options={'node_id': new_node_id})

        if xyz is not None:
            new_geom = manage_xyz.np_to_xyz(MoleculeA.geometry, xyz)
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy().set_values({"xyz": xyz}))
        elif fnm is not None:
            new_geom = manage_xyz.read_xyz(fnm, scale=1.)
            xyz = manage_xyz.xyz_to_np(new_geom)
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy().set_values({"xyz": xyz}))
        else:
            new_geom = MoleculeA.geometry
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy())

        return Molecule(MoleculeA.Data.copy().set_values({
            'PES': PES,
            'coord_obj': coord_obj,
            'geom': new_geom,
            'node_id': new_node_id,
            'copy_wavefunction': copy_wavefunction,
        }))

    def __init__(self,
                 coord_obj:InternalCoordinates,
                 *,
                 node_id=None,
                 comment="",
                 frozen_atoms=None,
                 logger=None,
                 energy=None,
                 gradient=None,
                 derivative_coupling=None,
                 energy_evaluator=None,
                 hessian=None,
                 primitive_hessian=None,
                 energy_evaluator_options=None
                 ):

        atoms = coord_obj.atoms
        self.xyz = coord_obj.xyz

        self.logger = dev.Logger.lookup(logger)
        if not np.issubdtype(atoms.dtype, np.str_):
            raise ValueError(f"list of atom strings required, got {atoms}")

        if self.xyz.shape[-1] != 3 or len(atoms) != self.xyz.shape[-2]:
            raise ValueError(f"expected `xyz` to be {len(atoms)}x3, got {self.xyz.shape}")

        # create a dictionary from atoms
        # atoms contain info you need to know about the atoms
        self.atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        self.frozen_atoms = frozen_atoms

        self._lot = energy_evaluator
        self._lot_options = energy_evaluator_options
        self._energy_expansion = None

        self.comment = comment
        self.node_id = node_id
        self.coord_obj = coord_obj

        #TODO
        self.gradrms = 0.
        self.isTSnode = False
        self.bdist = 0.
        self.newHess = 5

        self._hessian = hessian
        self._primitive_hessian = primitive_hessian
        self._energy = None
        self._gradient = None
        self._derivative_coupling = None

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
                 **opts):
        if coordinate_system_options is None:
            coordinate_system_options = {}
        coord_sys = construct_coordinate_system(atoms, xyz,
                                                bonds=bonds,
                                                primitives=primitives,
                                                internals=internals,
                                                coordinate_type=coordinate_type,
                                                **coordinate_system_options
                                                )

        return cls(coord_sys, **opts)

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

    @property
    def geometry(self):
        return manage_xyz.combine_atom_xyz(self.atom_symbols, self.xyz)

    @property
    def atom_symbols(self):
        return [a.symbol for a in self.atoms]

    @property
    def energy(self):
        return self.evaluator.get_energy(self.xyz)
        #return 0.

    @property
    def gradx(self):
        return np.reshape(self.PES.get_gradient(self.xyz, frozen_atoms=self.frozen_atoms), (-1, 3))

    @property
    def gradient(self):
        gradx = self.PES.get_gradient(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, gradx)  # CartesianCoordinate just returns gradx

    # for PES seams
    @property
    def avg_gradient(self):
        gradx = self.PES.get_avg_gradient(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, gradx)  # CartesianCoordinate just returns gradx

    @property
    def derivative_coupling(self):
        dvecx = self.PES.get_coupling(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, dvecx)

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
            if not isinstance(self.coord_obj, CartesianCoordinates):
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
        prim = self.coord_obj.Prims.guess_hessian(self.xyz)
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

    @property
    def xyz(self):
        return self._coords
    @xyz.setter
    def xyz(self, newxyz):
        if newxyz is not None:
            #TODO: check whether or not I should actually invalidate the cache
            self._coords = newxyz
            self._energy = None
            self._gradient = None
            self._primitive_hessian = None
            self._hessian = None

    @property
    def num_frozen_atoms(self):
        if self.frozen_atoms is not None:
            return len(self.frozen_atoms)
        else:
            return 0

    def update_xyz(self, dq=None, verbose=True):
        #print " updating xyz"
        if dq is not None:
            self.xyz = self.coord_obj.newCartesian(self.xyz, dq, frozen_atoms=self.frozen_atoms, verbose=verbose)
        return self.xyz

    def update_and_move(self, tan, deltadq, verbose=True):
        self.update_coordinate_basis(tan)
        dq = deltadq*self.constraint[:, 0]
        self.update_xyz(dq, verbose=verbose)

    def update_MW_xyz(self, mass, dq, verbose=True):
        self.xyz = self.coord_obj.massweighted_newCartesian(self.xyz, dq, mass, verbose)
        return self.xyz

    @property
    def finiteDifferenceHessian(self):
        return self.PES.get_finite_difference_hessian(self.xyz)

    @property
    def primitive_internal_coordinates(self):
        return self.coord_obj.Prims.Internals

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

    @property
    def node_id(self):
        return self._node_id

    @node_id.setter
    def node_id(self, value):
        self._node_id = value
        self.PES.lot.node_id = value


if __name__ == '__main__':
    from level_of_theories import Molpro
    filepath = '../../data/ethylene.xyz'
    molpro = Molpro.from_options(states=[(1, 0)], fnm=filepath, lot_inp_file='../../data/ethylene_molpro.com')

    pes = PES.from_options(lot=molpro, ad_idx=0, multiplicity=1)

    reactant = Molecule.from_options(fnm=filepath, PES=pes, coordinate_type="TRIC", Form_Hessian=False)

    print(reactant.coord_basis)
