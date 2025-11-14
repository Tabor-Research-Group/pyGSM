from __future__ import print_function
# standard library imports
import abc
import enum

from ..coordinate_systems import Distance, Angle, Dihedral, OutOfPlane
from .. import utilities as util #import nifty, options, block_matrix
from ..utilities import manage_xyz, Devutils as dev
from pyGSM.molecule import Molecule
import tempfile

# third party
import numpy as np
import multiprocessing as mp
from collections import Counter
from copy import copy
from itertools import chain


def worker(arg):
   obj, methname = arg[:2]
   return getattr(obj, methname)(*arg[2:])


#######################################################################################
#### This class contains the main constructor, object properties and staticmethods ####
#######################################################################################


# TODO interpolate is still sloppy. It shouldn't create a new molecule node itself
# but should create the xyz. GSM should create the new molecule based off that xyz.
# TODO nconstraints in ic_reparam and write_iters is irrelevant

class GrowthDirectionType(enum.Enum):
     Normal = 0
     Reactant = 1

class ReparametrizationMethod(enum.Enum):
    Geodesic = "Geodesic"
    DelocalizedCoordinate = "DLC"

class GSM(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_active(self, nR, nP):
        ...

    @abc.abstractmethod
    def go_gsm(self, max_iters=50, opt_steps=10, **etc):
        ...

    @abc.abstractmethod
    def grow_nodes(self):
        ...

    _default_options = None
    @classmethod
    def default_options(cls):
        if cls._default_options is not None:
            return cls._default_options.copy()

        opt = util.options.Options()

        opt.add_option(
            key='reactant',
            required=True,
            # allowed_types=[Molecule,wrappers.Molecule],
            doc='Molecule object as the initial reactant structure')

        opt.add_option(
            key='product',
            required=False,
            # allowed_types=[Molecule,wrappers.Molecule],
            doc='Molecule object for the product structure (not required for single-ended methods.')

        opt.add_option(
            key='nnodes',
            required=False,
            value=1,
            allowed_types=[int],
            doc="number of string nodes"
        )

        opt.add_option(
            key='optimizer',
            required=True,
            doc='Optimzer object  to use e.g. eigenvector_follow, conjugate_gradient,etc. \
                        most of the default options are okay for here since GSM will change them anyway',
        )

        opt.add_option(
            key='driving_coords',
            required=False,
            value=[],
            allowed_types=[list],
            doc='Provide a list of tuples to select coordinates to modify atoms\
                 indexed at 1')

        opt.add_option(
            key='CONV_TOL',
            value=0.0005,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='CONV_gmax',
            value=0.001,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='CONV_Ediff',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='CONV_dE',
            value=0.5,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='ADD_NODE_TOL',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
            key="growth_direction",
            value=0,
            required=False,
            doc="how to grow string,0=Normal,1=from reactant"
        )

        opt.add_option(
            key="DQMAG_MAX",
            value=0.8,
            required=False,
            doc="max step along tangent direction for SSM"
        )
        opt.add_option(
            key="DQMAG_MIN",
            value=0.2,
            required=False,
            doc=""
        )

        opt.add_option(
            key='print_level',
            value=1,
            required=False
        )

        opt.add_option(
            key='xyz_writer',
            value=manage_xyz.write_molden_geoms,
            required=False,
            doc='Function to be used to format and write XYZ files',
        )

        opt.add_option(
            key='mp_cores',
            value=1,
            doc='multiprocessing cores for parallel programming. Use this with caution.',
        )

        opt.add_option(
            key="BDIST_RATIO",
            value=0.5,
            required=False,
            doc="SE-Crossing uses this \
                        bdist must be less than 1-BDIST_RATIO of initial bdist in order to be \
                        to be considered grown.",
        )

        opt.add_option(
            key='ID',
            value=0,
            required=False,
            doc='A number for identification of Strings'
        )

        opt.add_option(
            key='interp_method',
            value='DLC',
                allowed_values=['Geodesic', 'DLC'],
            required=False,
            doc='Which reparameterization method to use',
        )

        opt.add_option(
            key='noise',
            value=100.0,
            allowed_types=[float],
            required=False,
            doc='Noise to check for intermediate',
        )

        cls._default_options = opt
        return cls._default_options.copy()

    @classmethod
    def get_default_tolerances(cls):
        # subclasses should inherit from this
        return dict(
            ADD_NODE_TOL=0.1,
            CONV_dE=0.5,
            CONV_Ediff=0.1,
            CONV_gmax=0.001,
            DQMAG_MAX=0.8,
            DQMAG_MIN=0.2,
            BDIST_RATIO=0.5
        )

    def __init__(
            self,
            *,
            optimizer,
            driving_coords=None,
            reactant=None,
            product=None,
            nodes:'list[Molecule]'=None,
            nnodes=None,
            scratch_dir=None,
            growth_direction=0,
            tolerances=None,
            interp_method="DLC",
            id=None,
            noise=100,
            mp_cores=None,
            xyz_writer=None,
            logger=True
    ):
        self.scratch_dir = scratch_dir
        if nodes is None:
            if reactant is None:
                raise ValueError("`reactant` is required if no nodes are supplied")
            # if product is None:
            #     raise ValueError("`product` is required if no nodes are supplied")
            nodes = [None]*nnodes
            nodes[0] = reactant
            nodes[-1] = product
        self.nodes = nodes
        self.driving_coords = driving_coords
        self.growth_direction = GrowthDirectionType(growth_direction)
        self.isRestarted = False
        if tolerances is None:
            tolerances = {}
        self.tolerances = dict(self.get_default_tolerances(), **tolerances)
        self.ID = id
        self.optimizer = []
        self.interp_method = ReparametrizationMethod(interp_method)
        self.noise = noise
        self.mp_cores = mp_cores
        self.xyz_writer = xyz_writer
        self.logger = dev.Logger.lookup(logger)
        self.optimizer = optimizer

        # Set initial values
        self.current_nnodes = len([x for x in self.nodes if x is not None])
        # TODO: figure this out
        self.nR = 1
        self.nP = 1
        self.climb = False
        self.find = False
        self.ts_exsteps = 3  # multiplier for ts node
        self.n0 = 1  # something to do with added nodes? "first node along current block"
        self.end_early = False
        self.tscontinue = True  # whether to continue with TS opt or not
        self.found_ts = False
        self.rn3m6 = self._get_num_coords()

        self.ictan = [None] * self.nnodes
        self.active = [False] * self.nnodes
        self.climber = False  # is this string a climber?
        self.finder = False   # is this string a finder?
        self.done_growing = False
        self.nclimb = 0
        self.nhessreset = 10  # are these used??? TODO
        self.hessrcount = 0   # are these used?!  TODO
        self.hess_counter = 0   # it is probably good to reset the hessian
        self.newclimbscale = 2.
        self.TS_E_0 = None
        self.dE_iter = 100.  # change in max TS node

        self.nopt_intermediate = 0     # might be a duplicate of endearly_counter
        self.flag_intermediate = False
        self.endearly_counter = 0  # Find the intermediate x time
        self.pot_min = []
        self.ran_out = False   # if it ran out of iterations

        self.newic = Molecule.copy_from_options(self.nodes[0])  # newic object is used for coordinate transformations

    @property
    def reactant(self):
        return self.nodes[0]
    @property
    def product(self):
        return self.nodes[-1]
    @property
    def nnodes(self):
        return len(self.nodes)

    @classmethod
    def from_options(cls, options:util.options.Options):
        return cls(**options)

    @property
    def _get_num_coords(self):
        return 3.*self.nodes[0].natoms-6.

    @property
    def TSnode(self):
        '''
        The current node with maximum energy
        '''
        # Treat GSM with penalty a little different since penalty will increase energy based on energy
        # make sure TS is not zero or last node
        return int(np.argmax(self.energies[1:self.nnodes-1])+1)

    @property
    def emax(self):
        return self.energies[self.TSnode]

    @property
    def npeaks(self):
        '''
        '''
        minnodes = []
        maxnodes = []
        energies = self.energies
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[self.nnodes-1] < energies[self.nnodes-2]:
            minnodes.append(self.nnodes-1)
        for n in range(self.n0, self.nnodes-1):
            if energies[n+1] > energies[n]:
                if energies[n] < energies[n-1]:
                    minnodes.append(n)
            if energies[n+1] < energies[n]:
                if energies[n] > energies[n-1]:
                    maxnodes.append(n)

        return len(maxnodes)

    @property
    def energies(self):
        '''
        Energies of string
        '''
        E = []
        for ico in self.nodes:
            if ico is not None:
                E.append(ico.energy - self.nodes[0].energy)
        return E

    @energies.setter
    def energies(self, list_of_E):
        '''
        setter for energies
        '''
        self.E = list_of_E

    @property
    def geometries(self):
        geoms = []
        for ico in self.nodes:
            if ico is not None:
                geoms.append(ico.geometry)
        return geoms

    @property
    def gradrmss(self):
        self._gradrmss = []
        for ico in self.nodes:
            if ico is not None:
                self._gradrmss.append(ico.gradrms)
        return self._gradrmss

    @property
    def dEs(self):
        self._dEs = []
        for ico in self.nodes:
            if ico is not None:
                self._dEs.append(ico.difference_energy)
        return self._dEs

    @property
    def ictan(self):
        return self._ictan

    @ictan.setter
    def ictan(self, value):
        self._ictan = value

    @property
    def dqmaga(self):
        return self._dqmaga

    @dqmaga.setter
    def dqmaga(self, value):
        self._dqmaga = value

    @staticmethod
    def add_xyz_along_tangent(
            xyz1,
            constraints,
            step,
            coord_obj,
    ):
        dq0 = step*constraints
        new_xyz = coord_obj.newCartesian(xyz1, dq0)

        return new_xyz

    @classmethod
    def add_node(
            cls,
            nodeR,
            nodeP,
            stepsize,
            node_id,
            driving_coords=None,
            DQMAG_MAX=0.8,
            DQMAG_MIN=0.2,
            logger=None,
    ):
        '''
        Add a node between  nodeR and nodeP or if nodeP is none use driving coordinate to add new node
        '''

        logger = dev.Logger.lookup(logger)
        if nodeP is None:

            if driving_coords is None:
                raise ValueError("You didn't supply a driving coordinate and product node is None!")

            BDISTMIN = 0.05
            ictan, bdist = cls.get_tangent(nodeR, None, driving_coords=driving_coords)

            if bdist < BDISTMIN:
                logger.log_print("bdist too small {bdist:.3f}", bdist=bdist)
                return None
            new_node = Molecule.copy_from_options(nodeR, new_node_id=node_id)
            new_node.update_coordinate_basis(constraints=ictan)
            constraint = new_node.constraints[:, 0]
            sign = -1.

            dqmag_scale = 1.5
            minmax = DQMAG_MAX - DQMAG_MIN
            a = bdist/dqmag_scale
            if a > 1.:
                a = 1.
            dqmag = sign*(DQMAG_MIN+minmax*a)
            if dqmag > DQMAG_MAX:
                dqmag = DQMAG_MAX
            logger.log_print(" dqmag: {dqmag:%4.3f} from bdist: {bdist:%4.3f}", dqmag=dqmag, bdist=bdist)

            dq0 = dqmag*constraint
            logger.log_print(" dq0[constraint]: %1.3f", dqmag=dqmag)

            new_node.update_xyz(dq0)
            new_node.bdist = bdist

        else:
            ictan, _ = cls.get_tangent(nodeR, nodeP)
            nodeR.update_coordinate_basis(constraints=ictan)
            constraint = nodeR.constraints[:, 0]
            dqmag = np.linalg.norm(ictan)
            logger.log_print(" dqmag: %1.3f", dqmag=dqmag)
            # sign=-1
            sign = 1.
            dqmag *= (sign*stepsize)
            logger.log_print(" scaled dqmag: %1.3f", dqmag=dqmag)

            dq0 = dqmag*constraint
            old_xyz = nodeR.xyz.copy()
            new_xyz = nodeR.coord_obj.newCartesian(old_xyz, dq0)
            new_node = Molecule.copy_from_options(MoleculeA=nodeR, xyz=new_xyz, new_node_id=node_id)

        return new_node

    @classmethod
    def interpolate_xyz(cls, nodeR, nodeP, stepsize, logger=None):
        '''
        Interpolate between two nodes
        '''

        logger = dev.Logger.lookup(logger)
        ictan, _ = cls.get_tangent(nodeR, nodeP)
        Vecs = nodeR.update_coordinate_basis(constraints=ictan)
        constraint = nodeR.constraints[:, 0]
        prim_constraint = util.block_matrix.dot(Vecs, constraint)
        dqmag = np.dot(prim_constraint.T, ictan)
        print(" dqmag: %1.3f" % dqmag)
        # sign=-1
        sign = 1.
        dqmag *= (sign*stepsize)
        logger.log_print(" scaled dqmag: {dqmag:1.3f}", dqmag=dqmag)

        dq0 = dqmag*constraint
        old_xyz = nodeR.xyz.copy()
        new_xyz = nodeR.coord_obj.newCartesian(old_xyz, dq0)

        return new_xyz

    @classmethod
    def interpolate(cls, start_node, end_node, num_interp, logger=None):
        logger = dev.Logger.lookup(logger)
        with logger.block(tag="interpolate"):

            num_nodes = num_interp + 2
            nodes = [None]*(num_nodes)
            nodes[0] = start_node
            nodes[-1] = end_node
            sign = 1
            nR = 1
            nP = 1
            nn = nR + nP

            for n in range(num_interp):
                if num_nodes - nn > 1:
                    stepsize = 1./float(num_nodes - nn)
                else:
                    stepsize = 0.5
                if sign == 1:
                    iR = nR-1
                    iP = num_nodes - nP
                    iN = nR
                    nodes[nR] = cls.add_node(nodes[iR], nodes[iP], stepsize, iN)
                    if nodes[nR] is None:
                        raise RuntimeError

                    # print(" Energy of node {} is {:5.4}".format(nR,nodes[nR].energy-E0))
                    nR += 1
                    nn += 1

                else:
                    n1 = num_nodes - nP
                    n2 = n1 - 1
                    n3 = nR - 1
                    nodes[n2] = cls.add_node(nodes[n1], nodes[n3], stepsize, n2)
                    if nodes[n2] is None:
                        raise RuntimeError
                    # print(" Energy of node {} is {:5.4}".format(nR,nodes[nR].energy-E0))
                    nP += 1
                    nn += 1
                sign *= -1

            return nodes

    @staticmethod
    def get_tangent_xyz(xyz1, xyz2, prim_coords):
        PMDiff = np.zeros(len(prim_coords))
        for k, prim in enumerate(prim_coords):
            if type(prim) is Distance:
                PMDiff[k] = 2.5 * prim.calcDiff(xyz2, xyz1)
            else:
                PMDiff[k] = prim.calcDiff(xyz2, xyz1)
        return np.reshape(PMDiff, (-1, 1))

    @classmethod
    def get_tangent(cls, node1, node2, *, driving_coords, logger=None):
        '''
        Get internal coordinate tangent between two nodes, assumes they have unique IDs
        '''
        logger = dev.Logger.lookup(logger)

        if node2 is not None and node1.node_id != node2.node_id:
            logger.log_print(
                " getting tangent from between {node2} {node1} pointing towards {node2}",
                node2=node2.node_id,
                node1=node1.node_id
            )

            PMDiff = np.zeros(node2.num_primitives)
            for k, prim in enumerate(node2.primitive_internal_coordinates):
                if type(prim) is Distance:
                    PMDiff[k] = 2.5 * prim.calcDiff(node2.xyz, node1.xyz)
                else:
                    PMDiff[k] = prim.calcDiff(node2.xyz, node1.xyz)

            return np.reshape(PMDiff, (-1, 1)), None
        else:
            logger.log_print(
                " getting tangent from node {node}",
                node=node1.node_id
            )

            c = Counter(elem[0] for elem in driving_coords)
            #TODO: why aren't these used?
            nadds = c['ADD']
            nbreaks = c['BREAK']
            nangles = c['nangles']
            ntorsions = c['ntorsions']

            ictan = np.zeros((node1.num_primitives, 1), dtype=float)
            # breakdq = 0.3
            bdist = 0.0
            atoms = node1.atoms
            xyz = node1.xyz.copy()

            for i in driving_coords:
                if "ADD" in i:

                    # order indices to avoid duplicate bonds
                    if i[1] < i[2]:
                        index = [i[1]-1, i[2]-1]
                    else:
                        index = [i[2]-1, i[1]-1]

                    bond = Distance(index[0], index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Distance')
                    if len(i) == 3:
                        # TODO why not just use the covalent radii?
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2.8
                    elif len(i) == 4:
                        d0 = i[3]
                    current_d = bond.value(xyz)

                    # TODO don't set tangent if value is too small
                    ictan[prim_idx] = -1*(d0-current_d)
                    # if nbreaks>0:
                    #    ictan[prim_idx] *= 2
                    # => calc bdist <=
                    if current_d > d0:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    logger.log_print(
                        " bond {bond} target (less than): {d0:4.3f} current d: {current_d:4.3f} diff: {ict:4.3f}",
                        bond=(i[1], i[2]),
                        d0=d0,
                        current_d=current_d,
                        ict=ictan[prim_idx]
                    )

                elif "BREAK" in i:
                    # order indices to avoid duplicate bonds
                    if i[1] < i[2]:
                        index = [i[1]-1, i[2]-1]
                    else:
                        index = [i[2]-1, i[1]-1]
                    bond = Distance(index[0], index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Distance')
                    if len(i) == 3:
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)
                    elif len(i) == 4:
                        d0 = i[3]

                    current_d = bond.value(xyz)
                    ictan[prim_idx] = -1*(d0-current_d)

                    # => calc bdist <=
                    if current_d < d0:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])

                    logger.log_print(
                        " bond {bond} target (greater than): {d0:4.3f} current d: {current_d:4.3f} diff: {ict:4.3f}",
                        bond=(i[1], i[2]),
                        d0=d0,
                        current_d=current_d,
                        ict=ictan[prim_idx]
                    )
                elif "ANGLE" in i:

                    if i[1] < i[3]:
                        index = [i[1]-1, i[2]-1, i[3]-1]
                    else:
                        index = [i[3]-1, i[2]-1, i[1]-1]
                    angle = Angle(index[0], index[1], index[2])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Angle')
                    anglet = i[4]
                    ang_value = angle.value(xyz)
                    ang_diff = anglet*np.pi/180. - ang_value
                    # print(" angle: %s is index %i " %(angle,ang_idx))
                    logger.log_print(
                        " anglev: {ang_value:4.3f} align to {anglet:4.3f} diff(rad): {ang_diff:4.3f}",
                        ang_value=ang_value,
                        anglet=anglet,
                        ang_diff=ang_diff
                    )
                    ictan[prim_idx] = -ang_diff
                    # TODO need to come up with an adist
                    # if abs(ang_diff)>0.1:
                    #    bdist+=ictan[ICoord1.BObj.nbonds+ang_idx]*ictan[ICoord1.BObj.nbonds+ang_idx]
                elif "TORSION" in i:

                    if i[1] < i[4]:
                        index = [i[1]-1, i[2]-1, i[3]-1, i[4]-1]
                    else:
                        index = [i[4]-1, i[3]-1, i[2]-1, i[1]-1]
                    torsion = Dihedral(index[0], index[1], index[2], index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Dihedral')
                    tort = i[5]
                    torv = torsion.value(xyz)
                    tor_diff = tort - torv*180./np.pi
                    if tor_diff > 180.:
                        tor_diff -= 360.
                    elif tor_diff < -180.:
                        tor_diff += 360.
                    ictan[prim_idx] = -tor_diff*np.pi/180.

                    if tor_diff*np.pi/180. > 0.1 or tor_diff*np.pi/180. < 0.1:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])

                    logger.log_print(
                        " current torv: {torv:4.3f} align to {tort:4.3f} diff(deg): {tor_diff:4.3f}",
                        torv=torv*180./np.pi,
                        tort=tort,
                        tor_diff=tor_diff
                    )

                elif "OOP" in i:
                    index = [i[1]-1, i[2]-1, i[3]-1, i[4]-1]
                    oop = OutOfPlane(index[0], index[1], index[2], index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'OutOfPlane')
                    oopt = i[5]
                    oopv = oop.value(xyz)
                    oop_diff = oopt - oopv*180./np.pi
                    if oop_diff > 180.:
                        oop_diff -= 360.
                    elif oop_diff < -180.:
                        oop_diff += 360.
                    ictan[prim_idx] = -oop_diff*np.pi/180.

                    if oop_diff*np.pi/180. > 0.1 or oop_diff*np.pi/180. < 0.1:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])

                    logger.log_print(
                        " current oopv: {oopv:4.3f} align to {oopt:4.3f} diff(deg): {oop_diff:4.3f}",
                        oopv=oopv*180./np.pi,
                        oopt=oopt,
                        oop_diff=oop_diff
                    )

            bdist = np.sqrt(bdist)
            if np.all(ictan == 0.0):
                raise ValueError(" All elements are zero")
            return ictan, bdist

    @classmethod
    def get_tangents(cls, nodes, n0=0, print_level=0):
        '''
        Get the normalized internal coordinate tangents and magnitudes between all nodes
        '''
        nnodes = len(nodes)
        dqmaga = [0.]*nnodes
        ictan = [[]]*nnodes

        for n in range(n0+1, nnodes):
            # print "getting tangent between %i %i" % (n,n-1)
            assert nodes[n] is not None, "n is bad"
            assert nodes[n-1] is not None, "n-1 is bad"
            ictan[n] = cls.get_tangent_xyz(nodes[n-1].xyz, nodes[n].xyz, nodes[0].primitive_internal_coordinates)

            dqmaga[n] = 0.
            # ictan0= np.copy(ictan[n])
            dqmaga[n] = np.linalg.norm(ictan[n])

            ictan[n] /= dqmaga[n]

            # NOTE:
            # vanilla GSM has a strange metric for distance
            # no longer following 7/1/2020
            # constraint = self.newic.constraints[:,0]
            # just a fancy way to get the normalized tangent vector
            # prim_constraint = block_matrix.dot(Vecs,constraint)
            # for prim in self.newic.primitive_internal_coordinates:
            #    if type(prim) is Distance:
            #        index = self.newic.coord_obj.Prims.dof_index(prim)
            #        prim_constraint[index] *= 2.5
            # dqmaga[n] = float(np.dot(prim_constraint.T,ictan0))
            # dqmaga[n] = float(np.sqrt(dqmaga[n]))
            if dqmaga[n] < 0.:
                raise RuntimeError

        # TEMPORORARY parallel idea
        # ictan = [0.]
        # ictan += [ Process(target=get_tangent,args=(n,)) for n in range(n0+1,self.nnodes)]
        # dqmaga = [ Process(target=get_dqmag,args=(n,ictan[n])) for n in range(n0+1,self.nnodes)]

        # if print_level > 1:
        #     print('------------printing ictan[:]-------------')
        #     for n in range(n0+1, nnodes):
        #         print("ictan[%i]" % n)
        #         print(ictan[n].T)
        # if print_level > 0:
        #     print('------------printing dqmaga---------------')
        #     for n in range(n0+1, nnodes):
        #         print(" {:5.4}".format(dqmaga[n]), end='')
        #         if (n) % 5 == 0:
        #             print()
        #     print()
        return ictan, dqmaga

    @classmethod
    def get_three_way_tangents(cls, nodes, energies, find=True, n0=0, logger=None):
        '''
        Calculates internal coordinate tangent with a three-way tangent at TS node
        '''
        logger = dev.Logger.lookup(logger)

        nnodes = len(nodes)
        ictan = [[]]*nnodes
        dqmaga = [0.]*nnodes
        # TSnode = np.argmax(energies[1:nnodes-1])+1
        TSnode = np.argmax(energies)   # allow for the possibility of TS node to be endpoints?

        last_node_max = (TSnode == nnodes-1)
        first_node_max = (TSnode == 0)
        if first_node_max or last_node_max:
            logger.log_print("*********** This will cause a range error in the following for loop *********")
            logger.log_print("** Setting the middle of the string to be TS node to get proper directions **")
            TSnode = nnodes//2

        for n in range(n0, nnodes):
            do3 = False
            logger.log_print('getting tan[{{{n}}]', n=n)
            if n < TSnode:
                # The order is very important here
                # the way it should be ;(
                intic_n = n+1
                newic_n = n

                # old way
                # intic_n = n
                # newic_n = n+1

            elif n > TSnode:
                # The order is very important here
                intic_n = n
                newic_n = n-1
            else:
                do3 = True
                newic_n = n
                intic_n = n+1
                int2ic_n = n-1

            if do3:
                if first_node_max or last_node_max:
                    t1, _ = cls.get_tangent(nodes[intic_n], nodes[newic_n])
                    t2, _ = cls.get_tangent(nodes[newic_n], nodes[int2ic_n])
                    logger.log_print(" done 3 way tangent")
                    ictan0 = t1 + t2
                else:
                    f1 = 0.
                    dE1 = abs(energies[n+1]-energies[n])
                    dE2 = abs(energies[n] - energies[n-1])
                    dEmax = max(dE1, dE2)
                    dEmin = min(dE1, dE2)
                    if energies[n+1] > energies[n-1]:
                        f1 = dEmax/(dEmax+dEmin+0.00000001)
                    else:
                        f1 = 1 - dEmax/(dEmax+dEmin+0.00000001)

                    logger.log_print(' 3 way tangent ({n}): f1:{f1:3.2}', n=n, f1=f1)

                    t1, _ = cls.get_tangent(nodes[intic_n], nodes[newic_n])
                    t2, _ = cls.get_tangent(nodes[newic_n], nodes[int2ic_n])
                    logger.log_print(" done 3 way tangent")
                    ictan0 = f1*t1 + (1.-f1)*t2
            else:
                ictan0, _ = cls.get_tangent(nodes[newic_n], nodes[intic_n])

            ictan[n] = ictan0/np.linalg.norm(ictan0)
            dqmaga[n] = np.linalg.norm(ictan0)

        return ictan, dqmaga

    @classmethod
    def ic_reparam(cls, nodes, energies, climbing=False, ic_reparam_steps=8, print_level=1, NUM_CORE=1, MAXRE=0.25,
                   logger=None):
        '''
        Reparameterizes the string using Delocalizedin internal coordinatesusing three-way tangents at the TS node
        Only pushes nodes outwards during reparameterization because otherwise too many things change.
            Be careful, however, if the path is allup or alldown then this can cause
        Parameters
        ----------
        nodes : list of molecule objects
        energies : list of energies in kcal/mol
        ic_reparam_steps : int max number of reparameterization steps
        print_level : int verbosity
        '''
        logger = dev.Logger.lookup(logger)
        with logger.block("reparametrizing string nodes"):
            nnodes = len(nodes)
            rpart = np.zeros(nnodes)
            for n in range(1, nnodes):
                rpart[n] = 1./(nnodes-1)
            deltadqs = np.zeros(nnodes)
            TSnode = np.argmax(energies)
            disprms = 100
            if ((TSnode == nnodes-1) or (TSnode == 0)) and climbing:
                raise ValueError(" TS node shouldn't be the first or last node")

            ideal_progress_gained = np.zeros(nnodes)
            if climbing:
                for n in range(1, TSnode):
                    ideal_progress_gained[n] = 1./(TSnode)
                for n in range(TSnode+1, nnodes):
                    ideal_progress_gained[n] = 1./(nnodes-TSnode-1)
                ideal_progress_gained[TSnode] = 0.
            else:
                for n in range(1, nnodes):
                    ideal_progress_gained[n] = 1./(nnodes-1)

            for i in range(ic_reparam_steps):

                ictan, dqmaga = cls.get_tangents(nodes)
                totaldqmag = np.sum(dqmaga)

                if climbing:
                    progress = np.zeros(nnodes)
                    progress_gained = np.zeros(nnodes)
                    h1dqmag = np.sum(dqmaga[:TSnode+1])
                    h2dqmag = np.sum(dqmaga[TSnode+1:nnodes])
                    logger.log_print(" h1dqmag, h2dqmag: {h1dqmag:3.2f} {h2dqmag:3.2f}",
                                     h1dqmag=h1dqmag, h2dqmag=h2dqmag)
                    progress_gained[:TSnode] = dqmaga[:TSnode]/h1dqmag
                    progress_gained[TSnode+1:] = dqmaga[TSnode+1:]/h2dqmag
                    progress[:TSnode] = np.cumsum(progress_gained[:TSnode])
                    progress[TSnode:] = np.cumsum(progress_gained[TSnode:])
                else:
                    progress = np.cumsum(dqmaga)/totaldqmag
                    progress_gained = dqmaga/totaldqmag

                if i == 0:
                    orig_dqmaga = copy(dqmaga)
                    orig_progress_gained = copy(progress_gained)

                if climbing:
                    difference = np.zeros(nnodes)
                    for n in range(TSnode):
                        difference[n] = ideal_progress_gained[n] - progress_gained[n]
                        deltadqs[n] = difference[n]*h1dqmag
                    for n in range(TSnode+1, nnodes):
                        difference[n] = ideal_progress_gained[n] - progress_gained[n]
                        deltadqs[n] = difference[n]*h2dqmag
                else:
                    difference = ideal_progress_gained - progress_gained
                    deltadqs = difference*totaldqmag

                ## TODO: add log info back in
                # logger.log_print(" ideal progress gained per step", end=' ')
                # for n in range(nnodes):
                #     logger.log_print(" step [{}]: {:1.3f}".format(n, ideal_progress_gained[n]))
                # print(" path progress                 ", end=' ')
                # for n in range(nnodes):
                #     print(" step [{}]: {:1.3f}".format(n, progress_gained[n]), end=' ')
                # print()
                # print(" difference                    ", end=' ')
                # for n in range(nnodes):
                #     print(" step [{}]: {:1.3f}".format(n, difference[n]), end=' ')
                # print()
                # print(" deltadqs                      ", end=' ')
                # for n in range(nnodes):
                #     print(" step [{}]: {:1.3f}".format(n, deltadqs[n]), end=' ')
                # print()

                # disprms = np.linalg.norm(deltadqs)/np.sqrt(nnodes-1)
                disprms = np.linalg.norm(deltadqs)/np.sqrt(nnodes-1)
                logger.log_print(" disprms: {disprms:1.3}\n", disprms=disprms)

                if disprms < 0.02:
                    break

                # Move nodes
                if climbing:
                    deltadqs[TSnode-2] -= deltadqs[TSnode-1]
                    deltadqs[nnodes-2] -= deltadqs[nnodes-1]
                    for n in range(1, nnodes-1):
                        if abs(deltadqs[n]) > MAXRE:
                            deltadqs[n] = np.sign(deltadqs[n])*MAXRE
                    for n in range(TSnode-1):
                        deltadqs[n+1] += deltadqs[n]
                    for n in range(TSnode+1, nnodes-2):
                        deltadqs[n+1] += deltadqs[n]
                    for n in range(nnodes):
                        if abs(deltadqs[n]) > MAXRE:
                            deltadqs[n] = np.sign(deltadqs[n])*MAXRE

                    if NUM_CORE > 1:

                        # 5/14/2021 TS node this up?!
                        tans = [ictan[n] if deltadqs[n] < 0 else ictan[n+1] for n in chain(range(1, TSnode), range(TSnode+1, nnodes-1))]  # + [ ictan[n] if deltadqs[n]<0 else ictan[n+1] for n in range(TSnode+1,nnodes-1)]
                        pool = mp.Pool(NUM_CORE)
                        Vecs = pool.map(worker, ((nodes[0].coord_obj, "build_dlc", node.xyz, tan) for node, tan in zip(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1], tans)))
                        pool.close()
                        pool.join()
                        for n, node in enumerate(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1]):
                            node.coord_basis = Vecs[n]

                        # move the positions
                        dqs = [deltadqs[n]*nodes[n].constraints[:, 0] for n in chain(range(1, TSnode), range(TSnode+1, nnodes-1))]
                        pool = mp.Pool(NUM_CORE)
                        newXyzs = pool.map(worker, ((node.coord_obj, "newCartesian", node.xyz, dq) for node, dq in zip(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1], dqs)))
                        pool.close()
                        pool.join()
                        for n, node in enumerate(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1]):
                            node.xyz = newXyzs[n]
                    else:
                        for n in chain(range(1, TSnode), range(TSnode+1, nnodes-1)):
                            if deltadqs[n] < 0:
                                # print(f" Moving node {n} along tan[{n}] this much {deltadqs[n]}")
                                logger.log_print(" Moving node {} along tan[{}] this much {}".format(n, n, deltadqs[n]))
                                nodes[n].update_coordinate_basis(ictan[n])
                                constraint = nodes[n].constraints[:, 0]
                                dq = deltadqs[n]*constraint
                                nodes[n].update_xyz(dq, verbose=(print_level > 1))
                            elif deltadqs[n] > 0:
                                logger.log_print(" Moving node {} along tan[{}] this much {}".format(n, n+1, deltadqs[n]))
                                nodes[n].update_coordinate_basis(ictan[n+1])
                                constraint = nodes[n].constraints[:, 0]
                                dq = deltadqs[n]*constraint
                                nodes[n].update_xyz(dq, verbose=(print_level > 1))
                else:
                    # e.g 11-2 = 9, deltadq[9] -= deltadqs[10]
                    deltadqs[nnodes-2] -= deltadqs[nnodes-1]
                    for n in range(1, nnodes-1):
                        if abs(deltadqs[n]) > MAXRE:
                            deltadqs[n] = np.sign(deltadqs[n])*MAXRE
                    for n in range(1, nnodes-2):
                        deltadqs[n+1] += deltadqs[n]
                    for n in range(1, nnodes-1):
                        if abs(deltadqs[n]) > MAXRE:
                            deltadqs[n] = np.sign(deltadqs[n])*MAXRE

                    if NUM_CORE > 1:
                        # Update the coordinate basis
                        tans = [ictan[n] if deltadqs[n] < 0 else ictan[n+1] for n in range(1, nnodes-1)]
                        pool = mp.Pool(NUM_CORE)
                        Vecs = pool.map(worker, ((nodes[0].coord_obj, "build_dlc", node.xyz, tan) for node, tan in zip(nodes[1:nnodes-1], tans)))
                        pool.close()
                        pool.join()
                        for n, node in enumerate(nodes[1:nnodes-1]):
                            node.coord_basis = Vecs[n]
                        # move the positions
                        dqs = [deltadqs[n]*nodes[n].constraints[:, 0] for n in range(1, nnodes-1)]
                        pool = mp.Pool(NUM_CORE)
                        newXyzs = pool.map(worker, ((node.coord_obj, "newCartesian", node.xyz, dq) for node, dq in zip(nodes[1:nnodes-1], dqs)))
                        pool.close()
                        pool.join()
                        for n, node in enumerate(nodes[1:nnodes-1]):
                            node.xyz = newXyzs[n]
                    else:
                        for n in range(1, nnodes-1):
                            if deltadqs[n] < 0:
                                # print(f" Moving node {n} along tan[{n}] this much {deltadqs[n]}")
                                logger.log_print(" Moving node {} along tan[{}] this much {}".format(n, n, deltadqs[n]))
                                nodes[n].update_coordinate_basis(ictan[n])
                                constraint = nodes[n].constraints[:, 0]
                                dq = deltadqs[n]*constraint
                                nodes[n].update_xyz(dq, verbose=(print_level > 1))
                            elif deltadqs[n] > 0:
                                logger.log_print(" Moving node {} along tan[{}] this much {}".format(n, n+1, deltadqs[n]))
                                nodes[n].update_coordinate_basis(ictan[n+1])
                                constraint = nodes[n].constraints[:, 0]
                                dq = deltadqs[n]*constraint
                                nodes[n].update_xyz(dq, verbose=(print_level > 1))

            if climbing:
                ictan, dqmaga = cls.get_tangents(nodes)
                h1dqmag = np.sum(dqmaga[:TSnode+1])
                h2dqmag = np.sum(dqmaga[TSnode+1:nnodes])
                logger.log_print(" h1dqmag, h2dqmag: {h1dqmag:3.2f} {h2dqmag:3.2f}", h1dqmag=h1dqmag, h2dqmag=h2dqmag)
                progress_gained[:TSnode] = dqmaga[:TSnode]/h1dqmag
                progress_gained[TSnode+1:] = dqmaga[TSnode+1:]/h2dqmag
                progress[:TSnode] = np.cumsum(progress_gained[:TSnode])
                progress[TSnode:] = np.cumsum(progress_gained[TSnode:])
            else:
                ictan, dqmaga = cls.get_tangents(nodes)
                totaldqmag = np.sum(dqmaga)
                progress = np.cumsum(dqmaga)/totaldqmag
                progress_gained = dqmaga/totaldqmag
            # print()
            # if print_level > 0:
            #     print(" ideal progress gained per step", end=' ')
            #     for n in range(nnodes):
            #         print(" step [{}]: {:1.3f}".format(n, ideal_progress_gained[n]), end=' ')
            #     print()
            #     print(" original path progress        ", end=' ')
            #     for n in range(nnodes):
            #         print(" step [{}]: {:1.3f}".format(n, orig_progress_gained[n]), end=' ')
            #     print()
            #     print(" reparameterized path progress ", end=' ')
            #     for n in range(nnodes):
            #         print(" step [{}]: {:1.3f}".format(n, progress_gained[n]), end=' ')
            #     print()

            logger.log_print(" spacings (begin ic_reparam, steps {spacings})",
                             spacings=[orig_dqmaga[n] for n in range(nnodes)],
                             preformatter=lambda *, spacings, **kw:dict(kw, spacings=" ".join("{:1.2}".format(s) for s in spacings))
                             )
            logger.log_print(
                [
                    " spacings (end ic_reparam, steps: {term}/{nstep}) {spacings}",
                    "disprms: {disprms:1.3}"
                ],
                term=i + 1,
                nstep=ic_reparam_steps,
                disprms=disprms,
                spacings=[dqmaga[n] for n in range(nnodes)],
                preformatter=lambda *, spacings, **kw: dict(kw, spacings=" ".join(
                    "{:1.2}".format(s) for s in spacings))
            )

    @classmethod
    def calc_optimization_metrics(cls, nodes, logger=None):
        logger = dev.Logger.lookup(logger)
        nnodes = len(nodes)
        rn3m6 = np.sqrt(3*nodes[0].natoms-6)
        totalgrad = 0.0
        gradrms = 0.0
        sum_gradrms = 0.0

        blocks = []
        subblock = []
        for i, ico in enumerate(nodes[1:nnodes-1]):
            if ico != None:
                subblock.append([i, float(ico.gradrms)])
                if i % 5 == 0:
                    blocks.append(subblock)
                    subblock = []
                totalgrad += ico.gradrms*rn3m6
                gradrms += ico.gradrms*ico.gradrms
                sum_gradrms += ico.gradrms
        blocks.append(subblock)

        logger.log_print("{fmt_grad}",
                         grad_data=blocks,
                         preformatter=lambda *, grad_data, **kw:dict(
                             kw,
                             fmt_grad="\n".join([
                                 " ".join(f"node: {i:02d} gradrms: {g:.6f}" for i,g in sub)
                                 for sub in grad_data
                                 ])
                         ))

        # TODO wrong for growth
        gradrms = np.sqrt(gradrms/(nnodes-2))
        return totalgrad, gradrms, sum_gradrms
