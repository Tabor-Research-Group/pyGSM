from __future__ import print_function
from ..utilities import manage_xyz, block_matrix, block_tensor

# standard library imports
import time

# third party
from copy import deepcopy,copy
import numpy as np

import itertools
from collections import OrderedDict, defaultdict
from scipy.linalg import block_diag

# local application import
from .internal_coordinates import InternalCoordinates, register_coordinate_system
from .topology import EdgeGraph, guess_bonds
from . import slots

CacheWarning = False

@register_coordinate_system("primitive")
class PrimitiveInternalCoordinates(InternalCoordinates):
    Internals: tuple[slots.PrimitiveCoordinate]

    _default_options = None
    @classmethod
    def default_options(cls):
        if cls._default_options is not None:
            return cls._default_options.copy()

        opt = super().default_options()

        opt.add_option(
            key='connect',
            value=False,
            allowed_types=[bool],
            doc="Connect the fragments/residues together with a minimum spanning bond,\
                            use for DLC, Don't use for TRIC, or HDLC.",
        )

        opt.add_option(
            key='addcart',
            value=False,
            allowed_types=[bool],
            doc="Add cartesian coordinates\
                            use to form HDLC ,Don't use for TRIC, DLC.",
        )

        opt.add_option(
            key='addtr',
            value=False,
            allowed_types=[bool],
            doc="Add translation and rotation coordinates\
                            use for TRIC.",
        )

        opt.add_option(
            key='constraints',
            value=None,
            allowed_types=[list],
            doc='A list of Distance,Angle,Torsion constraints (see slots.py),\
                            This is only useful if doing a constrained geometry optimization\
                            since GSM will handle the constraint automatically.'
        )
        opt.add_option(
            key='cVals',
            value=None,
            allowed_types=[list],
            doc='List of Distance,Angle,Torsion constraints values'
        )

        opt.add_option(
            key='form_topology',
            value=True,
            doc='A lazy argument for forming the topology on the fly, dont use this',
        )

        opt.add_option(
            key='topology',
            value=None,
            doc='This is the molecule topology, used for building primitives'
        )

        cls._default_options = opt
        return cls._default_options.copy()
    def __init__(self,
                 atoms,
                 xyz,
                 bonds=None,
                 constraints=None,
                 form_topology=True,
                 connect=False,
                 addcart=False,
                 addtr=False,
                 internals=None,
                 hybrid_idx_start_stop=None,
                 fragments=None,
                 block_info=None,
                 logger=None
                 ):

        super().__init__(atoms, xyz, bonds=bonds, logger=logger, constraints=constraints)

        self.connect = connect
        self.addcart = addcart
        self.addtr = addtr
        if addtr:
            if connect:
                raise RuntimeError(" Intermolecular displacements are defined by translation and rotations! \
                                    Don't add connect!")
        elif addcart:
            if connect:
                raise RuntimeError(" Intermolecular displacements are defined by cartesians! \
                                    Don't add connect!")
        else:
            pass

        # Cache some useful attributes
        self.atoms = atoms

        # initialize
        self.cPrims = []
        self.cVals = []
        self.Rotators = OrderedDict()
        self.natoms = len(self.atoms)
        self.built_bonds = False
        if hybrid_idx_start_stop is None:
            hybrid_idx_start_stop = []
        self.hybrid_idx_start_stop = hybrid_idx_start_stop

        # # Topology settings  -- CRA 3/2019 leftovers from Lee-Ping's code
        # but maybe useful in the future
        # self.top_settings = {
        #                    #'build_topology' : extra_kwargs.get('build_topology',True),
        #                    'make_primitives' : extra_kwargs.get('make_primitives',True),
        #                     }
        # bondfile = extra_kwargs.get('bondfile',None)

        # make_prims = self.top_settings['make_primitives']

        # setup
        if form_topology:
            if internals is not None:
                raise ValueError("`form_toplogy` will overwrite `internals`")

            self.fragments = self.topology.get_fragments()

            #ALEX CHANGE for lines 86-91
            self.get_hybrid_indices(xyz)
            #nifty.click()
            with self.logger.block(tag="Constructing primitives", log_level=self.logger.LogLevel.Debug):
                internals, self.block_info = self.newMakePrimitives(
                    atoms,
                    xyz,
                    self.topology,
                    fragments=self.fragments,
                    connect=connect,
                    addcart=addcart,
                    addtr=addtr,
                    hybrid_idx_start_stop=hybrid_idx_start_stop,
                    logger=self.logger
                )
            #time_build = nifty.click()
            #print(" make prim %.3f" % time_build)
        else:
            if internals is None:
                internals = []
            if fragments is None:
                fragments = []
            self.fragments = fragments
            if block_info is None:
                block_info = []
            self.block_info = block_info
        # Reorder primitives for checking with cc's code in TC.
        # Note that reorderPrimitives() _must_ be updated with each new InternalCoordinate class written.
        # self.reorderPrimitives()
        # time_reorder = nifty.click()
        # print("done reordering %.3f" % time_reorder)
        # self.makeConstraints(xyz, constraints, cvals)

        self.Internals = tuple(internals) # performance overhead, but need to catch attempted mutations

    @property
    def type_classes(self):
        return tuple(ic.type_class for ic in self.Internals)

    def copy(self):
        return type(self)(
            self.atoms,
            self.xyz,
            bonds=self.topology,
            internals=deepcopy(self.Internals),
            fragments=list(self.fragments),
            hybrid_idx_start_stop=self.hybrid_idx_start_stop,
            form_topology=False,
            connect=self.connect,
            addcart=self.addcart,
            addtr=self.addtr,
            logger=self.logger
        )

    def compute_bmatrix(self, xyz):
        xyz = xyz.reshape(-1, 3)

        Blist = []
        for info in self.block_info:
            sa = info[0]
            ea = info[1]
            sp = info[2]
            ep = info[3]
            Blist.append(
                np.array([p.derivative(xyz[sa:ea, :], start_idx=sa).flatten()
                          for p in self.Internals[sp:ep]])
            )

        ans = block_matrix(Blist)
        return ans

    # def calcGrad(self, xyz, gradx):
    #    #q0 = self.calculate(xyz)
    #    Ginv = self.GInverse(xyz)
    #    Bmat = self.wilsonB(xyz)
    #    # Internal coordinate gradient
    #    # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx)
    #    #Gq = multi_dot([Ginv, Bmat, gradx])
    #    #return Gq
    #    return block_matrix.dot( Ginv,block_matrix.dot(Bmat,gradx) )

    def makeConstraints(self, xyz, constraints, cvals=None):
        # Add the list of constraints.
        xyz = xyz.flatten()
        if cvals is None and constraints is not None:
            cvals = []
            # If coordinates are provided instead of a constraint value,
            # then calculate the constraint value from the positions.
            # If both are provided, then the coordinates are ignored.
            for c in constraints:
                cvals.append(c.value(xyz))
            if len(constraints) != len(cvals):
                raise RuntimeError("List of constraints should be same length as constraint values")
            for cons, cval in zip(constraints, cvals):
                self.addConstraint(cons, cval, xyz)

    # def __repr__(self):
    #     lines = ["Internal coordinate system (atoms numbered from 1):"]
    #     typedict = OrderedDict()
    #     for Internal in self.Internals:
    #         lines.append(Internal.__repr__())
    #         if str(type(Internal)) not in typedict:
    #             typedict[str(type(Internal))] = 1
    #         else:
    #             typedict[str(type(Internal))] += 1
    #     if len(lines) > 200:
    #         # Print only summary if too many
    #         lines = []
    #     for k, v in list(typedict.items()):
    #         lines.append("%s : %i" % (k, v))
    #     return '\n'.join(lines)

    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                self.logger.log_print("this prim is in p1 but not p2 ", i)
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                self.logger.log_print("this prim is in p2 but not p1", i)
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

    def update(self, other):
        Changed = False
        for i in self.Internals:
            if i not in other.Internals:
                if hasattr(i, 'inactive'):
                    i.inactive += 1
                else:
                    i.inactive = 0
                if i.inactive == 1:
                    # logger.info("Deleting:", i)
                    self.Internals.remove(i)
                    Changed = True
            else:
                i.inactive = 0
        for i in other.Internals:
            if i not in self.Internals:
                # logger.info("Adding:  ", i)
                self.Internals.append(i)
                Changed = True
        return Changed

    def join(self, other, bonds_only=False):
        Changed = False
        for i in other.Internals:
            if i not in self.Internals:
                if bonds_only and i.type_class != slots.CoordinateTypeClasses.Distance:
                    pass
                else:
                    # logger.info("Adding:  ", i)
                    self.logger.log_print(("Adding ", i))
                    self.Internals.append(i)
                    Changed = True
        return Changed

    def repr_diff(self, other):
        alines = ["-- Added: --"]
        for i in other.Internals:
            if i not in self.Internals:
                alines.append(i.__repr__())
        dlines = ["-- Deleted: --"]
        for i in self.Internals:
            if i not in other.Internals:
                dlines.append(i.__repr__())
        output = []
        if len(alines) > 1:
            output += alines
        if len(dlines) > 1:
            output += dlines
        return '\n'.join(output)

    def resetRotations(self, xyz):
        for Internal in self.Internals:
            if isinstance(Internal, slots.LinearAngle):
                Internal.reset(xyz)
        for rot in list(self.Rotators.values()):
            rot.reset(xyz)

    def largeRots(self):
        for Internal in self.Internals:
            if isinstance(Internal, slots.LinearAngle):
                if Internal.stored_dot2 > 0.75:
                    # Linear angle is almost parallel to reference axis
                    return True
            if isinstance(Internal, slots.Rotation):
                if Internal in self.cPrims:
                    continue
                if Internal.Rotator.stored_norm > 0.9*np.pi:
                    # Molecule has rotated by almost pi
                    return True
                if Internal.Rotator.stored_dot2 > 0.9:
                    # Linear molecule is almost parallel to reference axis
                    return True
        return False

    def calculate(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    def calculateDegrees(self, xyz):
        answer = []
        for Internal in self.Internals:
            value = Internal.value(xyz)
            if Internal.isAngular:
                value *= 180/np.pi
            answer.append(value)
        return np.array(answer)

    def getRotatorNorms(self):
        rots = []
        for Internal in self.Internals:
            if isinstance(Internal, slots.RotationA):
                rots.append(Internal.Rotator.stored_norm)
        return rots

    def getRotatorDots(self):
        dots = []
        for Internal in self.Internals:
            if isinstance(Internal, slots.RotationA):
                dots.append(Internal.Rotator.stored_dot2)
        return dots

    def printRotations(self, xyz):
        rotNorms = self.getRotatorNorms()
        if len(rotNorms) > 0:
            self.logger.log_print("Rotator Norms: ", " ".join(["% .4f" % i for i in rotNorms]))
        rotDots = self.getRotatorDots()
        if len(rotDots) > 0 and np.max(rotDots) > 1e-5:
            self.logger.log_print("Rotator Dots : ", " ".join(["% .4f" % i for i in rotDots]))
        linAngs = [ic.value(xyz) for ic in self.Internals if isinstance(ic, slots.LinearAngle)]
        if len(linAngs) > 0:
            self.logger.log_print("Linear Angles: ", " ".join(["% .4f" % i for i in linAngs]))

    def derivatives(self, xyz):
        self.calculate(xyz)
        answer = [p.derivative(xyz) for p in self.Internals]
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)

    def calcDiff(self, xyz1, xyz2):
        """ Calculate difference in internal coordinates (coord1-coord2), accounting for changes in 2*pi of angles. """
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.calcDiff(xyz1, xyz2))
        return np.array(answer)

    @classmethod
    def _addable_dof(cls, dof, internals):
        return (
            isinstance(dof, slots.CartesianPosition)
            or dof not in internals
        )
    @classmethod
    def _dispatch_add(cls, dof, internals, verbose=False):
        if cls._addable_dof(dof, internals):
            if verbose:
                print((" adding ", dof))
            internals.append(dof)
            return True
        else:
            return False

    def add(self, dof, verbose=False):
        if dof.__class__.__name__ in ['CartesianX', 'CartesianY', 'CartesianZ']:
            if verbose:
                self.logger.log_print((" adding ", dof))
            self.Internals.append(dof)
        elif dof not in self.Internals:
            if verbose:
                self.logger.log_print((" adding ", dof))
            self.Internals.append(dof)
            return True
        else:
            return False

    # def dof_index(self, dof):
    #     return self.Internals.index(dof)

    def dof_index(self, indice_tuple, dtype='Distance'):
        if dtype == "Distance":
            i, j = indice_tuple
            prim = slots.Distance(i, j)
        elif dtype == "Angle":
            i, j, k = indice_tuple
            prim = slots.Angle(i, j, k)
        elif dtype == "Dihedral":
            i, j, k, l = indice_tuple
            prim = slots.Dihedral(i, j, k, l)
        elif dtype == "OutOfPlane":
            i, j, k, l = indice_tuple
            prim = slots.OutOfPlane(i, j, k, l)
        else:
            self.logger.log_print(dtype)
            raise NotImplementedError(dtype)
        return self.Internals.index(prim)

    _primitive_ordering = [
        slots.Distance, slots.Angle, slots.LinearAngle,
        slots.OutOfPlane, slots.Dihedral,
        slots.CartesianX, slots.CartesianY, slots.CartesianZ,
        slots.TranslationX, slots.TranslationY, slots.TranslationZ,
        slots.RotationA, slots.RotationB, slots.RotationC
    ]
    def reorderPrimitives(self):
        # Reorder primitives to be in line with cc's code
        newPrims = []
        for cPrim in self.cPrims:
            newPrims.append(cPrim)

        for typ in self._primitive_ordering:
            for p in self.Internals:
                if type(p) is typ and p not in self.cPrims:
                    newPrims.append(p)
        if len(newPrims) != len(self.Internals):
            raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")
        self.Internals = tuple(newPrims)

        if not self.connect:
            self.reorderPrimsByFrag()
        else:
            # all atoms are considered one "fragment"
            self.block_info = [(1, self.natoms, len(newPrims), 'P')]

    @classmethod
    def _find_atom_lines(cls, frag, coords, *, linear_threshold):

        # Find groups of atoms that are in straight lines
        atom_lines = [list(i) for i in frag.edges()]
        for _ in range(len(frag.nodes()) ** 2):  # no reason to have an unconstrained loop here...really, really dumb
            # For a line of two atoms (one bond):
            # AB-AC
            # AX-AY
            # i.e. AB is the first one, AC is the second one
            # AX is the second-to-last one, AY is the last one
            # AB-AC-...-AX-AY
            # AB-(AC, AX)-AY
            line_lens = [len(l) for l in atom_lines]
            for aline in atom_lines:
                # Imagine a line of atoms going like ab-ac-ax-ay.
                # Our job is to extend the line until there are no more
                ab = aline[0]
                ay = aline[-1]
                for aa in frag.neighbors(ab):
                    if aa not in aline:
                        # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                        # Add AA to the front of the list
                        if all([np.abs(np.cos(slots.Angle(aa, ab, ac).value(coords))) > linear_threshold for ac in aline[1:] if
                                ac != ab]):
                            aline.insert(0, aa)
                for az in frag.neighbors(ay):
                    if az not in aline:
                        if all([np.abs(np.cos(slots.Angle(ax, ay, az).value(coords))) > linear_threshold for ax in aline[:-1] if
                                ax != ay]):
                            aline.append(az)
            if len(line_lens) == len(atom_lines) and line_lens == [len(l) for l in atom_lines]:
                break
        else:
            raise ValueError("atom line detection ran out of loop iterations?")
        atom_lines_uniq = []
        mask = set()
        for i in atom_lines:  #
            l = tuple(i)
            if l not in mask:
                mask.add(l)
                atom_lines_uniq.append(l)

        return atom_lines_uniq


    @classmethod
    def newMakePrimitives(cls,
                          atoms,
                          xyz,
                          topology,
                          fragments:'list[EdgeGraph]',
                          *,
                          logger,
                          connect=False,
                          addcart=False,
                          addtr=False,
                          hybrid_idx_start_stop=None,
                          linear_threshold=0.95 # This number works best for the iron complex
                          ):
        Internals = []
        Rotators = OrderedDict()
        block_info = []
        if hybrid_idx_start_stop is None:
            hybrid_idx_start_stop = []
        # coordinates in Angstrom
        coords = xyz.flatten()

        logger.log_print(" Creating block info", log_level=logger.LogLevel.Debug)
        tmp_block_info = []
        # get primitive blocks
        for frag in fragments:
            nodes = frag.L()
            tmp_block_info.append((nodes[0], nodes[-1]+1, frag, 'reg'))
            # TODO can assert blocks are contiguous here
        logger.log_print(" number of primitive blocks is {nfrag}", nfrag=len(fragments), log_level=logger.LogLevel.Debug)

        # get hybrid blocks
        for tup in hybrid_idx_start_stop:
            # Add primitive Cartesians for each atom in hybrid block
            sa = tup[0]
            ea = tup[1]
            leng = ea-sa
            for atom in range(sa, ea+1):
                tmp_block_info.append((atom, atom+1, None, 'hyb'))

        # sort the blocks
        tmp_block_info.sort(key=lambda tup: tup[0])
        # print("block info")
        # print(tmp_block_info)
        logger.log_print([
            " Done creating block info",
            "Now Making Primitives by block"
        ], log_level=logger.LogLevel.Debug)

        natoms = len(atoms)
        tmp_internals = []
        sp = 0
        for info in tmp_block_info:
            nprims = 0
            # This corresponds to the primitive coordinate region
            if info[-1] == 'reg':
                frag = info[2]
                noncov = []
                if connect:
                    raise NotImplementedError("disabled code path")
                    # Connect all non-bonded fragments together
                    # Make a distance matrix mapping atom pairs to interatomic distances
                    AtomIterator, dxij = Topology.distance_matrix(xyz, pbc=False)
                    D = {}
                    for i, j in zip(AtomIterator, dxij[0]):
                        assert i[0] < i[1]
                        D[tuple(i)] = j
                    dgraph = nx.Graph()
                    for i in range(natoms):
                        dgraph.add_node(i)
                    for k, v in list(D.items()):
                        dgraph.add_edge(k[0], k[1], weight=v)
                    mst = sorted(list(nx.minimum_spanning_edges(dgraph, data=False)))
                    for edge in mst:
                        if edge not in list(topology.edges()):
                            self.logger.log_print("Adding %s from minimum spanning tree" % str(edge))
                            topology.add_edge(edge[0], edge[1])
                            noncov.append(edge)

                else:  # Add Cart or TR
                    if addcart:
                        for i in range(info[0], info[1]):
                            cls._dispatch_add(slots.CartesianX(i, w=1.0), tmp_internals)
                            cls._dispatch_add(slots.CartesianY(i, w=1.0), tmp_internals)
                            cls._dispatch_add(slots.CartesianZ(i, w=1.0), tmp_internals)
                            nprims += 3
                    elif addtr:
                        nodes = frag.nodes()
                        # print(" Nodes")
                        # print(nodes)
                        if len(nodes) >= 2:
                            cls._dispatch_add(slots.TranslationX(nodes, w=np.ones(len(nodes))/len(nodes)), tmp_internals)
                            cls._dispatch_add(slots.TranslationY(nodes, w=np.ones(len(nodes))/len(nodes)), tmp_internals)
                            cls._dispatch_add(slots.TranslationZ(nodes, w=np.ones(len(nodes))/len(nodes)), tmp_internals)
                            sel = xyz.reshape(-1, 3)[nodes, :]
                            sel -= np.mean(sel, axis=0)
                            rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                            cls._dispatch_add(slots.RotationA(nodes, coords, Rotators, w=rg), tmp_internals)
                            cls._dispatch_add(slots.RotationB(nodes, coords, Rotators, w=rg), tmp_internals)
                            cls._dispatch_add(slots.RotationC(nodes, coords, Rotators, w=rg), tmp_internals)
                            nprims += 6
                        else:
                            for j in nodes:
                                cls._dispatch_add(slots.CartesianX(j, w=1.0), tmp_internals)
                                cls._dispatch_add(slots.CartesianY(j, w=1.0), tmp_internals)
                                cls._dispatch_add(slots.CartesianZ(j, w=1.0), tmp_internals)
                                nprims += 3

                # # Build a list of noncovalent distances
                # Add an internal coordinate for all interatomic distances
                for (a, b) in frag.edges():
                    # if a in list(range(info[0],info[1])):
                    if cls._dispatch_add(slots.Distance(a, b), tmp_internals):
                        nprims += 1

                # Add an internal coordinate for all angles
                AngDict = defaultdict(list)
                for b in frag.nodes():
                    for a in frag.neighbors(b):
                        for c in frag.neighbors(b):
                            if a < c:
                                # if (a, c) in self.topology.edges() or (c, a) in self.topology.edges(): continue
                                Ang = slots.Angle(a, b, c)
                                nnc = (min(a, b), max(a, b)) in noncov
                                nnc += (min(b, c), max(b, c)) in noncov
                                # if nnc >= 2: continue
                                # logger.info("LPW: cosine of angle", a, b, c, "is", np.abs(np.cos(Ang.value(coords))))
                                if np.abs(np.cos(Ang.value(coords))) < linear_threshold:
                                    if cls._dispatch_add(slots.Angle(a, b, c), tmp_internals):
                                        nprims += 1
                                    AngDict[b].append(Ang)
                                elif connect or not addcart:
                                    # logger.info("Adding linear angle")
                                    # Add linear angle IC's
                                    # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in selfs (e.g. HCCCN)
                                    # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                                    # Bringing back old code to use "translations" for the latter case, but should be investigated
                                    # more deeply in the future.
                                    if nnc == 0:
                                        if cls._dispatch_add(slots.LinearAngle(a, b, c, 0), tmp_internals):
                                            nprims += 1
                                        if cls._dispatch_add(slots.LinearAngle(a, b, c, 1), tmp_internals):
                                            nprims += 1
                                    else:
                                        # Unit vector connecting atoms a and c
                                        nac = xyz[c] - xyz[a]
                                        nac /= np.linalg.norm(nac)
                                        # Dot products of this vector with the Cartesian axes
                                        dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                        # Functions for adding Cartesian coordinate
                                        # carts = [CartesianX, CartesianY, CartesianZ]
                                        # print("warning, adding translation, did you mean this?")
                                        trans = [slots.TranslationX, slots.TranslationY, slots.TranslationZ]
                                        w = np.array([-1.0, 2.0, -1.0])
                                        # Add two of the most perpendicular Cartesian coordinates
                                        for i in np.argsort(dots)[:2]:
                                            if cls._dispatch_add(trans[i]([a, b, c], w=w), tmp_internals):
                                                nprims += 1

                # Make Dihedrals
                for b in frag.nodes():
                    for a in frag.neighbors(b):
                        for c in frag.neighbors(b):
                            for d in frag.neighbors(b):
                                if a < c < d:
                                    nnc = (min(a, b), max(a, b)) in noncov
                                    nnc += (min(b, c), max(b, c)) in noncov
                                    nnc += (min(b, d), max(b, d)) in noncov
                                    # if nnc >= 1: continue
                                    for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):
                                        Ang1 = slots.Angle(b, i, j)
                                        Ang2 = slots.Angle(i, j, k)
                                        if np.abs(np.cos(Ang1.value(coords))) > linear_threshold:
                                            continue
                                        if np.abs(np.cos(Ang2.value(coords))) > linear_threshold:
                                            continue
                                        if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > linear_threshold:
                                            if cls._dispatch_add(slots.Angle(i, b, j), tmp_internals):
                                                nprims -= 1
                                            if cls._dispatch_add(slots.OutOfPlane(b, i, j, k), tmp_internals):
                                                nprims += 1
                                            break

                atom_lines_uniq = cls._find_atom_lines(frag, coords, linear_threshold=linear_threshold)
                # lthree = [l for l in atom_lines_uniq if len(l) > 2]
                # TODO: Perhaps should reduce the times this is printed out in reaction paths
                # if len(lthree) > 0:
                #     self.logger.log_print "Lines of three or more atoms:", ', '.join(['-'.join(["%i" % (i+1) for i in l]) for l in lthree])

                # Normal dihedral code
                for aline in atom_lines_uniq:
                    # Go over ALL pairs of atoms in a line
                    for (b, c) in itertools.combinations(aline, 2):
                        if b > c:
                            (b, c) = (c, b)
                        # Go over all neighbors of b
                        for a in frag.neighbors(b):
                            # Go over all neighbors of c
                            for d in frag.neighbors(c):
                                # Make sure the end-atoms are not in the line and not the same as each other
                                if a not in aline and d not in aline and a != d:
                                    nnc = (min(a, b), max(a, b)) in noncov
                                    nnc += (min(b, c), max(b, c)) in noncov
                                    nnc += (min(c, d), max(c, d)) in noncov
                                    # print aline, a, b, c, d
                                    Ang1 = slots.Angle(a, b, c)
                                    Ang2 = slots.Angle(b, c, d)
                                    # Eliminate dihedrals containing angles that are almost linear
                                    # (should be eliminated already)
                                    if np.abs(np.cos(Ang1.value(coords))) > linear_threshold:
                                        continue
                                    if np.abs(np.cos(Ang2.value(coords))) > linear_threshold:
                                        continue
                                    if cls._dispatch_add(slots.Dihedral(a, b, c, d), tmp_internals):
                                        nprims += 1

            else:   # THIS ELSE CORRESPONS TO FRAGMENTS BUILT WITH THE HYBRID REGION (below)
                cls._dispatch_add(slots.CartesianX(info[0], w=1.0), tmp_internals)
                cls._dispatch_add(slots.CartesianY(info[0], w=1.0), tmp_internals)
                cls._dispatch_add(slots.CartesianZ(info[0], w=1.0), tmp_internals)
                nprims = 3

            # Add all elements in tmp_Internals to Internals and then clear list
            Internals += tmp_internals
            tmp_internals = []

            ep = sp+nprims
            block_info.append((info[0], info[1], sp, ep))
            sp = ep

        # print(self.Internals)
        prim_only_block_info = []
        for info1, info2 in zip(tmp_block_info, block_info):
            if info1[-1] == 'hyb':
                pass
            else:
                prim_only_block_info.append(info2)

        logger.log_print([
            " Done making primitives",
            " Made a total of {nprim} primitives",
            " num blocks {nblock}",
            " num prim blocks {nprim_block}"
            ],
            nprim=len(Internals),
            nblock=len(block_info),
            nprim_block=len(prim_only_block_info),
            log_level=logger.LogLevel.Debug
        )
        # print(self.prim_only_block_info)
        # if len(newPrims) != len(self.Internals):
        #    #print(np.setdiff1d(self.Internals,newPrims))
        #    raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")

        return Internals, block_info

    def insert_block_primitives(self, prims, reform_topology):
        '''
        The SE-GSM needs to add primitives, we have to do this carefully because of the blocks
        '''
        raise NotImplementedError("this wasn't implemented")

        return

    def reorderPrimsByFrag(self):
        '''
        Warning this assumes that the fragments aren't intermixed. you shouldn't do that!!!!
        '''

        # these are the subgraphs
        # frags = [m for m in self.fragments]
        newPrims = []

        # Orders the primitives by fragment, also takes into accoutn hybrid fragments (those that don't contain primitives)
        # if it's 'P' then its primitive and the BMatrix uses the derivative
        # if it's 'H' then its hybrid and the BMatrix uses the diagonal
        # TODO rename variables to reflect current understanding
        # TODO The 'P' and 'H' nomenclature is probably not necessary since the regions are
        # distinguishable by the number of primitives they contain,
        # gt 0 in the former and eq 0 in the latter
        # print(" Getting the block information")

        tmp_block_info = []

        self.logger.log_print(" Creating block info")
        # get primitive blocks
        for frag in self.fragments:
            nodes = frag.L()
            tmp_block_info.append((nodes[0], nodes[-1]+1, frag, 'reg'))
            # TODO can assert blocks are contiguous here

        # get hybrid blocks
        for tup in self.hybrid_idx_start_stop:
            # Add primitive Cartesians for each atom in hybrid block
            sa = tup[0]
            ea = tup[1]
            for atom in range(sa, ea+1):
                tmp_block_info.append((atom, atom+1, None, 'hyb'))

        # sort the blocks
        tmp_block_info.sort(key=lambda tup: tup[0])

        self.logger.log_print(" Done creating block info,\n Now Ordering Primitives by block")

        # Order primitives by block
        # probably faster to just reform the primitives!!!!

        self.block_info = []
        sp = 0

        for info in tmp_block_info:
            nprims = 0
            if info[-1] == 'reg':
                # TODO OLD
                for p in self.Internals:
                    atoms = p.atoms
                    if all([atom in range(info[0], info[1]) for atom in atoms]):
                        newPrims.append(p)
                        nprims += 1
            else:
                newPrims.append(slots.CartesianX(info[0], w=1.0))
                newPrims.append(slots.CartesianY(info[0], w=1.0))
                newPrims.append(slots.CartesianZ(info[0], w=1.0))
                nprims = 3

            ep = sp+nprims
            self.block_info.append((info[0], info[1], sp, ep))
            sp = ep

        # print(" block info")
        # print(self.block_info)
        # print(" Done Ordering prims by block")
        # print("num blocks ",len(self.block_info))
        # if len(newPrims) != len(self.Internals):
        #    #print(np.setdiff1d(self.Internals,newPrims))
        #    raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")
        self.Internals = tuple(newPrims)

        self.logger.log_print(self.Internals)
        self.clearCache()
        return

    def guess_hessian(self, coords, bonds=None):
        """
        Build a guess Hessian that roughly follows Schlegel's guidelines.
        """
        xyzs = coords.reshape(-1, 3)

        def covalent(a, b):
            if bonds is None:
                r = np.linalg.norm(xyzs[a]-xyzs[b])
                rcov = self.atoms[a].covalent_radius + self.atoms[b].covalent_radius
                return r/rcov < 1.2
            else:
                return b in bonds.neighbors(a)

        Hdiag = []
        for ic in self.Internals:
            if ic.type_class == slots.CoordinateTypeClasses.Distance:
                # r = np.linalg.norm(xyzs[ic.a]-xyzs[ic.b])
                # elem1 = min(self.atoms[ic.a].atomic_num, self.atoms[ic.b].atomic_num)
                # elem2 = max(self.atoms[ic.a].atomic_num, self.atoms[ic.b].atomic_num)
                # A = 1.734
                # if elem1 < 3:
                #    if elem2 < 3:
                #        B = -0.244
                #    elif elem2 < 11:
                #        B = 0.352
                #    else:
                #        B = 0.660
                # elif elem1 < 11:
                #    if elem2 < 11:
                #        B = 1.085
                #    else:
                #        B = 1.522
                # else:
                #    B = 2.068
                if covalent(ic.a, ic.b):
                    Hdiag.append(0.35)
                    # Hdiag.append(A/(r-B)**3)
                else:
                    Hdiag.append(0.1)
            elif ic.type_class == slots.CoordinateTypeClasses.Angle:
                a = ic.a
                c = ic.c
                if min(self.atoms[a].atomic_num,
                        self.atoms[ic.b].atomic_num,
                        self.atoms[c].atomic_num) < 3:
                    A = 0.160
                else:
                    A = 0.250
                if covalent(a, ic.b) and covalent(ic.b, c):
                    Hdiag.append(A)
                else:
                    Hdiag.append(0.1)
            elif ic.type_class == slots.CoordinateTypeClasses.Dihedral:
                # r = np.linalg.norm(xyzs[ic.b]-xyzs[ic.c])
                # rcov = self.atoms[ic.b].covalent_radius + self.atoms[ic.c].covalent_radius
                # Hdiag.append(0.1)
                Hdiag.append(0.023)
            elif ic.type_class == slots.CoordinateTypeClasses.OutOfPlane:
                # r1 = xyzs[ic.b]-xyzs[ic.a]
                # r2 = xyzs[ic.c]-xyzs[ic.a]
                # r3 = xyzs[ic.d]-xyzs[ic.a]
                # d = 1 - np.abs(np.dot(r1, np.cross(r2, r3))/np.linalg.norm(r1)/np.linalg.norm(r2)/np.linalg.norm(r3))
                # Hdiag.append(0.1)
                if covalent(ic.a, ic.b) and covalent(ic.a, ic.c) and covalent(ic.a, ic.d):
                    Hdiag.append(0.045)
                else:
                    Hdiag.append(0.023)
            elif ic.type_class in {
                slots.CoordinateTypeClasses.Cartesian,
                slots.CoordinateTypeClasses.Translation,
                slots.CoordinateTypeClasses.Rotation,
            }:
                Hdiag.append(0.05)
            else:
                raise RuntimeError('Failed to build guess Hessian matrix. Make sure all IC types are supported')
        return np.diag(Hdiag)

    def second_derivatives_nb(self, xyz):
        self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.second_derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        return np.array(answer)

    def second_derivatives(self, xyz):
        self.calculate(xyz)
        c_list = []
        for info in self.block_info:
            sa = info[0]
            ea = info[1]
            sp = info[2]
            ep = info[3]
            na = ea - sa
            SDer = np.array(
                [np.reshape(p.second_derivative(xyz[sa:ea, :], start_idx=sa), (3*na, 3*na)) for p in self.Internals[sp:ep]]
            )

            c_list.append(SDer)

        answer = block_tensor(c_list)
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) 3*Number of atoms
        # 4) 3*Number of atoms
        return answer

    def calcCg(self,xyz,gradx):
        '''
        Calculates the tensor product Cxg where C is the tensor of second derivatives and g
        is the internal coordinate gradient
        '''
        self.calculate(xyz)
        Gq = self.calcGrad(xyz, gradx).flatten()
        natoms = len(xyz)

        result_list = []
        for info in self.block_info:
            sa = int(info[0])
            ea = int(info[1])
            sp = int(info[2])
            ep = int(info[3])
            na = ea - sa
            mini_result = np.zeros( (3*na,3*na), dtype=float )
            for p,prim in enumerate(self.Internals[sp:ep]):
                sder = np.reshape(prim.second_derivative(xyz[sa:ea,:],start_idx=sa), (3*na,3*na))
                mini_result += sder*Gq[p+sp]
            result_list.append(mini_result)

        result = block_diag(*result_list)

        return result

    def get_hybrid_indices(self,xyz):
        '''
        Get the hybrid indices if they exist
        '''

        natoms = len(xyz)

        # print("fragments")

        # need the primitive start and stop indices
        prim_idx_start_stop = []
        new = True
        for frag in self.fragments:
            nodes = frag.L()
            # print(nodes)
            prim_idx_start_stop.append((nodes[0], nodes[-1]))
        # print("prim start stop")
        # print(prim_idx_start_stop)

        prim_idx = []
        for info in prim_idx_start_stop:
            prim_idx += list(range(int(info[0]), int(info[1]+1)))
        # print('prim indices')
        # print(prim_idx)

        # print(natoms)
        new_hybrid_indices = list(range(int(natoms)))
        # print(new_hybrid_indices)
        # for count,i in enumerate(prim_idx):
        #    self.logger.log_print(i,end=' ')
        #    if (count+1) %20==0:
        #        self.logger.log_print('')
        # print()
        # print(type(new_hybrid_indices[0]))
        for elem in prim_idx:
            try:
                new_hybrid_indices.remove(elem)
            except:
                self.logger.log_print(elem)
                self.logger.log_print(type(elem))
                raise RuntimeError
        # print('hybrid indices')
        # print(new_hybrid_indices)

        # get the hybrid start and stop indices
        new = True
        for i in range(natoms+1):
            if i in new_hybrid_indices:
                if new:
                    start = i
                    new = False
            else:
                if new is False:
                    end = i-1
                    new = True
                    self.hybrid_idx_start_stop.append((start, end))
        # print(" hybrid start stop")
        # print(self.hybrid_idx_start_stop)

    def append_prim_to_block(self, prim, count=None):
        total_blocks = len(self.block_info)

        if count is None:
            count = 0
            for info in self.block_info:
                if info[3]-info[2] != 3:  # this is a hybrid block skipping
                    if all([atom in range(info[0], info[1]) for atom in prim.atoms]):
                        break
                count += 1
            # print(" the prim lives in block {}".format(count))

        # the start and end of the primitives is stored in block info
        # the third element is the end index for that blocks prims
        elem = self.block_info[count][3]

        self.Internals.insert(elem, prim)
        # print(" prims after inserting at elem {}".format(elem))
        # print(self.Internals)

        new_block_info = []
        for i, info in enumerate(self.block_info):
            if i < count:
                # sa,ea,sp,ep --> therefore all sps before count are unaffected
                new_block_info.append((info[0], info[1], info[2], info[3]))
            elif i == count:
                new_block_info.append((info[0], info[1], info[2], info[3]+1))
            else:
                new_block_info.append((info[0], info[1], info[2]+1, info[3]+1))
        # print(new_block_info)
        self.block_info = new_block_info

        return

    def add_union_primitives(self, other):

        # Can make this faster if only check primitive indices
        # Need the primitive internal coordinates -- not the Cartesian internal coordinates
        self.logger.log_print(" Number of primitives before {}".format(len(self.Internals)))
        # print(' block info before')
        # print(self.block_info)

        # prim_idx1 =[]
        # for count,prim in enumerate(self.Internals):
        #    if type(prim) not in [CartesianX,CartesianY,CartesianZ]:
        #        prim_idx1.append(count)

        # prim_idx2 =[]
        # for count,prim in enumerate(other.Internals):
        #    if type(prim) not in [CartesianX,CartesianY,CartesianZ]:
        #        prim_idx2.append(count)

        # tmp_internals1 = [self.Internals[i] for i in prim_idx1]
        # tmp_internals2 = [other.Internals[i] for i in prim_idx2]

        # #for i in other.Internals:
        # #    if i not in self.Internals:
        # for i in tmp_internals2:
        #    if i not in tmp_internals1:
        #        #print("this prim is in p2 but not p1",i)
        #        print("Adding prim {} that is in Other to Internals".format(i))
        #        self.append_prim_to_block(i)

        # NEW
        # will be changing block info and self.Internals therefore
        # need to create temporary Internals list to check other against
        block_info = copy(self.block_info)
        count = 0
        for info1, info2 in zip(block_info, other.block_info):
            sa1, ea1, sp1, ep1 = info1
            sa2, ea2, sp2, ep2 = info2
            for i in other.Internals[sp2:ep2]:
                # Dont check Cartesians
                if i.type_class != slots.CoordinateTypeClasses.Cartesian:
                    if i not in self.Internals[sp1:ep1]:
                        self.logger.log_print("Adding prim {} that is in Other to Internals".format(i))
                        self.append_prim_to_block(i, count)
            count += 1

        # print(self.Internals)
        # print(len(self.Internals))
        self.logger.log_print(" Number of primitives after {}".format(len(self.Internals)))
        # print(' block info after')
        # print(self.block_info)

    def add_driving_coord_prim(self, driving_coordinates):
        driving_coord_prims = []
        for dc in driving_coordinates:
            prim = get_driving_coord_prim(dc)
            if prim is not None:
                driving_coord_prims.append(prim)
        for dc in driving_coord_prims:
            if dc.type_class != slots.CoordinateTypeClasses.Distance:  # Already handled in topology
                if dc not in self.Internals:
                    self.logger.log_print("Adding driving coord prim {} to Internals".format(dc))
                    self.append_prim_to_block(dc)

    @classmethod
    def get_driving_coord_prim(cls, dc):
        prim = None
        if "ADD" in dc or "BREAK" in dc:
            if dc[1] < dc[2]:
                prim = slots.Distance(dc[1]-1, dc[2]-1)
            else:
                prim = slots.Distance(dc[2]-1, dc[1]-1)
        elif "ANGLE" in dc:
            if dc[1] < dc[3]:
                prim = slots.Angle(dc[1]-1, dc[2]-1, dc[3]-1)
            else:
                prim = slots.Angle(dc[3]-1, dc[2]-1, dc[1]-1)
        elif "TORSION" in dc:
            if dc[1] < dc[4]:
                prim = slots.Dihedral(dc[1]-1, dc[2]-1, dc[3]-1, dc[4]-1)
            else:
                prim = slots.Dihedral(dc[4]-1, dc[3]-1, dc[2]-1, dc[1]-1)
        elif "OOP" in dc:
            # if dc[1]<dc[4]:
            prim = slots.OutOfPlane(dc[1]-1, dc[2]-1, dc[3]-1, dc[4]-1)
            # else:
            #    prim = OutOfPlane(dc[4]-1,dc[3]-1,dc[2]-1,dc[1]-1)
        return prim