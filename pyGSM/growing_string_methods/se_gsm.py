from __future__ import print_function

from ..coordinate_systems import (
    Distance, Angle, Dihedral, OutOfPlane,
    PrimitiveInternalCoordinates
)
from ..coordinate_systems import slots
from .gsm import TSOptimizationStrategy
from .main_gsm import MainGSM
from ..molecule import Molecule
# standard library imports

# third party
from collections import Counter
import numpy as np

__all__ = [
    "SE_GSM"
]

class SE_GSM(MainGSM):
    default_rtype = TSOptimizationStrategy(2)
    default_max_opt_steps = 3

    @classmethod
    def preadjust_nodes(cls, nodes, evaluator, driving_coords):
        if isinstance(driving_coords, str) or hasattr(driving_coords, 'read'):
            driving_coords = cls.read_isomers_file(driving_coords)

        driving_coord_prims = []
        for dc in driving_coords:
            prim = PrimitiveInternalCoordinates.get_driving_coord_prim(dc)
            if prim is not None:
                if prim.type_class != slots.CoordinateTypeClasses.Distance:
                    raise NotImplementedError(f"driving a reaction by anything other than a distance not supported, (got {prim.type_class})")
                driving_coord_prims.append(prim)

        reactant:Molecule = nodes[0]
        base_edges = list(reactant.bond_graph.edges())
        target_edges = [
            (prim.atoms[0], prim.atoms[1])
            for prim in driving_coord_prims
        ]
        nodes = cls.add_bonds_to_nodes(
            nodes,
            base_edges,
            target_edges
        )

        nodes = cls.add_evaluator_to_nodes(nodes, evaluator)

        return nodes

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.current_nnodes = 1

        self.logger.log_print(" Assuming the isomers are initialized!")
        self.logger.log_print(" Primitive Internal Coordinates")
        self.logger.log_print(self.nodes[0].primitive_internal_coordinates[0:50])
        self.logger.log_print(" number of primitives is", self.nodes[0].num_primitives)

        self.logger.log_print('Driving Coordinates: ')
        self.logger.log_print(self.driving_coords)
        sys.stdout.flush()

        # stash bdist for node 0
        ictan, self.nodes[0].bdist = self.get_tangent(
            self.nodes[0],
            None,
            driving_coords=self.driving_coords,
        )
        self.nodes[0].update_coordinate_basis(constraints=ictan)

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy
        #TODO should be actual gradient
        self.nodes[0].gradrms = 0.

    def isomer_init(self):
        '''
        The purpose of this function is to add to the primitives the driving coordinate prims if
        they dont exist.
        This is depracated because it's better to build the topology properly before initializing
        GSM. See main.py
        '''

        # TODO ANGLE, TORSION or OOP between fragments will not work if using TRIC with BLOCK LA
        changed_top = False

        # TODO first check if there is any add/break then rebuild topology and makePrimitives

        for i in self.driving_coords:
            if "ADD" in i or "BREAK" in i:
                # order
                if i[1] < i[2]:
                    bond = Distance(i[1]-1, i[2]-1)
                else:
                    bond = Distance(i[2]-1, i[1]-1)
                self.nodes[0].coord_obj.Prims.add(bond, verbose=True)
                changed_top = True
            if "ANGLE" in i:
                if i[1] < i[3]:
                    angle = Angle(i[1]-1, i[2]-1, i[3]-1)
                else:
                    angle = Angle(i[3]-1, i[2]-1, i[1]-1)
                self.nodes[0].coord_obj.Prims.add(angle, verbose=True)
            if "TORSION" in i:
                if i[1] < i[4]:
                    torsion = Dihedral(i[1]-1, i[2]-1, i[3]-1, i[4]-1)
                else:
                    torsion = Dihedral(i[4]-1, i[3]-1, i[2]-1, i[1]-1)
                self.nodes[0].coord_obj.Prims.add(torsion, verbose=True)
            if "OOP" in i:
                if i[1] < i[4]:
                    oop = OutOfPlane(i[1]-1, i[2]-1, i[3]-1, i[4]-1)
                else:
                    oop = OutOfPlane(i[4]-1, i[3]-1, i[2]-1, i[1]-1)
                self.nodes[0].coord_obj.Prims.add(oop, verbose=True)

        self.nodes[0].coord_obj.Prims.clearCache()
        if changed_top:
            self.nodes[0].coord_obj.Prims.rebuild_topology_from_prim_bonds(self.nodes[0].xyz)
        self.nodes[0].coord_obj.Prims.reorderPrimitives()
        self.nodes[0].update_coordinate_basis()

    def go_gsm(self, max_iters=50, opt_steps=10, *, rtype=None):
        """
        rtype=2 Find and Climb TS,
        1 Climb with no exact find,
        0 turning of climbing image and TS search
        """
        self.set_V0()

        if rtype is None:
            rtype = self.rtype

        if not self.isRestarted is False:
            self.nodes[0].gradrms = 0.
            self.nodes[0].V0 = self.nodes[0].energy
            self.logger.log_print(" Initial energy is %1.4f" % self.nodes[0].energy)
            self.add_GSM_nodeR()
            self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
            if self.tscontinue:
                self.pastts = self.past_ts()
                self.logger.log_print("pastts {}".format(self.pastts))
                try:
                    if self.pastts == 1: #normal over the hill
                        self.add_GSM_nodeR(1)
                        self.add_last_node(2)
                    elif self.pastts == 2 or self.pastts==3: #when cgrad is positive
                        self.add_last_node(1)
                        if self.nodes[self.nR-1].gradrms > 5.*self.options['CONV_TOL']:
                            self.add_last_node(1)
                    elif self.pastts == 3: #product detected by bonding
                        self.add_last_node(1)
                except:
                    self.logger.log_print("Failed to add last node, continuing.")
                    # probably need to make sure last node is optimized

            # self.num_nodes = self.nR
            self.nodes = self.nodes[:self.nR]
            energies = self.energies

            if self.TSnode == self.nR-1:
                self.logger.log_print(" The highest energy node is the last")
                self.logger.log_print(" not continuing with TS optimization.")
                self.tscontinue = False

            self.logger.log_print(" Number of nodes is ", self.num_nodes)
            self.logger.log_print(" Warning last node still not optimized fully")
            self.xyz_writer('grown_string_{:03}.xyz'.format(self.ID), self.geometries, self.energies, self.gradrmss, self.dEs)
            self.logger.log_print(" SSM growth phase over")
            self.done_growing = True

            self.logger.log_print(" beginning opt phase")
            self.logger.log_print("Setting all interior nodes to active")
            for n in range(1, self.num_nodes-1):
                self.active[n] = True
            self.active[self.num_nodes-1] = False
            self.active[0] = False

        if not self.isRestarted:
            self.logger.log_print(" initial ic_reparam")
            self.reparameterize(ic_reparam_steps=25)
            self.logger.log_print(" V_profile (after reparam): ", end=' ')
            energies = self.energies
            for n in range(self.num_nodes):
                self.logger.log_print(" {:7.3f}".format(float(energies[n])), end=' ')
            self.logger.log_print()
            self.xyz_writer('grown_string1_{:03}.xyz'.format(self.ID), self.geometries, self.energies, self.gradrmss, self.dEs)

        if self.tscontinue:
            self.optimize_string(max_iter=max_iters, opt_steps=3, rtype=rtype)  # opt steps fixed at 3 for rtype=1 and 2, else set it to be the large number :) muah hahaahah
        else:
            self.logger.log_print("Exiting early")
            self.end_early = True

        filename = "opt_converged_{:03d}.xyz".format(self.ID)
        self.logger.log_print(" Printing string to " + filename)
        self.xyz_writer(filename, self.geometries, self.energies, self.gradrmss, self.dEs)
        self.logger.log_print("Finished GSM!")

    def set_new_node_tolerances(self, index):
        super().set_new_node_tolerances(index)

        ictan, bdist = self.get_tangent(
            self.nodes[index],
            None,
            driving_coords=self.driving_coords,
        )
        self.nodes[self.nR].bdist = bdist
    def add_last_node(self, rtype):
        assert rtype == 1 or rtype == 2, "rtype must be 1 or 2"
        noptsteps = 100
        if self.nodes[self.nR-1].PES.lot.do_coupling:
            opt_type = 'MECI'
        else:
            opt_type = 'UNCONSTRAINED'

        if rtype == 1:
            self.logger.log_print(" copying last node, opting")
            self.nodes[self.nR] = Molecule.copy_from_options(self.nodes[self.nR-1], new_node_id=self.nR)
            self.logger.log_print(" Optimizing node %i" % self.nR)
            self.optimizer[self.nR].conv_grms = self.options['CONV_TOL']
            self.optimizer[self.nR].conv_gmax = self.options['CONV_gmax']
            self.optimizer[self.nR].conv_Ediff = self.options['CONV_Ediff']
            self.optimizer[self.nR].conv_dE = self.options['CONV_dE']
            path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, self.nR))
            self.optimizer[self.nR].optimize(
                molecule=self.nodes[self.nR],
                refE=self.nodes[0].V0,
                opt_steps=noptsteps,
                opt_type=opt_type,
                path=path,
            )
            self.active[self.nR] = True
            if (self.nodes[self.nR].xyz == self.nodes[self.nR-1].xyz).all():
                self.logger.log_print(" Opt did not produce new geometry")
            else:
                self.nR += 1
        elif rtype == 2:
            self.logger.log_print(" already created node, opting")
            self.optimizer[self.nR-1].conv_grms = self.options['CONV_TOL']
            self.optimizer[self.nR-1].conv_gmax = self.options['CONV_gmax']
            self.optimizer[self.nR-1].conv_Ediff = self.options['CONV_Ediff']
            self.optimizer[self.nR-1].conv_dE = self.options['CONV_dE']
            path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, self.nR-1))
            self.optimizer[self.nR-1].optimize(
                molecule=self.nodes[self.nR-1],
                refE=self.nodes[0].V0,
                opt_steps=noptsteps,
                opt_type=opt_type,
                path=path,
            )
        # print(" Aligning")
        # self.nodes[self.nR-1].xyz = self.com_rotate_move(self.nR-2,self.nR,self.nR-1)
        return

    def grow_nodes(self):
        if self.nodes[self.nR-1].gradrms < self.options['ADD_NODE_TOL']:
            if self.nR == self.num_nodes:
                self.logger.log_print(" Ran out of nodes, exiting GSM")
                raise ValueError
            if self.nodes[self.nR] is None:
                self.add_GSM_nodeR()
                self.logger.log_print(" getting energy for node %d: %5.4f" % (self.nR-1, self.nodes[self.nR-1].energy - self.nodes[0].V0))
        return

    def grow_and_handle_termination(self):
        try:
            base = super().grow_and_handle_termination()
        except ValueError:
            self.logger.log_print([
                "can't add anymore nodes, bdist too small",
            ])

            if self.nodes[self.nR - 1].PES.lot.do_coupling:
                opt_type = 'MECI'
            else:
                opt_type = 'UNCONSTRAINED'
            self.logger.log_print(" optimizing last node")
            self.optimizer[self.nR - 1].conv_grms = self.CONV_TOL
            self.logger.log_print(self.optimizer[self.nR - 1].conv_grms)
            path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, self.nR - 1))
            self.optimizer[self.nR - 1].optimize(
                molecule=self.nodes[self.nR - 1],
                refE=self.nodes[0].V0,
                opt_steps=50,
                opt_type=opt_type,
                path=path,
            )
            return True
        return base

    def add_GSM_nodes(self, newnodes=1):
        if self.nn+newnodes > self.num_nodes:
            self.logger.log_print("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.add_GSM_nodeR()

    def ic_reparam_g(self, ic_reparam_steps=4, n0=0, nconstraints=1):  # see line 3863 of gstring.cpp
        '''
        Dont do ic_reparam_g for SE-GSM
        '''
        return

    def set_frontier_convergence(self, nR):
        # set
        self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        self.optimizer[nR].conv_gmax = 100.  # self.options['ADD_NODE_TOL'] # could use some multiplier times CONV_GMAX...
        self.optimizer[nR].conv_Ediff = 1000.  # 2.5
        self.logger.log_print(" conv_tol of node %d is %.4f" % (nR, self.optimizer[nR].conv_grms))

    def set_active(self, nR, nP=None):
        # print(" Here is active:",self.active)
        self.logger.log_print((" setting active node to %i " % nR))

        for i in range(self.num_nodes):
            if self.nodes[i] is not None:
                self.active[i] = False

        self.set_frontier_convergence(nR)
        self.active[nR] = True
        # print(" Here is new active:",self.active)

    def make_tan_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent-1):
            if nlist[2*n] not in param_list:
                param_list.append(nlist[2*n])
        return param_list

    def make_move_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent):
            if nlist[2*n+1] not in param_list:
                param_list.append(nlist[2*n+1])
        return param_list

    def make_difference_node_list(self):
        ncurrent = 0
        nlist = [0]*(2*self.num_nodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1
        nlist[2*ncurrent+1] = self.nR - 1
        nlist[2*ncurrent] = self.nR - 1
        ncurrent += 1
        return ncurrent, nlist

    def past_ts(self):
        '''
        '''
        ispast = ispast1 = ispast2 = ispast3 = 0
        THRESH1 = 5.
        THRESH2 = 3.
        THRESH3 = -1.
        # THRESHB = 0.05
        CTHRESH = 0.005
        OTHRESH = -0.015
        emax = -100.
        nodemax = 1
        # n0 is zero until after finished growing
        ns = self.n0-1
        if ns < nodemax:
            ns = nodemax

        self.logger.log_print(" Energies", end=' ')
        energies = self.energies
        for n in range(ns, self.nR):
            self.logger.log_print(" {:4.3f}".format(energies[n]), end=' ')
            if energies[n] > emax:
                nodemax = n
                emax = energies[n]
        self.logger.log_print("\n nodemax ", nodemax)

        for n in range(nodemax, self.nR):
            if energies[n] < emax-THRESH1:
                ispast1 += 1
            if energies[n] < emax-THRESH2:
                ispast2 += 1
            if energies[n] < emax-THRESH3:
                ispast3 += 1
            if ispast1 > 1:
                break
        self.logger.log_print(" ispast1", ispast1)
        self.logger.log_print(" ispast2", ispast2)
        self.logger.log_print(" ispast3", ispast3)

        # TODO 5/9/2019 what about multiple constraints
        # Done 6/23/2019
        constraints = self.nodes[self.nR-1].constraints[:, 0]
        gradient = self.nodes[self.nR-1].gradient

        overlap = np.dot(gradient.T, constraints)
        cgrad = overlap*constraints

        cgrad = np.linalg.norm(cgrad)*np.sign(overlap)

        self.logger.log_print((" cgrad: %4.3f nodemax: %i nR: %i" % (cgrad, nodemax, self.nR)))

        # 6/17 THIS should check if the last node is high in energy
        if cgrad > CTHRESH and not self.nodes[self.nR-1].PES.lot.do_coupling and nodemax != self.TSnode:
            self.logger.log_print(" constraint gradient positive")
            ispast = 2
        elif ispast1 > 0 and cgrad > OTHRESH:
            self.logger.log_print(" over the hill(1)")
            ispast = 1
        elif ispast2 > 1:
            self.logger.log_print(" over the hill(2)")
            ispast = 1
        else:
            ispast = 0

        if ispast == 0:
            bch = self.check_for_reaction_g(1, self.driving_coords)
            if ispast3 > 1 and bch:
                self.logger.log_print("over the hill(3) connection changed %r " % bch)
                ispast = 3
        self.logger.log_print(" ispast=", ispast)
        return ispast

    def check_if_grown(self):
        '''
        Check if the string is grown
        Returns True if grown
        '''

        self.pastts = self.past_ts()
        isDone = False
        # TODO break planes
        condition1 = (abs(self.nodes[self.nR-1].bdist) <= (1-self.BDIST_RATIO)*abs(self.nodes[0].bdist))
        self.logger.log_print(" bdist %.3f" % self.nodes[self.nR-1].bdist)

        fp = self.find_peaks('growing')
        if self.pastts and self.current_nnodes > 3 and condition1:  # TODO extra criterion here
            self.logger.log_print(" pastts is ", self.pastts)
            if self.TSnode == self.nR-1:
                self.logger.log_print(" The highest energy node is the last")
                self.logger.log_print(" not continuing with TS optimization.")
                self.tscontinue = False
            nifty.printcool("Over the hill")
            isDone = True
        elif fp == -1 and self.energies[self.nR-1] > 200. and self.nodes[self.nR-1].gradrms > self.options['CONV_TOL']*5:
            self.logger.log_print("growth_iters over: all uphill and high energy")
            self.end_early = 2
            self.tscontinue = False
            # self.num_nodes = self.nR
            isDone = True
        elif fp == -2:
            self.logger.log_print("growth_iters over: all uphill and flattening out")
            self.end_early = 2
            self.tscontinue = False
            # self.num_nodes = self.nR
            isDone = True

        # ADD extra criteria here to check if TS is higher energy than product
        return isDone

    def is_converged(self, totalgrad, fp, rtype, ts_cgradq):

        if self.TSnode == self.num_nodes-2 and (self.find or totalgrad < 0.2) and fp == 1:
            if self.nodes[self.nR-1].gradrms > self.options['CONV_TOL']:
                self.logger.log_print("TS node is second to last node, adding one more node")
                self.add_last_node(1)
                # self.num_nodes = self.nR
                self.active[self.num_nodes-1] = False  # GSM makes self.active[self.num_nodes-1]=True as well
                self.active[self.num_nodes-2] = True  # GSM makes self.active[self.num_nodes-1]=True as well
                self.logger.log_print("done adding node")
                self.logger.log_print("nnodes = ", self.num_nodes)
                self.ictan, self.dqmaga = self.get_tangents(self.nodes)
                self.refresh_coordinates()
            return False

        # => check string profile <= #
        if fp == -1:  # total string is uphill
            self.logger.log_print("fp == -1, check V_profile")
            self.logger.log_print("total dissociation")
            self.endearly = True  # bools
            self.tscontinue = False
            return True
        elif fp == -2:
            self.logger.log_print("termination due to dissociation")
            self.tscontinue = False
            self.endearly = True  # bools
            return True
        elif fp == 0:
            self.tscontinue = False
            self.endearly = True  # bools
            return True
        elif self.climb and fp > 0 and self.finder:
            fp = self.find_peaks('opting')
            if fp > 1:
                rxnocc, wint = self.check_for_reaction()
            if fp > 1 and rxnocc and wint < self.num_nodes-1:
                self.logger.log_print("Need to trim string")
                self.tscontinue = False
                self.endearly = True  # bools
                return True
            else:
                return False
        else:
            return super(SE_GSM, self).is_converged(totalgrad, fp, rtype, ts_cgradq)

    def check_for_reaction(self):
        '''
        '''
        isrxn = self.check_for_reaction_g(1, self.driving_coords)
        minnodes = []
        maxnodes = []
        wint = 0
        energies = self.energies
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[self.num_nodes-1] < energies[self.num_nodes-2]:
            minnodes.append(self.num_nodes-1)
        for n in range(self.n0, self.num_nodes-1):
            if energies[n+1] > energies[n]:
                if energies[n] < energies[n-1]:
                    minnodes.append(n)
            if energies[n+1] < energies[n]:
                if energies[n] > energies[n-1]:
                    maxnodes.append(n)
        if len(minnodes) > 2 and len(maxnodes) > 1:
            wint = minnodes[1]  # the real reaction ends at first minimum
            self.logger.log_print(" wint ", wint)

        return isrxn, wint

    def check_for_reaction_g(self, rtype, driving_coords):
        '''
        '''

        c = Counter(elem[0] for elem in driving_coords)
        nadds = c['ADD']
        nbreaks = c['BREAK']
        isrxn = False

        if (nadds+nbreaks) < 1:
            return False
        nadded = 0
        nbroken = 0
        nnR = self.nR-1
        xyz = self.nodes[nnR].xyz
        atoms = self.nodes[nnR].atoms

        for i in driving_coords:
            if "ADD" in i:
                index = [i[1]-1, i[2]-1]
                bond = Distance(index[0], index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2
                if d < d0:
                    nadded += 1
            if "BREAK" in i:
                index = [i[1]-1, i[2]-1]
                bond = Distance(index[0], index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2
                if d > d0:
                    nbroken += 1
        if rtype == 1:
            if (nadded+nbroken) >= (nadds+nbreaks):
                isrxn = True
                # isrxn=nadded+nbroken
        else:
            isrxn = True
            # isrxn=nadded+nbroken
        self.logger.log_print(" check_for_reaction_g isrxn: %r nadd+nbrk: %i" % (isrxn, nadds+nbreaks))
        return isrxn

        # # => Convergence Criteria
        # dE_iter = abs(self.emaxp - self.emax)
        # TS_conv = self.options['CONV_TOL']
        # if self.find and self.optimizer[self.TSnode].nneg>1:
        #    print(" reducing TS convergence because nneg>1")
        #    TS_conv = self.options['CONV_TOL']/2.
        # self.optimizer[self.TSnode].conv_grms = TS_conv

        # if (rtype == 2 and self.find ):
        #    if self.nodes[self.TSnode].gradrms< TS_conv:
        #        self.tscontinue=False
        #        isDone=True
        #        #print(" Number of imaginary frequencies %i" % self.optimizer[self.TSnode].nneg)
        #        return isDone
        #    if totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*TS_conv and dE_iter < 0.02:
        #        self.tscontinue=False
        #        isDone=True
        #        #print(" Number of imaginary frequencies %i" % self.optimizer[self.TSnode].nneg)
        # if rtype==1 and self.climb:
        #    if self.nodes[self.TSnode].gradrms<TS_conv and ts_cgradq < self.options['CONV_TOL']:
        #        isDone=True

    def optimize_iteration(self, opt_steps):
        _, refE = super().optimize_iteration(opt_steps)

        if self.done_growing:
            fp = self.find_peaks('opting')
            if self.energies[self.num_nodes-1] > self.energies[self.num_nodes-2] and fp > 0 and self.nodes[self.num_nodes-1].gradrms > self.CONV_TOL:
                printcool('Last node is not a minimum, Might need to verify that the last node is a minimum')
                # path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, self.num_nodes-1))
                self.optimizer[self.num_nodes-1].optimize(
                    molecule=self.nodes[self.num_nodes-1],
                    refE=refE,
                    opt_type='UNCONSTRAINED',
                    opt_steps=osteps,
                    ictan=None,
                    # path=path
                )

    def get_tangent_vector_guess(self, nlist, n):
        ictan0, _ = self.get_tangent(
            node1=self.nodes[nlist[2 * n]],
            node2=self.nodes[nlist[2 * n + 1]],
            driving_coords=self.driving_coords,
        )
        return ictan0

if __name__ == '__main__':
    from .qchem import QChem
    from .pes import PES
    from .dlc_new import DelocalizedInternalCoordinates
    from .eigenvector_follow import eigenvector_follow
    from ._linesearch import backtrack, NoLineSearch
    from .molecule import Molecule

    basis = '6-31G'
    nproc = 8
    functional = 'B3LYP'
    filepath1 = "examples/tests/butadiene_ethene.xyz"
    lot1 = QChem.from_options(states=[(1, 0)], charge=0, basis=basis, functional=functional, nproc=nproc, fnm=filepath1)
    pes1 = PES.from_options(lot=lot1, ad_idx=0, multiplicity=1)
    M1 = Molecule.from_options(fnm=filepath1, PES=pes1, coordinate_type="DLC")
    optimizer = eigenvector_follow.from_options(print_level=1)  # default parameters fine here/opt_type will get set by GSM

    gsm = SE_GSM.from_options(reactant=M1, nnodes=20, driving_coords=[("ADD", 6, 4), ("ADD", 5, 1)], optimizer=optimizer, print_level=1)
    gsm.go_gsm()
