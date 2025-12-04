from __future__ import print_function

import traceback as tb

from ..molecule import Molecule
from .gsm import NodeAdditionStrategy, TSOptimizationStrategy
from .main_gsm import MainGSM
# from ..utilities import Devutils as dev

__all__ = [
    "DE_GSM"
]

class DE_GSM(MainGSM):
    default_rtype = TSOptimizationStrategy(2)
    default_max_opt_steps = 20

    @classmethod
    def preadjust_nodes(cls, nodes, evaluator, driving_coords):
        # Add bonds to top1 that are present in top2
        # It's not clear if we should form the topology so the bonds
        # are the same since this might affect the Primitives of the xyz1 (slightly)
        # Later we stil need to form the union of bonds, angles and torsions
        # However, I think this is important, the way its formulated, for identifiyin
        # the number of fragments and blocks, which is used in hybrid TRIC.
        reactant:Molecule = nodes[0]
        product:Molecule = nodes[-1]

        base_edges = list(reactant.bond_graph.edges())
        target_edges = product.bond_graph.edges()
        nodes = cls.add_bonds_to_nodes(
            nodes,
            base_edges,
            target_edges
        )

        nodes = cls.set_consistent_node_dlcs(nodes)

        nodes = cls.add_evaluator_to_nodes(nodes, evaluator)

        return nodes

    def _handle_initial_growth(self, *, max_iters, opt_steps, reparameterize=True):
        if self.growth_direction == NodeAdditionStrategy.Normal:
            self.add_GSM_nodes(2)
        elif self.growth_direction == NodeAdditionStrategy.Reactant:
            self.add_GSM_nodeR(1)
        elif self.growth_direction == NodeAdditionStrategy.Product:
            self.add_GSM_nodeP(1)

        # Grow String
        self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
        self.logger.log_print("Done Growing the String!!!")

        if reparameterize:
            self.reparameterize()

    def _optimize_string_for_ts(self, *, max_iters, opt_steps, rtype):
        try:
            self.optimize_string(max_iter=max_iters, opt_steps=opt_steps, rtype=rtype)
        except ValueError as error:
            # self.logger.log_print(tb.format_exc())
            if str(error) == "Ran out of iterations":
                self.end_early = True
            else:
                raise

    def add_GSM_nodes(self, newnodes=1):
        if self.current_nnodes+newnodes > self.num_nodes:
            self.logger.log_print("Adding too many nodes, cannot add_GSM_node")
        sign = -1
        for i in range(newnodes):
            sign *= -1
            if sign == 1:
                self.add_GSM_nodeR()
            else:
                self.add_GSM_nodeP()

    def set_active(self, nR, nP):
        # print(" Here is active:",self.active)
        if nR != nP and self.growth_direction  == NodeAdditionStrategy.Normal:
            self.logger.log_print(" setting active nodes to {nR} and {nP}", nR=nR, nP=nP)
        elif self.growth_direction  == NodeAdditionStrategy.Reactant:
            self.logger.log_print(" setting active node to {nR} ", nR=nR)
        elif self.growth_direction  == NodeAdditionStrategy.Product:
            self.logger.log_print(" setting active node to {nP} ", nP=nP)
        else:
            self.logger.log_print(" setting active node to {nR} ", nR=nR)

        for i in range(self.num_nodes):
            if self.nodes[i] is not None:
                self.optimizer[i].conv_grms = self.tolerances['CONV_TOL']*2.
        self.optimizer[nR].conv_grms = self.tolerances['ADD_NODE_TOL']
        self.optimizer[nP].conv_grms = self.tolerances['ADD_NODE_TOL']
        self.logger.log_print(" conv_tol of node %d is %.4f" % (nR, self.optimizer[nR].conv_grms))
        self.logger.log_print(" conv_tol of node %d is %.4f" % (nP, self.optimizer[nP].conv_grms))
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction  == NodeAdditionStrategy.Reactant:
            self.active[nP] = False
        if self.growth_direction  == NodeAdditionStrategy.Product:
            self.active[nR] = False
        # print(" Here is new active:",self.active)

    def check_if_grown(self):
        '''
        Check if the string is grown
        Returns True if grown 
        '''

        return self.current_nnodes == self.num_nodes

    def grow_nodes(self):
        '''
        Grow nodes
        '''

        if self.nodes[self.nR-1].gradrms < self.tolerances['ADD_NODE_TOL'] and self.growth_direction  != NodeAdditionStrategy.Product:
            if self.nodes[self.nR] is None:
                self.add_GSM_nodeR()
                self.logger.log_print(
                    " getting energy for node {nR}: {dE:5.4f}",
                    nR=self.nR-1,
                    dE=self.nodes[self.nR-1].energy - self.nodes[0].V0
                )
        if self.nodes[self.num_nodes-self.nP].gradrms < self.tolerances['ADD_NODE_TOL'] and self.growth_direction  != NodeAdditionStrategy.Reactant:
            if self.nodes[-self.nP-1] is None:
                self.add_GSM_nodeP()
                self.logger.log_print(
                    " getting energy for node {node_num}: {E:5.4f}",
                    node_num=self.num_nodes-self.nP,
                    E=self.nodes[-self.nP].energy - self.nodes[0].V0
                )
        return

    def make_tan_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent-2):
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

    def get_tangent_vector_guess(self, nlist, n):
        # print(" getting tangent [%i ]from between %i %i pointing towards %i" % (nlist[2 * n], nlist[2 * n],
        #                                                                         nlist[2 * n + 1], nlist[2 * n]))
        return self.get_tangent_xyz(self.nodes[nlist[2 * n]].xyz,
                                      self.nodes[nlist[2 * n + 1]].xyz,
                                      self.nodes[0].primitive_internal_coordinates)

    def make_difference_node_list(self):
        '''
        Returns ncurrent and a list of indices that can be iterated over to produce
        tangents for the string pathway.
        '''
        # TODO: THis can probably be done more succinctly using a list of tuples
        ncurrent = 0
        nlist = [0]*(2*self.num_nodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1

        for n in range(self.num_nodes-self.nP+1, self.num_nodes):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n-1
            ncurrent += 1

        nlist[2*ncurrent] = self.nR - 1
        nlist[2*ncurrent+1] = self.num_nodes - self.nP

        if False:
            nlist[2*ncurrent+1] = self.nR - 2  # for isMAP_SE

        # TODO is this actually used?
        # if self.nR == 0: nlist[2*ncurrent] += 1
        # if self.nP == 0: nlist[2*ncurrent+1] -= 1
        ncurrent += 1
        nlist[2*ncurrent] = self.num_nodes - self.nP
        nlist[2*ncurrent+1] = self.nR-1
        # #TODO is this actually used?
        # if self.nR == 0: nlist[2*ncurrent+1] += 1
        # if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1

        return ncurrent, nlist

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy

        # TODO should be actual gradient
        self.nodes[0].gradrms = 0.
        if self.growth_direction  != NodeAdditionStrategy.Reactant:
            self.nodes[-1].gradrms = 0.
            self.logger.log_print(" Energy of the end points are %4.3f, %4.3f" % (self.nodes[0].energy, self.nodes[-1].energy))
            self.logger.log_print(" relative E %4.3f, %4.3f" % (0.0, self.nodes[-1].energy-self.nodes[0].energy))
        else:
            self.logger.log_print(" Energy of end points are %4.3f " % self.nodes[0].energy)
            # self.nodes[-1].energy = self.nodes[0].energy
            # self.nodes[-1].gradrms = 0.