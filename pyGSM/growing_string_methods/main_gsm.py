from __future__ import print_function

import abc

import numpy as np
import os
import traceback as tb

import multiprocessing as mp
from itertools import chain

from . import TSOptimizationStrategy
from .gsm import GSM, NodeAdditionStrategy
from ..optimizers import eigenvector_follow
from ..utilities import Devutils as dev, math_utils, units, block_matrix
from ..molecule import Molecule

def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj, methname)(*arg[2:])

__all__ = [
    "MainGSM"
]

class MainGSM(GSM):

    @abc.abstractmethod
    def make_difference_node_list(self):
        ...

    def grow_and_handle_termination(self) -> bool:
        self.grow_nodes()
        return False

    def grow_string(self,
                    max_iters=30,
                    max_opt_steps=3,
                    nconstraints=1,
                    logger=None
                    ):
        '''
        Grow the string

        Parameters
        ----------
        max_iter : int
             Maximum number of GSM iterations
        nconstraints : int
        optsteps : int
            Maximum number of optimization steps per node of string
        '''
        with self.logger.block(tag="In growth_iters"):

            ncurrent, nlist = self.make_difference_node_list()
            self.ictan, self.dqmaga = self.get_tangents_growing()
            self.refresh_coordinates()
            self.set_active(self.nR-1, self.num_nodes-self.nP)

            isGrown = False
            iteration = 0
            while not isGrown:
                if iteration > max_iters:
                    self.logger.log_print(" Ran out of iterations")
                    return
                    # raise Exception(" Ran out of iterations")
                self.logger.log_print("Starting growth iteration {iteration}", iteration=iteration)
                self.optimize_iteration(max_opt_steps)
                totalgrad, gradrms, sum_gradrms = self.calc_optimization_metrics(self.nodes)

                self.checkpointer[f"growth_iters_{iteration}"] = {
                    "coords":np.array(self.geometries),
                    "energies":np.array(self.energies),
                    "gradrms":np.array(self.gradrmss)
                }
                # self.scratch_writer('growth_iters_{:03}.xyz'.format(iteration),
                #                     self.nodes[0].atoms,
                #                     self.geometries, self.energies, self.gradrmss, self.dEs)
                self.logger.log_print(
                    "gopt_iter: {iteration:2} totalgrad: {totalgrad:4.3} gradrms: {gradrms:5.4} max E: {emax:5.4}",
                    iteration=iteration,
                    totalgrad=totalgrad,
                    gradrms=gradrms,
                    emax=self.emax
                )

                needs_term = self.grow_and_handle_termination()
                if needs_term:
                    break

                self.set_active(self.nR-1, self.num_nodes-self.nP)
                self.ic_reparam_g()
                self.ictan, self.dqmaga = self.get_tangents_growing()
                self.refresh_coordinates()

                iteration += 1
                isGrown = self.check_if_grown()

            # create newic object
            self.logger.log_print(" creating newic molecule--used for ic_reparam")
            # self.newic = self.nodes[0].copy()

            # TODO should something be done for growthdirection 2?
            if self.growth_direction == NodeAdditionStrategy.Reactant:
                self.logger.log_print("Setting LOT of last node")
                self.nodes[-1] = self.nodes[-2].modify(xyz=self.nodes[-1].xyz)
            return

    def optimize_string(self, max_iter=30, nconstraints=1, opt_steps=1, rtype=None):
        '''
        Optimize the grown string until convergence

        Parameters
        ----------
        max_iter : int
             Maximum number of GSM iterations 
        nconstraints : int
        optsteps : int
            Maximum number of optimization steps per node of string
        rtype : int
            An option to change how GSM optimizes  
            TODO change this s***
            0 is no-climb
            1 is climber
            2 is finder
        '''
        if rtype is None:
            rtype = self.rtype

        if self.only_drive:
            raise NotImplementedError("need to handle plain interpolation")

        self.nclimb = 0
        self.nhessreset = 10  # are these used??? TODO
        self.hessrcount = 0   # are these used?!  TODO
        self.newclimbscale = 2.
        self.set_finder(rtype)

        self.isConverged = False

        # enter loop
        self.logger.log_print("performing opts up to {max_iter} iterations", max_iter=max_iter)
        for oi in range(max_iter):
            with self.logger.block(tag="Starting opt iter {oi}", oi=oi):
                if self.climb and not self.find:
                    self.logger.log_print("CLIMBING")
                elif self.find:
                    self.logger.log_print("TS SEARCHING")

                # stash previous TSnode
                self.pTSnode = self.TSnode
                self.emaxp = self.emax
                ts_node_changed = False

                # store reparam energies
                self.logger.log_print(
                    [
                    "V_profile (beginning of iteration): {eng_list}"
                    ],
                    eng_list=self.energies,
                    preformatter=lambda *, eng_list, **kw:dict(kw, eng_list=' '.join(f"{e:7.3f}" for e in eng_list))
                )

                # => Get all tangents 3-way <= #
                self.get_tangents_opting()
                self.refresh_coordinates()

                # => do opt steps <= #
                self.set_node_convergence()
                self.optimize_iteration(opt_steps)

                self.logger.log_print(
                    [
                    "V_profile: {eng_list}"
                    ],
                    eng_list=self.energies,
                    preformatter=lambda *, eng_list, **kw:dict(kw, eng_list=' '.join(f"{e:7.3f}" for e in eng_list))
                )

                # TODO resetting
                # TODO special SSM criteria if first opt'd node is too high?
                if self.TSnode == self.num_nodes-2 and (self.climb or self.find):
                    self.logger.log_print("WARNING\n: TS node shouldn't be second to last node for tangent reasons")
                    self.add_node_after_TS()
                    added = True
                elif self.TSnode == 1 and (self.climb or self.find):
                    self.logger.log_print("WARNING\n: TS node shouldn't be first  node for tangent reasons")
                    self.add_node_before_TS()
                    added = True
                else:
                    added = False

                # => find peaks <= #
                fp = self.find_peaks('opting')

                ts_cgradq = 0.
                if not self.find:
                    ts_cgradq = np.linalg.norm(
                        np.dot(self.nodes[self.TSnode].gradient.T,
                               self.nodes[self.TSnode].constraints[:, 0])*self.nodes[self.TSnode].constraints[:, 0])
                    self.logger.log_print(" ts_cgradq %5.4f" % ts_cgradq)

                ts_gradrms = self.nodes[self.TSnode].gradrms
                self.dE_iter = abs(self.emax-self.emaxp)
                self.logger.log_print(" dE_iter ={:2.2f}".format(self.dE_iter))

                # => calculate totalgrad <= #
                totalgrad, gradrms, sum_gradrms = self.calc_optimization_metrics(self.nodes)

                # Check if allup or alldown
                energies = np.array(self.energies)
                if (np.all(energies[1:]+0.5 >= energies[:-1]) or np.all(energies[1:]-0.5 <= energies[:-1])) and (self.climber or self.finder):
                    self.logger.log_printcool(" There is no TS, turning off TS search")
                    rtype = 0
                    self.climber = self.finder = self.find = self.climb = False
                    self.tolerances['CONV_TOL'] = self.tolerances['CONV_TOL']*5

                # if self.has_intermediate(5) and rtype>0 and (self.climb or self.find):
                #    printcool(" THERE IS AN INTERMEDIATE, OPTIMIZE THE INTERMEDIATE AND TRY AGAIN")
                #    self.endearly=True
                #    isConverged=True
                #    self.tscontinue=False

                # => Check Convergence <= #
                self.isConverged = self.is_converged(totalgrad, fp, rtype, ts_cgradq)

                # => set stage <= #
                stage_changed = self.set_stage(totalgrad, sum_gradrms, ts_cgradq, ts_gradrms, fp)

                if not stage_changed:
                    # Decrement stuff that controls stage
                    if self.climb:
                        self.nclimb -= 1
                    self.nhessreset -= 1
                    if self.nopt_intermediate > 0:
                        self.nopt_intermediate -= 1

                    if self.pTSnode != self.TSnode and self.climb:
                        self.logger.log_print("TS node changed after opting")
                        self.climb = False
                        #self.slow_down_climb()
                        ts_node_changed = True
                        self.pTSnode = self.TSnode

                    # opt decided Hess is not good because of overlap
                    if self.find and (not self.optimizer[self.TSnode].maxol_good or added):
                        self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                        self.modify_TS_Hess()
                    elif self.find and (self.optimizer[self.TSnode].nneg > 3 or self.optimizer[self.TSnode].nneg == 0 or self.hess_counter > 10 or np.abs(self.TS_E_0 - self.emax) > 10.) and not self.optimizer[self.TSnode].converged:

                        # Reform the guess primitive Hessian
                        self.nodes[self.TSnode].form_Primitive_Hessian()
                        if self.hessrcount < 1 and self.pTSnode == self.TSnode:
                            self.logger.log_print(" resetting TS node coords Ut (and Hessian)")
                            self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                            self.modify_TS_Hess()
                            self.nhessreset = 10
                            self.hessrcount = 1
                        else:
                            self.logger.log_print(" Hessian consistently bad, going back to climb (for 3 iterations)")
                            self.find = False
                            self.nclimb = 2
                    elif self.find and self.optimizer[self.TSnode].nneg <= 3:
                        self.hessrcount -= 1
                        self.hess_counter += 1
                self.logger.log_print(f'{stage_changed=}: {self.climb=} {self.find=}')

                # => write Convergence to file <= #
                self.checkpointer[f"opt_iter_{oi}"] = {
                    "coords":np.array(self.geometries),
                    "energies":np.array(self.energies),
                    "gradrms":np.array(self.gradrmss)
                }
                # self.scratch_writer(
                #     'opt_iters_{:03}.xyz'.format( oi),
                #     self.geometries, self.energies, self.gradrmss, self.dEs
                # )

                self.logger.log_print(" End early counter {}".format(self.endearly_counter))

                # TODO prints tgrads and jobGradCount
                self.logger.log_print("opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}".format(oi, float(totalgrad), float(gradrms), self.TSnode, float(self.emax)))
                oi += 1

                if self.isConverged and not added and not ts_node_changed and not stage_changed:
                    self.logger.log_print("Converged")
                    return

                # => Reparam the String <= #
                if oi < max_iter and not self.isConverged and not stage_changed:
                    self.reparameterize(nconstraints=nconstraints)
                    self.get_tangents_opting()
                    self.refresh_coordinates()
                    if self.pTSnode != self.TSnode and self.climb:
                        self.logger.log_print("TS node changed after reparameterizing")
                        self.slow_down_climb()
                elif oi == max_iter and not self.isConverged:
                    self.ran_out = True
                    self.logger.log_print(" Ran out of iterations")
                    return
                    # raise Exception(" Ran out of iterations")

            # TODO Optimize TS node to a finer convergence
            # if rtype==2:
            # return

    def refresh_coordinates(self, update_TS=False):
        '''
        Refresh the DLC coordinates for the string
        '''

        if not self.done_growing:
            # TODO

            if self.mp_cores == 1:
                for n in range(1, self.num_nodes-1):
                    if self.nodes[n] is not None:
                        self.nodes[n] = self.nodes[n].modify_coordinate_system(
                            constraints=self.ictan[n]
                        )
                        # Vecs = self.newic.modify_.build_dlc(self.nodes[n].xyz, self.ictan[n],
                        #                                       logger=self.logger
                        #                                       )
                        # self.nodes[n].coord_basis = Vecs

            else:
                pool = mp.Pool(self.mp_cores)
                Vecs = pool.map(worker, ((self.newic.coord_obj, "update_dlc", self.nodes[n].xyz, self.ictan[n]) for n in range(1, self.num_nodes-1) if self.nodes[n] is not None))
                pool.close()
                pool.join()

                i = 0
                for n in range(1, self.num_nodes-1):
                    if self.nodes[n] is not None:
                        self.nodes[n].coord_basis = Vecs[i]
                        i += 1
        else:
            if self.find or self.climb:
                TSnode = self.TSnode
                # if self.mp_cores == 1:
                for n in range(1, self.num_nodes-1):
                    # don't update tsnode coord basis
                    if n != TSnode or (n == TSnode and update_TS):
                        self.nodes[n].update_coordinate_basis(self.nodes[n].xyz, constraints=self.ictan[n])
                # else:
                #     ...
                    # pool = mp.Pool(self.mp_cores)
                    # Vecs = pool.map(worker, ((self.newic.coord_obj, "update_dlc", self.nodes[n].xyz, self.ictan[n]) for n in range(1, self.num_nodes-1) if n != TSnode))
                    # pool.close()
                    # pool.join()
                    # for i, n in enumerate(chain(range(1, TSnode), range(TSnode+1, self.num_nodes-1))):
                    #     self.nodes[n].coord_basis = Vecs[i]
                    # if update_TS:
                    #     self.nodes[TSnode].update_coordinate_basis(self.nodes[TSnode].xyz, self.ictan[TSnode])

            else:
                # if self.mp_cores == 1:
                #     Vecs = []
                #     for n in range(1, self.num_nodes-1):
                #         Vecs.append(
                #             self.newic.coord_obj.update_dlc(self.nodes[n].xyz, self.ictan[n])
                #         )
                # elif self.mp_cores > 1:
                #     pool = mp.Pool(self.mp_cores)
                #     Vecs = pool.map(worker, ((self.newic.coord_obj, "update_dlc", self.nodes[n].xyz, self.ictan[n]) for n in range(1, self.num_nodes-1)))
                #     pool.close()
                #     pool.join()
                for node, ictan in zip(self.nodes[1:self.num_nodes-1], self.ictan[1:self.num_nodes-1]):
                    node.update_coordinate_basis(node.xyz, constraints=ictan)

    def optimize_iteration(self, opt_steps):
        '''
        Optimize string iteration
        '''

        refE = self.nodes[0].energy

        for n in range(self.num_nodes):
            if self.nodes[n] and self.active[n]:
                with self.logger.block(tag="Optimizing node {n}", n=n):
                    opt_type = self.set_opt_type(n)
                    self.logger.log_print("setting node {n} opt_type to {opt_type}", n=n, opt_type=opt_type)
                    osteps = self.mult_steps(n, opt_steps)
                    self.optimizer[n].optimize(
                        molecule=self.nodes[n],
                        refE=refE,
                        opt_type=opt_type,
                        opt_steps=osteps,
                        ictan=self.ictan[n],
                        xyzframerate=1
                    )

        return self.nodes[0].energy, refE
        # return self.isConverged, refE

    def get_tangents_opting(self, print_level=1):
        if self.climb or self.find:
            self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
        else:
            self.ictan, self.dqmaga = self.get_tangents(self.nodes)

    @abc.abstractmethod
    def get_tangent_vector_guess(self, nlist, n):
        ...

    @classmethod
    def _array_split(cls, vec, max_el):
        l = len(vec)
        return np.array_split(np.arange(l), l // max_el + (l % max_el > 0))
    @classmethod
    def _format_block_str(cls, vec, n, item_fmt, join_fmt):
        blocks = cls._array_split(vec, n)
        return "\n".join(
            join_fmt.join(item_fmt.format(f) for f in b)
            for b in blocks
        )

    def get_tangents_growing(self, print_level=1):
        """
        Finds the tangents during the growth phase. 
        Tangents referenced to left or right during growing phase.
        Also updates coordinates
        Not a static method beause no one should ever call this outside of GSM
        """

        ncurrent, nlist = self.make_difference_node_list()
        dqmaga = [0.]*self.num_nodes
        ictan = [None]*self.num_nodes

        self.logger.log_print(
            [
                "ncurrent {ncurrent}",
                "nlist {nlist}"
            ],
            ncurrent=ncurrent,
            nlist=nlist,
            log_level=self.logger.LogLevel.Debug
        )

        for n in range(ncurrent):
            # ictan0,_ = self.get_tangent(
            #        node1=self.nodes[nlist[2*n]],
            #        node2=self.nodes[nlist[2*n+1]],
            #        driving_coords=self.driving_coords,
            #        )

            ictan0 = self.get_tangent_vector_guess(nlist, n)
            if math_utils.is_zero_array(ictan0[:]):
                self.logger.log_print([
                    " ICTAN IS ZERO!",
                    "{ni}",
                    "{nin}",
                ],
                    ni=nlist[2*n],
                    nin=nlist[2*n+1]
                )
                raise ValueError("constraint tangent vector is zero")

            self.logger.log_print("forming space for {ni}", ni=nlist[2*n], log_level=self.logger.LogLevel.MoreDebug)
            self.logger.log_print("forming space for {ni}", ni=nlist[2*n+1], log_level=self.logger.LogLevel.MoreDebug)

            # normalize ictan
            norm = np.linalg.norm(ictan0)
            ictan[nlist[2*n]] = ictan0/norm

            # NOTE regular GSM does something weird here
            # Vecs = self.nodes[nlist[2*n]].update_coordinate_basis(constraints=self.ictan[nlist[2*n]])
            # constraint = self.nodes[nlist[2*n]].constraints
            # prim_constraint = block_matrix.dot(Vecs,constraint)
            # but this is not followed here anymore 7/1/2020
            # dqmaga[nlist[2*n]] = np.dot(prim_constraint.T,ictan0)
            # dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))
            # tmp_dqmaga = np.dot(prim_constraint.T,ictan0)
            # tmp_dqmaga = np.sqrt(tmp_dqmaga)

            dqmaga[nlist[2*n]] = norm

        self.logger.log_print(
            [
                "printing dqmaga",
                "{dqmaga_str}"
            ],
            dqmaga=dqmaga,
            preformatter=lambda *, dqmaga, **kwargs: dict(
                kwargs,
                dqmaga_str=self._format_block_str(dqmaga, 5, "{:5.3}", " ")),
            log_level=self.logger.LogLevel.Debug
        )

        # if print_level > 1:
        #     for n in range(ncurrent):
        #         self.logger.log_print("dqmag[%i] =%1.2f" % (nlist[2*n], self.dqmaga[nlist[2*n]]))
        #         self.logger.log_print("printing ictan[%i]" % nlist[2*n])
        #         self.logger.log_print(self.ictan[nlist[2*n]].T)
        for i, tan in enumerate(ictan):
            if tan is not None and math_utils.is_zero_array(tan):
                raise ValueError(f"tangent vector {i} is zero")

        return ictan, dqmaga

    # Refactor this code!
    # TODO remove return form_TS hess  3/2021
    def set_stage(self, totalgrad, sumgradrms, ts_cgradq, ts_gradrms, fp):

        # checking sum gradrms is not good because if one node is converged a lot while others a re not this is bad
        self.logger.log_print('In set stage')
        all_converged = all([self.nodes[n].gradrms < self.optimizer[n].conv_grms*1.1 for n in range(1, self.num_nodes-1)])
        all_converged_climb = all([self.nodes[n].gradrms < self.optimizer[n].conv_grms*2.5 for n in range(1, self.num_nodes-1)])
        stage_changed = False

        # TODO totalgrad is not a good criteria for large systems
        # if fp>0 and (((totalgrad < 0.3 or ts_cgradq < 0.01) and self.dE_iter < 2.) or all_converged) and self.nopt_intermediate<1: # extra criterion in og-gsm for added

        self.logger.log_print('set stage ts_cgradq {ts_cgradq}', ts_cgradq=ts_cgradq)

        if fp > 0: # and all_converged_climb and self.dE_iter < 2.:  # and self.nopt_intermediate<1:
            if not self.climb and self.climber:
                self.logger.log_print(" ** starting climb **")
                self.climb = True
                self.logger.log_print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" % (totalgrad, ts_gradrms, ts_cgradq))
                # overwrite this here just in case TSnode changed wont cause slow down climb
                self.pTSnode = self.TSnode
                stage_changed = True

            # TODO deserves to be rethought 3/2021
                     #(totalgrad < 0.2 and ts_gradrms < self.tolerances['CONV_TOL']*20. and ts_cgradq < 0.02) or  #
            elif (self.climb and not self.find and self.finder and self.nclimb < 1 and
                    ((totalgrad < 0.3 and ts_gradrms < self.tolerances['CONV_TOL']*20.) or  # I hate totalgrad  and ts_cgradq < 0.02
                     (all_converged) or
                     (ts_gradrms < self.tolerances['CONV_TOL']*2.5 and ts_cgradq < 0.01)  # used to be 5
                     )) and self.dE_iter < 5.:
                self.logger.log_print(" ** starting exact climb **")
                self.logger.log_print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" % (totalgrad, ts_gradrms, ts_cgradq))
                self.find = True

                # Modify TS Hessian
                self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                self.modify_TS_Hess()

                if self.optimizer[self.TSnode].max_step > 0.1:
                    self.optimizer[self.TSnode].max_step = 0.1
                # this is likely to break but I don't want to tweak this kid's code too much
                self.optimizer[self.TSnode] = eigenvector_follow(**self.optimizer[self.TSnode].get_state_dict())
                self.optimizer[self.TSnode].SCALEQN = 1.
                self.nhessreset = 10  # are these used??? TODO
                self.hessrcount = 0   # are these used?!  TODO
                stage_changed = True

        return stage_changed

    def add_GSM_nodeR(self, newnodes=1):
        '''
        Add a node between endpoints on the reactant side, should only be called inside GSM
        '''
        with self.logger.block(tag="Adding reactant node"):

            if self.current_nnodes+newnodes > self.num_nodes:
                raise ValueError("Adding too many nodes, cannot interpolate")
            for i in range(newnodes):
                iR = self.nR-1
                iP = self.num_nodes-self.nP
                iN = self.nR
                self.logger.log_print(" adding node: {iN} between {iR} {iP} from {iR}", iN=iN, iR=iR, iP=iP)
                if self.num_nodes - self.current_nnodes > 1:
                    stepsize = 1./float(self.num_nodes-self.current_nnodes+1)
                else:
                    stepsize = 0.5

                self.nodes[self.nR] = self.add_node(
                    self.nodes[iR],
                    self.nodes[iP],
                    stepsize,
                    iN,
                    node_idR=iR,
                    node_idP=iP,
                    DQMAG_MAX=self.DQMAG_MAX,
                    DQMAG_MIN=self.DQMAG_MIN,
                    driving_coords=self.driving_coords,
                )

                if self.nodes[self.nR] is None:
                    raise ValueError('Ran out of space')

                self.set_new_node_tolerances(self.nR)

                self.current_nnodes += 1
                self.nR += 1
                self.logger.log_print(" nn={nn},nR={nR}", nn=self.current_nnodes, nR=self.nR)
                self.active[self.nR-1] = True

                # align center of mass  and rotation
                # print("%i %i %i" %(iR,iP,iN))

                # print(" Aligning")
                # self.nodes[self.nR-1].xyz = self.com_rotate_move(iR,iP,iN)

    def set_new_node_tolerances(self, index):
        self.optimizer[index].max_step = self.optimizer[index - 1].max_step

    def add_GSM_nodeP(self, newnodes=1):
        '''
        Add a node between endpoints on the product side, should only be called inside GSM
        '''
        with self.logger.block(tag="Adding product node"):
            if self.current_nnodes+newnodes > self.num_nodes:
                raise ValueError("Adding too many nodes, cannot interpolate")

            for i in range(newnodes):
                # self.nodes[-self.nP-1] = BaseClass.add_node(self.num_nodes-self.nP,self.num_nodes-self.nP-1,self.num_nodes-self.nP)
                n1 = self.num_nodes-self.nP
                n2 = self.num_nodes-self.nP-1
                n3 = self.nR-1
                self.logger.log_print(" adding node: {n2} between {n1} {n3} from {n1}", n2=n2, n1=n1, n3=n3)
                if self.num_nodes - self.current_nnodes > 1:
                    stepsize = 1./float(self.num_nodes-self.current_nnodes+1)
                else:
                    stepsize = 0.5

                self.nodes[-self.nP-1] = GSM.add_node(
                    self.nodes[n1],
                    self.nodes[n3],
                    stepsize,
                    n2,
                    node_idR=n1,
                    node_idP=n3
                )
                if self.nodes[-self.nP-1] is None:
                    raise Exception('Ran out of space')

                self.optimizer[n2].max_step = self.optimizer[n1].max_step
                self.current_nnodes += 1
                self.nP += 1
                self.logger.log_print(" nn={nn},nP={nP}", nn=self.current_nnodes, nP=self.nP)
                self.active[-self.nP] = True

                # align center of mass  and rotation
                # print("%i %i %i" %(n1,n3,n2))
                # print(" Aligning")
                # self.nodes[-self.nP].xyz = self.com_rotate_move(n1,n3,n2)
                # print(" getting energy for node %d: %5.4f" %(self.num_nodes-self.nP,self.nodes[-self.nP].energy - self.nodes[0].V0))
            return

    def reparameterize(self, ic_reparam_steps=8, n0=0, nconstraints=1):
        '''
        Reparameterize the string
        '''
        if self.interp_method == 'DLC':
            # print('reparameterizing')
            self.ic_reparam(nodes=self.nodes, energies=self.energies, climbing=(self.climb or self.find), ic_reparam_steps=ic_reparam_steps, NUM_CORE=self.mp_cores)
        return

    def ic_reparam_g(self, ic_reparam_steps=4, n0=0, reparam_interior=True):  # see line 3863 of gstring.cpp
        """
        Reparameterize during growth phase
        """

        with self.logger.block(tag="Reparamerizing string nodes"):
            # close_dist_fix(0) #done here in GString line 3427.
            rpmove = np.zeros(self.num_nodes)
            rpart = np.zeros(self.num_nodes)
            disprms = 0.0

            if self.current_nnodes == self.num_nodes:
                return

            for i in range(ic_reparam_steps):
                self.ictan, self.dqmaga = self.get_tangents_growing()
                totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.num_nodes-self.nP+1:self.num_nodes])
                if i == 0:
                    self.logger.log_print(" totaldqmag (without inner): {totaldqmag:1.2}", totaldqmag=totaldqmag,
                                          log_level=self.logger.LogLevel.Debug)
                self.logger.log_print(
                    [
                        " printing spacings dqmaga: ",
                        "{dqmaga_str}"
                    ],
                    dqmaga_str=self.dqmaga,
                    preformatter=lambda *, dqmaga, **kwargs: dict(
                        kwargs,
                        dqmaga_str=self._format_block_str(dqmaga, 5, "{:5.3}", " ")),
                    log_level=self.logger.LogLevel.Debug
                )

                if i == 0:
                    if self.current_nnodes != self.num_nodes:
                        rpart = np.zeros(self.num_nodes)
                        for n in range(n0+1, self.nR):
                            rpart[n] = 1.0/(self.current_nnodes-2)
                        for n in range(self.num_nodes-self.nP, self.num_nodes-1):
                            rpart[n] = 1.0/(self.current_nnodes-2)
                    else:
                        for n in range(n0+1, self.num_nodes):
                            rpart[n] = 1./(self.num_nodes-1)
                    if i == 0:
                        self.logger.log_print(
                            [
                                " rpart: ",
                                "{rpart}",
                            ],
                            rpart=rpart,
                            preformatter=lambda *, rpart, **kwargs: dict(
                                kwargs,
                                rpart_str=self._format_block_str(rpart, 5, "{:1.2}", " ")),
                            log_level=self.logger.LogLevel.Debug
                        )
                nR0 = self.nR
                nP0 = self.nP

                # TODO CRA 3/2019 why is this here?
                if not reparam_interior:
                    if self.num_nodes-self.current_nnodes > 2:
                        nR0 -= 1
                        nP0 -= 1

                deltadq = 0.0
                for n in range(n0+1, nR0):
                    deltadq = self.dqmaga[n-1] - totaldqmag*rpart[n]
                    rpmove[n] = -deltadq
                for n in range(self.num_nodes-nP0, self.num_nodes-1):
                    deltadq = self.dqmaga[n+1] - totaldqmag*rpart[n]
                    rpmove[n] = -deltadq

                MAXRE = 1.1

                for n in range(n0+1, self.num_nodes-1):
                    if abs(rpmove[n]) > MAXRE:
                        rpmove[n] = float(np.sign(rpmove[n])*MAXRE)

                disprms = float(np.linalg.norm(rpmove[n0+1:self.num_nodes-1]))
                self.logger.log_print(
                    [
                        "{rpmove}",
                    ],
                    rpmove=rpmove[n0+1:],
                    preformatter=lambda *, rpmove, **kwargs: dict(
                        kwargs,
                        rpart_str=self._format_block_str(rpart, 5, "{:1.2}", " ")),
                    log_level=self.logger.LogLevel.Debug
                )
                # if self.print_level > 0:
                #     for n in range(n0+1, self.num_nodes-1):
                #         self.logger.log_print(" disp[{}]: {:1.2f}".format(n, rpmove[n]), end=' ')
                #         if (n) % 5 == 0:
                #             self.logger.log_print()
                #     print()
                #     print(" disprms: {:1.3}\n".format(disprms))

                if disprms < 1e-2:
                    break

                move_list = self.make_move_list()
                tan_list = self.make_tan_list()

                if self.mp_cores > 1:
                    pool = mp.Pool(self.mp_cores)
                    Vecs = pool.map(worker, ((self.nodes[0].coord_obj, "update_dlc", self.nodes[n].xyz, self.ictan[ntan]) for n, ntan in zip(move_list, tan_list) if rpmove[n] < 0))
                    pool.close()
                    pool.join()

                    i = 0
                    for n in move_list:
                        if rpmove[n] < 0:
                            self.nodes[n].coord_basis = Vecs[i]
                            i += 1

                    # move the positions
                    pool = mp.Pool(self.mp_cores)
                    newXyzs = pool.map(worker, ((self.nodes[n].coord_obj, "newCartesian", self.nodes[n].xyz, rpmove[n]*self.nodes[n].constraints[:, 0]) for n in move_list if rpmove[n] < 0))
                    pool.close()
                    pool.join()
                    i = 0
                    for n in move_list:
                        if rpmove[n] < 0:
                            self.nodes[n].xyz = newXyzs[i]
                            i += 1
                else:
                    for nmove, ntan in zip(move_list, tan_list):
                        if rpmove[nmove] < 0:
                            self.logger.log_print('Moving {nmove} along ictan[{ntan}]', nmove=nmove, ntan=ntan,
                                                  log_level=self.logger.LogLevel.MoreDebug)
                            self.nodes[nmove].update_coordinate_basis(constraints=self.ictan[ntan])
                            constraint = self.nodes[nmove].constraints[:, 0]
                            dq0 = rpmove[nmove]*constraint
                            self.nodes[nmove].update_xyz(dq0, verbose=True)

            self.logger.log_print(
                " spacings (end ic_reparam, steps: {i}/{ic_reparam_steps}): {spacings_str} disprms: {disprms:1.3}",
                i=i+1,
                spacings=self.dqmaga,
                ic_reparam_steps=ic_reparam_steps,
                disprms=disprms,
                preformatter=lambda *, spacings, **kw:dict(kw, spacings_str=(
                    " ".join("{:1.2}".format(q) for q in spacings)
                ))
            )

        # TODO old GSM does this here
        # Failed = check_array(self.num_nodes,self.dqmaga)
        # If failed, do exit 1

    def modify_TS_Hess(self):
        ''' Modifies Hessian using RP direction'''
        self.logger.log_print("modifying %i Hessian with RP" % self.TSnode)

        TSnode = self.TSnode
        # a variable to determine how many time since last modify
        self.hess_counter = 0
        self.TS_E_0 = self.energies[TSnode]
        self.logger.log_print(f"{TSnode=}")

        E0 = self.energies[TSnode]/units.KCAL_MOL_PER_AU
        Em1 = self.energies[TSnode-1]/units.KCAL_MOL_PER_AU
        if self.TSnode+1 < self.num_nodes:
            Ep1 = self.energies[TSnode+1]/units.KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        # Update TS node coord basis
        Vecs = self.nodes[TSnode].update_coordinate_basis(constraints=None)

        # get constrained coord basis
        self.newic.xyz = self.nodes[TSnode].xyz.copy()
        const_vec = self.newic.update_coordinate_basis(constraints=self.ictan[TSnode])
        q0 = self.newic.coordinates[0]
        constraint = self.newic.constraints[:, 0]

        # this should just give back ictan[TSnode]?
        tan0 = block_matrix.dot(const_vec, constraint)

        # get qm1 (don't update basis)
        self.newic.xyz = self.nodes[TSnode-1].xyz.copy()
        qm1 = self.newic.coordinates[0]

        if TSnode+1 < self.num_nodes:
            # get qp1 (don't update basis)
            self.newic.xyz = self.nodes[TSnode+1].xyz.copy()
            qp1 = self.newic.coordinates[0]
        else:
            qp1 = qm1

        self.logger.log_print(" TS Hess init'd w/ existing Hintp")

        # Go to non-constrained basis
        self.newic.xyz = self.nodes[TSnode].xyz.copy()
        self.newic.coord_basis = Vecs
        self.newic.Primitive_Hessian = self.nodes[TSnode].Primitive_Hessian.copy()
        self.newic.form_Hessian_in_basis()

        tan = block_matrix.dot(block_matrix.transpose(Vecs), tan0)   # (nicd,1
        Ht = np.dot(self.newic.Hessian, tan)                         # (nicd,nicd)(nicd,1) = nicd,1
        tHt = np.dot(tan.T, Ht)

        a = abs(q0-qm1)
        b = abs(qp1-q0)
        if a < 1e-12:
            self.logger.log_print(" ill-defined TS displacement vector {q0} vs {qm1} for nodes {xyz1} and {xyz2}",
                                  q0=q0,
                                  qm1=qm1,
                                  xyz1=self.nodes[TSnode].xyz,
                                  xyz2=self.nodes[TSnode-1].xyz,
                                  )
            raise ValueError(f"ill-defined TS displacement vector {q0}-{qm1} for coords")
        if b < 1e-12:
            self.logger.log_print(" ill-defined TS displacement vector {q0} vs {qp1} for nodes {xyz1} and {xyz2}",
                                  q0=q0,
                                  qp1=qp1,
                                  xyz1=self.nodes[TSnode].xyz,
                                  xyz2=self.nodes[TSnode+1].xyz,
                                  )
            raise ValueError(f"ill-defined TS displacement vector {qp1}-{q0}")
        c = 2*(Em1/a/(a+b) - E0/a/b + Ep1/b/(a+b))
        self.logger.log_print(" tHt {tHt:1.3f} a: {a:1.1f} b: {b:1.1f} c: {c:1.3f}",
                              tHt=tHt[0, 0],
                              a=a[0],
                              b=b[0],
                              c=c[0]
                              )

        ttt = np.outer(tan, tan)

        # Hint before
        # with np.printoptions(threshold=np.inf):
        #    print self.newic.Hessian
        # eig,tmph = np.linalg.eigh(self.newic.Hessian)
        # print "initial eigenvalues"
        # print eig

        # Finalize Hessian
        self.newic.Hessian += (c-tHt)*ttt
        self.nodes[TSnode].Hessian = self.newic.Hessian.copy()

        # Hint after
        # with np.printoptions(threshold=np.inf):
        #    print self.nodes[TSnode].Hessian
        # print "shape of Hessian is %s" % (np.shape(self.nodes[TSnode].Hessian),)

        self.nodes[TSnode].newHess = 5

        if False:
            self.logger.log_print("newHess of node %i %i" % (TSnode, self.nodes[TSnode].newHess))
            eigen, tmph = np.linalg.eigh(self.nodes[TSnode].Hessian)  # nicd,nicd
            self.logger.log_print("eigenvalues of new Hess")
            self.logger.log_print(eigen)

        # reset pgradrms ?

    def mult_steps(self, n, opt_steps):
        exsteps = 1
        tsnode = int(self.TSnode)

        if (self.find or self.climb) and self.energies[n] > self.energies[self.TSnode]*0.9 and n != tsnode:  #
            exsteps = 2
            self.logger.log_print(" multiplying steps for node %i by %i" % (n, exsteps))
        elif self.find and n == tsnode and self.energies[tsnode] > self.energies[tsnode-1]*1.1 and self.energies[tsnode] > self.energies[tsnode+1]*1.1:  # Can also try self.climb but i hate climbing image
            exsteps = 2
            self.logger.log_print(" multiplying steps for node %i by %i" % (n, exsteps))
        # elif not self.find and not self.climb and n==tsnode  and self.energies[tsnode]>self.energies[tsnode-1]*1.5 and self.energies[tsnode]>self.energies[tsnode+1]*1.5 and self.climber:
        #    exsteps=2
        #    print(" multiplying steps for node %i by %i" % (n,exsteps))

        # elif not (self.find and self.climb) and self.energies[tsnode] > 1.75*self.energies[tsnode-1] and self.energies[tsnode] > 1.75*self.energies[tsnode+1] and self.done_growing and n==tsnode:  #or self.climb
        #    exsteps=2
        #    print(" multiplying steps for node %i by %i" % (n,exsteps))
        return exsteps*opt_steps

    def set_opt_type(self, n):
        # TODO: add an input variable that allows better control of this process
        #       don't infer intent from some random 'PES' attribute...
        opt_type = 'ICTAN'
        if n == self.TSnode:
            if self.find:
                opt_type = 'TS'
            elif self.climb:
                opt_type = 'CLIMB'
        # if not quiet:
        #     print((" setting node %i opt_type to %s" % (n, opt_type)))
        # if isinstance(self.optimizer[n],beales_cg) and opt_type!="BEALES_CG":
        #    raise RuntimeError("This shouldn't happen")

        return opt_type

    #TODO Remove me does not deserve to be a function
    def set_finder(self, rtype):
        rtype = TSOptimizationStrategy(rtype)
        # assert rtype in [0, 1, 2], "rtype not defined"
        msg = [
            "*********************************************************************"
        ]
        # print('')
        if rtype == TSOptimizationStrategy.Exact:
            msg.append("****************** set climber and finder to True *******************")
            self.climber = True
            self.finder = True
        elif rtype == 1:
            msg.append("***************** setting climber to True*************************")
            self.climber = True
        else:
            msg.append("******** Turning off climbing image and exact TS search **********")
        msg.append("*********************************************************************")
        self.logger.log_print(msg)

    def com_rotate_move(self, iR, iP, iN):
        self.logger.log_print(" aligning com and to Eckart Condition")

        mfrac = 0.5
        if self.num_nodes - self.current_nnodes+1 != 1:
            mfrac = 1./(self.num_nodes - self.current_nnodes+1)

        # if self.__class__.__name__ != "DE_GSM":
        #    # no "product" structure exists, use initial structure
        #    iP = 0

        xyz0 = self.nodes[iR].xyz.copy()
        xyz1 = self.nodes[iN].xyz.copy()
        com0 = self.nodes[iR].center_of_mass
        com1 = self.nodes[iN].center_of_mass
        masses = self.nodes[iR].mass_amu

        # From the old GSM code doesn't work
        # com1 = mfrac*(com2-com0)
        # print("com1")
        # print(com1)
        # # align centers of mass
        # xyz1 += com1
        # Eckart_align(xyz1,xyz2,masses,mfrac)

        # rotate to be in maximal coincidence with 0
        # assumes iP i.e. 2 is also in maximal coincidence
        U = rotate.get_rot(xyz0, xyz1)
        xyz1 = np.dot(xyz1, U)

        # # align
        # if self.nodes[iP] != None:
        #    xyz2 = self.nodes[iP].xyz.copy()
        #    com2 = self.nodes[iP].center_of_mass

        #    if abs(iN-iR) > abs(iN-iP):
        #        avg_com = mfrac*com2 + (1.-mfrac)*com0
        #    else:
        #        avg_com = mfrac*com0 + (1.-mfrac)*com2
        #    dist = avg_com - com1  #final minus initial
        # else:
        #    dist = com0 - com1  #final minus initial

        # print("aligning to com")
        # print(dist)
        # xyz1 += dist

        return xyz1

    def find_peaks(self, rtype='opting'):
        '''
        This doesnt actually calculate peaks, it calculates some other thing
        '''
        # rtype 1: growing
        # rtype 2: opting
        # rtype 3: intermediate check
        if rtype not in ['growing', 'opting', 'intermediate']:
            raise RuntimeError

        # if rtype==1:
        if rtype == "growing":
            nnodes = self.nR
        elif rtype == "opting" or rtype == "intermediate":
            nnodes = self.num_nodes
        else:
            raise ValueError("find peaks bad input")
        # if rtype==1 or rtype==2:
        #    print "Energy"
        alluptol = 0.1
        alluptol2 = 0.5
        allup = True
        diss = False
        energies = self.energies
        for n in range(1, len(energies[:nnodes])):
            if energies[n]+alluptol < energies[n-1]:
                allup = False
                break

        if energies[nnodes-1] > 15.0:
            if nnodes-3 > 0:
                if ((energies[nnodes-1]-energies[nnodes-2]) < alluptol2 and
                    (energies[nnodes-2]-energies[nnodes-3]) < alluptol2 and
                        (energies[nnodes-3]-energies[nnodes-4]) < alluptol2):
                    self.logger.log_print(" possible dissociative profile")
                    diss = True

        self.logger.log_print(" nnodes {nnodes}", nnodes=nnodes)
        self.logger.log_print(" all uphill? {allup}", allup=allup)
        self.logger.log_print(" dissociative? {diss}", diss=diss)
        npeaks1 = 0
        npeaks2 = 0
        minnodes = []
        maxnodes = []
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[nnodes-1] < energies[nnodes-2]:
            minnodes.append(nnodes-1)
        for n in range(self.n0, nnodes-1):
            if energies[n+1] > energies[n]:
                if energies[n] < energies[n-1]:
                    minnodes.append(n)
            if energies[n+1] < energies[n]:
                if energies[n] > energies[n-1]:
                    maxnodes.append(n)

        self.logger.log_print(" min nodes {minnodes}", minnodes=minnodes)
        self.logger.log_print(" max nodes {maxnodes}", maxnodes=maxnodes)
        npeaks1 = len(maxnodes)
        # print "number of peaks is ",npeaks1
        ediff = 0.5
        PEAK4_EDIFF = 2.0
        if rtype == "growing":
            ediff = 1.
        if rtype == "intermediate":
            ediff = PEAK4_EDIFF

        if rtype == "growing":
            nmax = np.argmax(energies[:self.nR])
            emax = float(max(energies[:self.nR]))
        else:
            emax = float(max(energies))
            nmax = np.argmax(energies)

        self.logger.log_print(" emax and nmax in find peaks {emax:3.4f},{nmax}", emax=emax, nmax=nmax)

        #check if any node after peak is less than 2 kcal below
        for n in maxnodes:
            diffs = (energies[n]-e > ediff for e in energies[n:nnodes])
            if any(diffs):
                found = n
                npeaks2 += 1
        npeaks = npeaks2
        self.logger.log_print(" found {npeaks} significant peak(s) TOL {ediff:3.2f}", npeaks=npeaks, ediff=ediff)

        # handle dissociative case
        if rtype == "intermediate" and npeaks == 1:
            nextmin = 0
            for n in range(found, nnodes-1):
                if n in minnodes:
                    nextmin = n
                    break
            if nextmin > 0:
                npeaks = 2

        # if rtype==3:
        #    return nmax
        if allup is True and npeaks == 0:
            return -1
        if diss is True and npeaks == 0:
            return -2

        return npeaks

    def is_converged(self, totalgrad, fp, rtype, ts_cgradq):
        '''
        Check if optimization is converged
        '''

        # Important the factor 5 here corresponds to the same convergence criteria in the TS optimizer
        TS_conv = self.tolerances['CONV_TOL']*5
        # => Check if intermediate exists
        # ALEX REMOVED CLIMB REQUIREMENT
        if self.has_intermediate(self.noise):
            self.logger.log_print("New pot min: {}".format(self.get_intermediate(self.noise)))
            self.logger.log_print("Old pot min: {}".format(self.pot_min))
            if self.get_intermediate(self.noise) == self.pot_min:
                self.endearly_counter += 1
            else:
                self.pot_min = self.get_intermediate(self.noise)
                self.endearly_counter = 1
            if self.endearly_counter >= 3:
                self.end_early = True
                self.tscontinue = False
                self.logger.log_printcool(" THERE IS AN INTERMEDIATE, OPTIMIZE THE INTERMEDIATE AND TRY AGAIN")
                return True

        elif not self.has_intermediate(self.noise):
            self.endearly_counter = 0
            self.pot_min = self.get_intermediate(self.noise)

        # print(" Number of imaginary frequencies %i" % self.optimizer[self.TSnode].nneg)

        # or (totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*TS_conv and self.dE_iter<0.02 and self.optimizer[self.TSnode].nneg <2)  #TODO extra crit here

        # 4/8/2022 Bug with this because it isn't recalculated and abs(ts_cgradq) < TS_conv
        # 5/10/2022 self.optimizer[self.TSnode].nneg < 3
        if (self.finder and self.find):
            return (self.nodes[self.TSnode].gradrms < self.tolerances['CONV_TOL']  and self.dE_iter < self.optimizer[self.TSnode].conv_Ediff*3)
        elif self.climber and self.climb:
            return (self.nodes[self.TSnode].gradrms < self.tolerances['CONV_TOL'] and abs(ts_cgradq) < TS_conv and self.dE_iter < self.optimizer[self.TSnode].conv_Ediff*3)
        elif not self.climber and not self.finder:
            self.logger.log_print(" CONV_TOL=%.4f" % self.tolerances['CONV_TOL'])
            return all([self.optimizer[n].converged for n in range(1, self.num_nodes-1)])

        return False

    def get_intermediate(self, noise):
        '''
        Check string for intermediates
        noise is a leeway factor for determining intermediate
        '''

        energies = self.energies
        potential_min = []
        for i in range(1, (len(energies) - 1)):
            rnoise = 0
            pnoise = 0
            a = 1
            b = 1
            while (energies[i-a] >= energies[i]):
                if (energies[i-a] - energies[i]) > rnoise:
                    rnoise = energies[i-a] - energies[i]
                if rnoise > noise:
                    break
                if (i-a) == 0:
                    break
                a += 1

            while (energies[i+b] >= energies[i]):
                if (energies[i+b] - energies[i]) > pnoise:
                    pnoise = energies[i+b] - energies[i]
                if pnoise > noise:
                    break
                if (i+b) == len(energies) - 1:
                    break
                b += 1
            if ((rnoise > noise) and (pnoise > noise)):
                self.logger.log_print('Potential minimum at image %s' % i)
                potential_min.append(i)

        return potential_min

    def has_intermediate(self, noise):
        pot_min = self.get_intermediate(noise)
        return len(pot_min) > 0

    def setup_from_geometries(self, input_geoms, reparametrize=True, restart_energies=True, start_climb_immediately=False):
        '''
        Restart
        input_geoms list of geometries
        reparameterize (boolean) : reparameterize the initial string to make the nodes equidistant
        restart_energies (boolean) : generate the initial energies
        start_climb_immediately (boolean) : set climb to True or False
        '''

        self.logger.log_printcool("Restarting GSM from geometries")
        self.growth_direction = NodeAdditionStrategy.Normal
        nstructs = len(input_geoms)

        if nstructs != self.num_nodes:
            self.logger.log_print('need to interpolate')
            # if self.interp_method=="DLC": TODO
            raise NotImplementedError
        else:
            geoms = input_geoms

        self.gradrms = [0.]*nstructs
        self.dE = [1000.]*nstructs

        self.isRestarted = True
        self.done_growing = True

        # set coordinates from geoms
        self.nodes[0].xyz = xyz_to_np(geoms[0])
        self.nodes[nstructs-1].xyz = xyz_to_np(geoms[-1])
        for struct in range(1, nstructs-1):
            self.nodes[struct] = Molecule.copy_from_options(self.nodes[struct-1],
                                                            xyz_to_np(geoms[struct]),
                                                            new_node_id=struct,
                                                            copy_wavefunction=False)
            self.nodes[struct].newHess = 5
            # Turning this off
            # self.nodes[struct].gradrms = np.sqrt(np.dot(self.nodes[struct].gradient,self.nodes
            # self.nodes[struct].gradrms=grmss[struct]
            # self.nodes[struct].PES.dE = dE[struct]
        self.num_nodes = self.nR = nstructs

        if start_climb_immediately:
            # should check that this is a climber...
            self.climb = True
        #ALEX CHANGE - rearranged reparameterize and restart_energies 'if' blocks
        if restart_energies:
            # self.interpolate_orbitals()
            self.logger.log_print([
                " V_profile: {vprof_str}"
                ],
                vprof=self.energies,
                preformatter=lambda *,vprof,**kw: dict(kw, vprof_str=" ".join(f'{e:7.3f}' for e in vprof))
            )
        if reparametrize:
            self.logger.log_printcool("Reparametrizing")
            self.reparameterize(ic_reparam_steps=8)
            self.xyz_writer('grown_string_{:03}.xyz'.format(self.ID), self.geometries, self.energies, self.gradrmss, self.dEs)


        self.ictan, self.dqmaga = self.get_tangents(self.nodes)
        self.refresh_coordinates()
        self.logger.log_print(" setting all interior nodes to active")
        for n in range(1, self.num_nodes-1):
            self.active[n] = True
            self.optimizer[n].conv_grms = self.tolerances['CONV_TOL']*2.5
            self.optimizer[n].max_step = 0.05

        return

    def add_node_before_TS(self):
        '''
        '''

        # New node is TS node
        new_node = self.add_node(
            self.nodes[self.TSnode-1],
            self.nodes[self.TSnode],
            stepsize=0.5,
            node_id=self.TSnode,
            node_idR=self.TSnode - 1,
            node_idP=self.TSnode
        )
        new_node_list = [None]*(self.num_nodes+1)
        new_optimizers = [None]*(self.num_nodes+1)

        for n in range(0, self.TSnode):
            new_node_list[n] = self.nodes[n]
            new_optimizers[n] = self.optimizer[n]

        new_node_list[self.TSnode] = new_node
        new_optimizers[self.TSnode] = self.optimizer[0].copy()

        for n in range(self.TSnode+1, self.num_nodes+1):
            new_node_list[n] = self.nodes[n-1].copy()
            new_optimizers[n] = self.optimizer[n-1]

        self.nodes:list[Molecule] = new_node_list
        self.optimizer = new_optimizers
        self.logger.log_print(' New number of nodes %d' % self.num_nodes)
        self.active = [True] * self.num_nodes
        self.active[0] = False
        self.active[self.num_nodes-1] = False
        self.logger.log_print([
            "0",
            "{n0}",
            "1",
            "{n1}",
            "-1",
            "{nf}"
        ],
            n0=self.nodes[0].xyz,
            n1=self.nodes[1].xyz,
            nf=self.nodes[-1].xyz,
        )

    def add_node_after_TS(self):
        '''
        '''
        new_node = GSM.add_node(
            self.nodes[self.TSnode],
            self.nodes[self.TSnode+1],
            stepsize=0.5,
            node_id=self.TSnode+1,
            node_idR=self.TSnode,
            node_idP=self.TSnode+1,
        )
        new_node_list = [None]*(self.num_nodes+1)
        new_optimizers = [None]*(self.num_nodes+1)
        for n in range(0, self.TSnode+1):
            new_node_list[n] = self.nodes[n]
            new_optimizers[n] = self.optimizer[n]
        new_node_list[self.TSnode+1] = new_node
        new_optimizers[self.TSnode+1] = self.optimizer[0].copy()

        for n in range(self.TSnode+2, self.num_nodes+1):
            new_node_list[n] = self.nodes[n-1].copy()
            new_optimizers[n] = self.optimizer[n-1]
        self.nodes = new_node_list
        self.optimizer = new_optimizers
        self.logger.log_print(' New number of nodes %d' % self.num_nodes)
        self.active = [True] * self.num_nodes
        self.active[0] = False
        self.active[self.num_nodes-1] = False

    def set_node_convergence(self):
        ''' set convergence for nodes
        '''

        factor = 5. if (self.climber or self.finder) else 1.
        TSnode = self.TSnode
        for n in range(1, self.num_nodes-1):
            if self.nodes[n] is not None:
                self.optimizer[n].conv_grms = self.tolerances['CONV_TOL']*factor
                self.optimizer[n].conv_gmax = self.tolerances['CONV_gmax']*factor
                self.optimizer[n].conv_Ediff = self.tolerances['CONV_Ediff']*factor
                if self.optimizer[n].converged and n != TSnode:
                    self.optimizer[n].check_only_grad_converged=True
                if (self.climb or self.find) and self.energies[n]>self.energies[TSnode]*0.75 and n!=TSnode:
                    self.optimizer[n].conv_grms = self.tolerances['CONV_TOL']     
                    self.optimizer[n].conv_gmax = self.tolerances['CONV_gmax']
                    self.optimizer[n].conv_Ediff = self.tolerances['CONV_Ediff']
                    self.optimizer[n].check_only_grad_converged = False
                if n == self.TSnode and (self.climb or self.find):
                    self.optimizer[n].conv_grms = self.tolerances['CONV_TOL']
                    self.optimizer[n].conv_gmax = self.tolerances['CONV_gmax']
                    self.optimizer[n].conv_Ediff = self.tolerances['CONV_Ediff']
                    self.optimizer[n].check_only_grad_converged = False

    def slow_down_climb(self):
        if self.climb and not self.find:
            self.logger.log_print(" slowing down climb optimization")
            self.optimizer[self.TSnode].max_step /= self.newclimbscale
            self.optimizer[self.TSnode].SCALEQN = 2.
            if self.optimizer[self.TSnode].SCALE_CLIMB < 5.:
                self.optimizer[self.TSnode].SCALE_CLIMB += 1.
            self.optimizer[self.pTSnode].SCALEQN = 1.
            self.ts_exsteps = 1
            if self.newclimbscale < 5.0:
                self.newclimbscale += 1.
        elif self.find:
            self.find = False
            self.climb = True
            self.nclimb = 1
            self.logger.log_print(" Find bad, going back to climb")

    def interpolate_orbitals(self):
        '''
        Interpolate orbitals
        '''
        self.logger.log_print("Interpolating orbitals")
        nnodes = len(self.nodes)
        #nn = nnodes//2 + 1
        nn = - (nnodes // -2)
        couples = [(i, nnodes-i-1) for i in range(nn)]
        first = True
        options = {}
        for i, j in couples:
            if first:
                # Calculate the energy of the i, j
                self.logger.log_print("Calculating initial energy for node: {}".format(i))
                self.nodes[i].energy
                self.logger.log_print("Calculating initial energy for node: {}".format(j))
                self.nodes[j].energy
                first = False
            elif j - i <= 1:
                #even nnodes case
                if i == j - 1:  
                    self.nodes[i].PES.lot = type(self.nodes[i-1].PES.lot).copy(
                        self.nodes[i-1].PES.lot, copy_wavefunction=True)
                    self.nodes[i].PES.lot.node_id = i
                    self.nodes[i].energy
                    i += 1
                if i == j:
                    self.logger.log_print("Checking if energies match for wavefunction guesses from either direction for node: {}".format(i))

                    options={'node_id':self.nodes[i].node_id}
                    self.nodes[i].PES.lot = type(self.nodes[i-1].PES.lot).copy(
                        self.nodes[i-1].PES.lot, options, copy_wavefunction=True)
                    self.logger.log_print("Getting forward energy")
                    self.nodes[i].PES.lot.node_id = i
                    energy_forward = self.nodes[i].energy
                    self.logger.log_print("Getting backward energy")
                    options={'node_id':self.nodes[i+1].node_id}
                    self.nodes[i].PES.lot = type(self.nodes[i+1].PES.lot).copy(
                        self.nodes[i+1].PES.lot, options, copy_wavefunction=True)
                    self.nodes[i].PES.lot.node_id = i
                    self.nodes[i].PES.lot.hasRanForCurrentCoords = False
                    energy_backward = self.nodes[i].energy
                    self.logger.log_print("Forward direction energy: {}".format(energy_forward))
                    self.logger.log_print("Backward direction energy: {}".format(energy_backward))
                    if abs(energy_forward - energy_backward) < 0.1:
                        self.logger.log_print("Energies match")
                    else:
                        self.logger.log_print("Energies do not match")
                        if energy_backward < energy_forward:
                            for k in range(i):
                                options={'node_id':self.nodes[i+k-1].node_id}
                                self.nodes[i-k-1].PES.lot = type(self.nodes[i-k].PES.lot).copy(
                                    self.nodes[i-k].PES.lot, options, copy_wavefunction=True)
                                self.nodes[i-k-1].PES.lot.node_id = i-k-1
                                self.nodes[i-k -1].PES.lot.hasRanForCurrentCoords = False
                                self.logger.log_print("node_id {}".format(self.nodes[i+k].node_id))
                                self.logger.log_print("Calculating new initial energy for node: {}".format(i-k-1))
                                self.logger.log_print("New energy: {}".format(self.nodes[i-k-1].energy))
                        else:
                            for k in range(i+1):
                                #lower energy is in forward direction, so do node i using i-1's wavefunction
                                options={'node_id':self.nodes[i+k].node_id}
                                self.nodes[i+k].PES.lot = type(self.nodes[i+k-1].PES.lot).copy(
                                    self.nodes[i+k-1].PES.lot, options, copy_wavefunction=True)
                                self.nodes[i+k].PES.lot.node_id = i+k
                                self.nodes[i+k].PES.lot.hasRanForCurrentCoords = False
                                self.logger.log_print("node_id {}".format(self.nodes[i+k].node_id))
                                self.logger.log_print("Calculating new initial energy for node: {}".format(i+k))
                                self.logger.log_print("New energy: {}".format(self.nodes[i+k].energy))

                     
                    
            else:
                # Copy the orbital of i-1 to i
                self.nodes[i].PES.lot = type(self.nodes[i-1].PES.lot).copy(
                    self.nodes[i-1].PES.lot, options, copy_wavefunction=True)
                self.nodes[i].PES.lot.node_id = i
                self.logger.log_print("Calculating initial energy for node: {}".format(i))
                self.nodes[i].energy
                # Copy the orbital of j+1 to j
                self.nodes[j].PES.lot = type(self.nodes[j+1].PES.lot).copy(
                    self.nodes[j+1].PES.lot, options, copy_wavefunction=True)
                self.nodes[j].PES.lot.node_id = j
                self.logger.log_print("Calculating initial energy for node: {}".format(j))
                self.nodes[j].energy
        return
