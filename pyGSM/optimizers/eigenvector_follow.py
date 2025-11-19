from __future__ import print_function

# standard library imports
import sys
import os
from io import StringIO

# third party
import numpy as np

# local application imports
from ._linesearch import NoLineSearch
from .hessian_update_optimizers import hessian_update_optimizer
from ..utilities import units, block_matrix, manage_xyz
from .. import coordinate_systems as coord_ops


class eigenvector_follow(hessian_update_optimizer):

    def eigenvector_step(self, molecule, g):

        SCALE = self.SCALEQN
        if molecule.newHess > 0:
            SCALE = SCALE*molecule.newHess
        if self.SCALEQN > 10.0: # ? just SCALEQN, what about SCALE?
            SCALE = 10.0

        self.logger.log_print("new_hess {scaling}", scaling=molecule.newHess, log_level=self.logger.LogLevel.Debug)
        self.logger.log_print("constraints {constraints}", constraints=molecule.constraints.T,
                              log_level=self.logger.LogLevel.Debug)

        P = np.eye(len(molecule.constraints), dtype=float)
        for c in molecule.constraints.T:
            P -= np.outer(c[:, np.newaxis], c[:, np.newaxis].T)
        self.Hessian = np.dot(np.dot(P, molecule.Hessian), P)

        e, v_temp = np.linalg.eigh(self.Hessian)
        gqe = np.dot(v_temp.T, g)
        lambda1 = self.set_lambda1('NOT-TS', e)

        self.logger.log_print([
            "eigenvalues {e}",
            "eigenvectors {v}",
            "g {g}"
            "gqe {gqe}",
            ],
            e=e,
            v=v_temp,
            g=g.T,
            gqe=gqe.T,
            log_level=self.logger.LogLevel.Debug)

        dqe0 = -gqe.flatten()/(e+lambda1)/SCALE
        dqe0 = [np.sign(i)*self.MAXAD if abs(i) > self.MAXAD else i for i in dqe0]
        dqe0 = np.asarray(dqe0)

        # => Convert step back to DLC basis <= #
        dq = np.dot(v_temp, dqe0)
        dq = [np.sign(i)*self.MAXAD if abs(i) > self.MAXAD else i for i in dq]
        dq = np.asarray(dq)

        dq = np.reshape(dq, (-1, 1))
        for c in molecule.constraints.T:
            dq -= np.dot(c[:, np.newaxis].T, dq)*c[:, np.newaxis]

        # print("check overlap")
        # print(np.dot(dq.T,molecule.constraints))
        self.logger.log_print("dq {dq}", dq=dq.T, log_level=self.logger.LogLevel.Debug)
        return np.reshape(dq, (-1, 1))

    # need to modify this only for the DLC region
    def TS_eigenvector_step(self, molecule, g, ictan):
        '''
        Takes an eigenvector step using the Bofill updated Hessian ~1 negative eigenvalue in the
        direction of the reaction path.

        '''
        SCALE = self.SCALEQN
        if molecule.newHess > 0:
            SCALE = self.SCALEQN*molecule.newHess
        if self.SCALEQN > 10.0:
            SCALE = 10.0

        # constraint vector
        norm = np.linalg.norm(ictan)
        C = ictan/norm
        Vecs = molecule.coord_basis
        Cn = block_matrix.dot(block_matrix.dot(Vecs, block_matrix.transpose(Vecs)), C)
        norm = np.linalg.norm(Cn)
        Cn = Cn/norm

        # => get eigensolution of Hessian <=
        self.Hessian = molecule.Hessian.copy()
        eigen, tmph = np.linalg.eigh(self.Hessian)  # nicd,nicd
        tmph = tmph.T

        # TODO nneg should be self and checked
        self.nneg = sum(1 for e in eigen if e < -0.01)

        # => Overlap metric <= #
        overlap = np.dot(block_matrix.dot(tmph, block_matrix.transpose(Vecs)), Cn)

        self.logger.log_print(" overlap {overlap}", overlap=overlap[:4].T)
        self.logger.log_print(" nneg {nneg}", nneg=self.nneg)
        # Max overlap metrics
        path_overlap, maxoln = self.maxol_w_Hess(overlap[0:4])
        self.logger.log_print(" t/ol {maxoln}: {path_overlap:3.2f}", maxoln=maxoln, path_overlap=path_overlap)

        # => set lamda1 scale factor <=#
        lambda1 = self.set_lambda1('TS', eigen, maxoln)

        self.maxol_good = True
        if path_overlap < self.HESS_TANG_TOL_TS:
            self.maxol_good = False

        if self.maxol_good:
            # => grad in eigenvector basis <= #
            gqe = np.dot(tmph, g)
            path_overlap_e_g = gqe[maxoln]
            self.logger.log_print(' gtse: {:1.4f} '.format(path_overlap_e_g[0]))
            # save gtse in memory ...
            self.gtse = abs(path_overlap_e_g[0])
            # => calculate eigenvector step <=#
            dqe0 = np.zeros((molecule.num_coordinates, 1))
            for i in range(molecule.num_coordinates):
                if i != maxoln:
                    dqe0[i] = -gqe[i] / (abs(eigen[i])+lambda1) / SCALE
            lambda0 = 0.0025
            dqe0[maxoln] = gqe[maxoln] / (abs(eigen[maxoln]) + lambda0)/SCALE

            # => Convert step back to DLC basis <= #
            dq = np.dot(tmph.T, dqe0)  # should it be transposed?
            dq = [np.sign(i)*self.MAXAD if abs(i) > self.MAXAD else i for i in dq]
            dq = np.asarray(dq)

            dq = np.reshape(dq, (-1, 1))
        else:
            # => if overlap is small use Cn as Constraint <= #
            molecule.update_coordinate_basis(ictan)
            g = molecule.gradient
            self.logger.log_print("constraints: {c}", c=molecule.constraints.T)
            molecule.form_Hessian_in_basis()
            dq = self.eigenvector_step(molecule, g)

        return dq

    def optimize(
            self,
            molecule,
            refE=0.,
            opt_type='UNCONSTRAINED',
            opt_steps=3,
            ictan=None,
            xyzframerate=4,
            verbose=False,
            path=None
    ):

        # stash/initialize some useful attributes
        self.check_inputs(molecule, opt_type, ictan)
        nconstraints = self.get_nconstraints(opt_type)
        self.buf = StringIO()

        # print " refE %5.4f" % refE
        self.logger.log_print(" initial E {dE:5.4f}", dE=(molecule.energy - refE), log_level=self.logger.LogLevel.MoreDebug)
        self.logger.log_print(" CONV_TOL {ctol:1.5f}", ctol=self.conv_grms, log_level=self.logger.LogLevel.MoreDebug)
        geoms = []
        energies = []
        geoms.append(molecule.xyz)
        energies.append(molecule.energy-refE)
        self.converged = False

        # form initial coord basis
        if opt_type != 'TS':
            constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
            molecule.update_coordinate_basis(constraints=constraints)
            molecule.form_Hessian_in_basis()

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()
        # project out the constraint
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T, c[:, np.newaxis])*c[:, np.newaxis]
        gmax = float(np.max(np.absolute(gc)))

        if self.check_only_grad_converged:
            if molecule.gradrms < self.conv_grms and gmax < self.conv_gmax:
                self.converged = True
                return geoms, energies
            else:
                self.check_only_grad_converged = False

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        if opt_type == 'TS':
            self.Linesearch = NoLineSearch
        if opt_type == 'SEAM' or opt_type == 'MECI' or opt_type == "TS-SEAM":
            self.opt_cross = True

        # TODO are these used? -- n is used for gradrms,linesearch
        if coord_ops.is_cartesian(molecule.coord_obj):
            n = molecule.num_coordinates
        else:
            n_actual = molecule.num_coordinates
            n = n_actual - nconstraints
            self.x_prim = np.zeros((molecule.num_primitives, 1), dtype=float)
            self.g_prim = np.zeros((molecule.num_primitives, 1), dtype=float)

        molecule.gradrms = np.sqrt(np.dot(gc.T, gc)/n)[0, 0]
        # dE = molecule.difference_energy
        update_hess = False

        # ====>  Do opt steps <======= #
        for ostep in range(opt_steps):
            self.logger.log_print(" On opt step {sn}", sn=ostep+1, log_level=self.logger.LogLevel.MoreDebug)

            # update Hess
            if update_hess:
                if opt_type != 'TS':
                    self.update_Hessian(molecule, 'BFGS')
                else:
                    self.update_Hessian(molecule, 'BOFILL')
            update_hess = True

            # => Form eigenvector step <= #
            if coord_ops.is_cartesian(molecule.coord_obj):
                raise NotImplementedError
            else:
                if opt_type != 'TS':
                    dq = self.eigenvector_step(molecule, gc)
                else:
                    dq = self.TS_eigenvector_step(molecule, g, ictan)
                    if not self.maxol_good:
                        self.logger.log_print(" Switching to climb! Maxol not good!", log_level=self.logger.LogLevel.MoreDebug)
                        nconstraints = 1
                        opt_type = 'CLIMB'

            actual_step = np.linalg.norm(dq)
            # print(" actual_step= %1.2f"% actual_step)
            dq = dq/actual_step  # normalize
            if actual_step > self.max_step:
                step = self.max_step
                # print(" reducing step, new step = %1.2f" %step)
            else:
                step = actual_step

            # store values
            xp = x.copy()
            gp = g.copy()
            xyzp = xyz.copy()
            fxp = fx
            pgradrms = molecule.gradrms
            if not coord_ops.is_cartesian(molecule.coord_obj):
                # xp_prim = self.x_prim.copy()
                gp_prim = self.g_prim.copy()

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)

            # print(" ### Starting  line search ###")
            ls = self.Linesearch(nconstraints, x, fx, g, dq, step, xp, constraint_steps, self.linesearch_parameters, molecule, verbose)

            # get values from linesearch
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']

            if ls['status'] == -2:
                self.logger.log_print('[ERROR] the point return to the privious point', log_level=self.logger.LogLevel.MoreDebug)
                x = xp.copy()
                molecule.xyz = xyzp
                g = gp.copy()
                fx = fxp
                ratio = 0.
                molecule.newHess = 5
                # return ls['status']

            if ls['step'] > self.max_step:
                if ls['step'] <= self.max_step:  # absolute max
                    self.logger.log_print(" Increasing DMAX to {step}", step=ls['step'], log_level=self.logger.LogLevel.MoreDebug)
                    self.max_step = ls['step']
                else:
                    self.max_step = self.max_step
            elif ls['step'] < self.max_step:
                if ls['step'] >= self.DMIN:     # absolute min
                    self.logger.log_print(" Decreasing DMAX to {step}", step=ls['step'], log_level=self.logger.LogLevel.MoreDebug)
                    self.max_step = ls['step']
                elif ls['step'] <= self.DMIN:
                    self.max_step = self.DMIN
                    self.logger.log_print(" Decreasing DMAX to {step}", step=self.DMIN, log_level=self.logger.LogLevel.MoreDebug)

            # calculate predicted value from Hessian, gp is previous constrained gradient
            scaled_dq = dq*step
            dEtemp = np.dot(self.Hessian, scaled_dq)
            constraint_energy = np.dot(gp.T, constraint_steps)
            dEpre = np.array(
                            np.dot(np.transpose(scaled_dq), gc)
                            + 0.5*np.dot(np.transpose(dEtemp), scaled_dq)
                            + constraint_energy
            ).flatten()[0] * units.KCAL_MOL_PER_AU
            # print(constraint_steps.T)
            # constraint_energy = *units.KCAL_MOL_PER_AU
            # print("constraint_energy: %1.4f" % constraint_energy)
            # dEpre += constraint_energy # this appeares to be the original intent, not sure why this is like this
            # if abs(dEpre)<0.01:
            #    dEpre = np.sign(dEpre)*0.01

            # project out the constraint
            gc = g.copy()
            for c in molecule.constraints.T:
                gc -= np.dot(gc.T, c[:, np.newaxis])*c[:, np.newaxis]

            # control step size
            dEstep = fx - fxp
            self.logger.log_print(" dEstep={dEstep:5.4f}", dEstep=dEstep, log_level=self.logger.LogLevel.MoreDebug)
            ratio = dEstep/dEpre
            molecule.gradrms = np.sqrt(np.dot(gc.T, gc)/n)[0, 0]
            if ls['status'] != -2:
                self.step_controller(actual_step, ratio, molecule.gradrms, pgradrms, dEpre, opt_type, dEstep)

            # update molecule xyz
            xyz = molecule.update_xyz(x-xp)
            if ostep % xyzframerate == 0:
                geoms.append(molecule.xyz)
                energies.append(molecule.energy-refE)
                # manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path, molecule.node_id), geoms, energies, scale=1.)

            # save variables for update Hessian!
            if not coord_ops.is_cartesian(molecule.coord_obj):
                # only form g_prim for non-constrained
                self.g_prim = block_matrix.dot(molecule.coord_basis, gc)
                self.dx = x-xp
                self.dg = g - gp

                self.dx_prim_actual = molecule.coord_obj.Prims.calcDiff(xyz, xyzp)
                self.dx_prim_actual = np.reshape(self.dx_prim_actual, (-1, 1))
                self.dx_prim = block_matrix.dot(molecule.coord_basis, scaled_dq)
                self.dg_prim = self.g_prim - gp_prim

            else:
                raise NotImplementedError(" ef not implemented for CART")

            log_fmt = (
                " Opt step: {step} E: {E:5.4f} predE: {predE:5.4f} ratio: {ratio:4.3f}"
                " gradrms: {gradrms:1.5f} ss: {step_size:4.3f} DMAX: {DMAX:4.3f}"
            )
            log_str = log_fmt.format(
                step=ostep+1,
                E=fx-refE,
                predE=dEpre,
                ratio=ratio,
                gradrms=molecule.gradrms,
                step_size=step,
                DMAX=self.max_step
            )
            self.logger.log_print(log_str, log_level=self.logger.LogLevel.MoreDebug)
            self.buf.write("\n" + log_str)

            # check for convergence TODO
            fx = molecule.energy
            # dE = molecule.difference_energy
            # if dE < 1000.:
            #     self.logger.log_print(" difference energy is %5.4f" % dE)
            gmax = float(np.max(np.absolute(gc)))
            disp = float(np.linalg.norm((xyz-xyzp).flatten()))
            xnorm = np.sqrt(np.dot(x.T, x))
            # gnorm = np.sqrt(np.dot(g.T, g))
            if xnorm < 1.0:
                xnorm = 1.0

            self.logger.log_print(
                " gmax {gmax:5.4f} disp {disp:5.4f} Ediff {dEstep:5.4f} gradrms {gradrms:5.4f}",
                gmax=gmax,
                disp=disp,
                dEstep=dEstep,
                gradrms=molecule.gradrms,
                log_level=self.logger.LogLevel.MoreDebug
            )

            # TODO turn back on conv_DE
            if self.opt_cross:
                if abs(dE) < self.conv_dE and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                    if opt_type == "TS-SEAM":
                        gts = np.dot(g.T, molecule.constraints[:, 0])
                        self.logger.log_print(" gts %1.4f" % gts)
                        if abs(gts) < self.conv_grms*5:
                            self.converged = True
                    else:
                        self.converged = True
            elif molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                if opt_type == "CLIMB":
                    gts = np.dot(g.T, molecule.constraints[:, 0])
                    if abs(gts) < self.conv_grms*5.:
                        self.converged = True
                elif opt_type == "TS":
                    if self.gtse < self.conv_grms*5.:
                        self.converged = True
                else:
                    self.converged = True

            if self.converged:
                self.logger.log_print(" converged", log_level=self.logger.LogLevel.MoreDebug)
                if ostep % xyzframerate != 0:
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy-refE)
                    # manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path, molecule.node_id), geoms, energies, scale=1.)
                break

            # update DLC  --> this changes q, g, Hint
            if not coord_ops.is_cartesian(molecule.coord_obj):
                if opt_type != 'TS':
                    constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    g = molecule.gradient.copy()
                    # project out the constraint
                    gc = g.copy()
                    for c in molecule.constraints.T:
                        gc -= np.dot(gc.T, c[:, np.newaxis])*c[:, np.newaxis]

        with self.logger.block(tag="opt-summary"):
            self.logger.log_print(self.buf.getvalue().splitlines())
        return geoms, energies