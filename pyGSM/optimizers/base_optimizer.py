import abc
import enum

# third party
import numpy as np

# local application imports
from ._linesearch import backtrack, NoLineSearch
from ..utilities import units, block_matrix, Devutils as dev


def sorted_eigh(mat, asc=False):
    """ 
    Return eigenvalues and eigenvectors of a symmetric matrix
    in descending order and associated eigenvectors.

    This is just a convenience function to get eigenvectors
    in descending or ascending order as desired.
    """
    L, Q = np.linalg.eigh(mat)
    if asc:
        idx = L.argsort()
    else:
        idx = L.argsort()[::-1]
    L = L[idx]
    Q = Q[:, idx]
    return L, Q

def force_positive_definite(H):
    """
    Force all eigenvalues to be positive.
    """
    # Sorted eigenvalues and corresponding eigenvectors of the Hessian
    Hvals, Hvecs = sorted_eigh(H, asc=True)
    Hs = np.zeros_like(H)
    for i in range(H.shape[0]):
        if Hvals[i] > 0:
            Hs += Hvals[i] * np.outer(Hvecs[:, i], Hvecs[:, i])
        else:
            Hs -= Hvals[i] * np.outer(Hvecs[:, i], Hvecs[:, i])
    return Hs

class LinesearchSetting(enum.Enum):
    Disabled = "NoLineSearch"
    Backtracking = "backtrack"

class OptimizationTypes(enum.Enum):
    TransitionState = "TS"
    Unconstrained = "UNCONSTRAINED"
    ICTAN = "ICTAN"
    MECI = "MECI"
    Climbing = "CLIMB"
    Beales = "BEALES_CG"
    Seam = "SEAM"
    TransitionStateSeam = "TS-SEAM"
    NotTS = 'NOT-TS' # uuh...I feel like they were just hacking things together



# TODO Add primitive constraint e.g. a list of internal coordinates to be left basically frozen throughout optimization
class base_optimizer(metaclass=abc.ABCMeta):
    ''' some common functions that the children can use (ef, cg, hybrid ef/cg, etc).
    e.g. walk_up, dgrad_step, what else?
    '''

    # it is terrible design to force every optimizer to need to know about the
    # defaults of every other optimizer, but I don't want to fix the mistakes
    # of other people right now
    def __init__(self,
                 linesearch="NoLineSearch",
                 ftol=1e-4,
                 conv_disp=100.,
                 conv_gmax=100.,
                 conv_Ediff=100.,
                 conv_dE=1.,
                 OPTTHRESH=0.0005,
                 opt_cross=False,
                 opt_climb=False,
                 DMAX=0.4,
                 MAXAD=0.075,
                 SCALEQN=1.0,
                 SCALEW=1.0,
                 SCALE_CLIMB=1.0,
                 logger=None
                 ):

        self.logger = dev.Logger.lookup(logger)
        if not callable(linesearch):
            linesearch = LinesearchSetting(linesearch)
            if linesearch == LinesearchSetting.Backtracking:
                linesearch = backtrack
            elif linesearch == LinesearchSetting.Disabled:
                linesearch = NoLineSearch
            else:
                raise ValueError("")
        self.Linesearch = linesearch

        # additional convergence criterion (default parameters for Q-Chem)
        self.conv_disp = conv_disp  # 12e-4 #max atomic displacement
        self.conv_gmax = conv_gmax  # 3e-4 #max gradient
        self.conv_Ediff = conv_Ediff  # 1e-6 #E diff
        self.conv_dE = conv_dE
        self.conv_grms = OPTTHRESH

        # TS node properties
        self.nneg = 0  # number of negative eigenvalues
        self.DMIN = 0.0001

        # MECI
        self.opt_cross = opt_cross
        self.opt_climb = opt_climb
        
        self.ftol = ftol
        self.max_step = DMAX
        self.MAXAD = MAXAD

        # additional parameters needed by linesearch
        self.linesearch_parameters = {
            'epsilon': 1e-5,
            'ftol': self.ftol,  # 1e-4,
            'wolfe': 0.9,
            'max_linesearch': 3,
            'min_step': self.DMIN,
            'max_step': self.max_step
        }

        self.SCALEQN = SCALEQN
        self.SCALEW = SCALEW
        self.SCALE_CLIMB = SCALE_CLIMB

        # Converged
        self.converged = False
        self.check_only_grad_converged = False

    @abc.abstractmethod
    def optimize(self, molecule, refE=0., opt_type='UNCONSTRAINED', opt_steps=3, ictan=None):
        ...

    def get_state_dict(self):
        return dict(
            linesearch=self.Linesearch,
            ftol=self.ftol,
            conv_disp=self.conv_disp,
            conv_gmax=self.conv_gmax,
            conv_Ediff=self.conv_Ediff,
            conv_dE=self.conv_dE,
            OPTTHRESH=self.conv_grms,
            opt_cross=self.opt_cross,
            opt_climb=self.opt_climb,
            DMAX=self.max_step,
            MAXAD=self.MAXAD,
            SCALEQN=self.SCALEQN,
            SCALEW=self.SCALEW,
            SCALE_CLIMB=self.SCALE_CLIMB,
            logger=self.logger
        )
    def copy(self):
        return type(self)(
            **self.get_state_dict()
        )

    def get_nconstraints(self, opt_type):
        opt_type = OptimizationTypes(opt_type)
        if opt_type in {OptimizationTypes.ICTAN, OptimizationTypes.Climbing}:
            nconstraints = 1
        elif opt_type in {OptimizationTypes.MECI}:
            nconstraints = 2
        elif opt_type in {OptimizationTypes.Seam, OptimizationTypes.TransitionStateSeam}:
            nconstraints = 3
        else:
            nconstraints = 0
        return nconstraints

    def check_inputs(self, molecule, opt_type, ictan):
        opt_type = OptimizationTypes(opt_type)
        if opt_type in {
            OptimizationTypes.MECI,
            OptimizationTypes.Seam,
            OptimizationTypes.TransitionStateSeam
        }:
            assert molecule.evaluator.do_coupling is True, "Turn do_coupling on."
        # elif opt_type not in ['MECI','SEAM','TS-SEAM']:
        #    assert molecule.PES.lot.do_coupling==False,"Turn do_coupling off."
        elif opt_type in  {
            OptimizationTypes.Unconstrained
        }:
            assert ictan is None
        elif opt_type in  {
            OptimizationTypes.ICTAN,
            OptimizationTypes.Climbing,
            OptimizationTypes.TransitionState,
            OptimizationTypes.TransitionStateSeam,
            OptimizationTypes.Beales
        } and not ictan.any():
            raise RuntimeError("Need ictan")
        # if opt_type in ['TS','TS-SEAM']:
        #     assert molecule.isTSnode,"only run climb and eigenvector follow on TSnode."

    # def converged(self,g,nconstraints):
    #    # check if finished
    #    gradrms = np.sqrt(np.dot(g[nconstraints:].T,g[nconstraints:])/n)
    #    #print "current gradrms= %r au" % gradrms
    #    #print "gnorm =",gnorm
    #
    #    gmax = np.max(g[nconstraints:])/ANGSTROM_TO_AU
    #    #print "maximum gradient component (au)", gmax

    #    if gradrms <self.conv_grms:
    #        self.logger.log_print('[INFO] converged')
    #        return True

    #    #if gradrms <= self.conv_grms  or \
    #    #    (self.disp <= self.conv_disp and self.Ediff <= self.conv_Ediff) or \
    #    #    (gmax <= self.conv_gmax and self.Ediff <= self.conv_Ediff):
    #    #    print '[INFO] converged'
    #    #    return True
    #    return False

    def set_lambda1(self, opt_type, eigen, maxoln=None):
        opt_type = OptimizationTypes(opt_type)
        if opt_type == opt_type.TransitionState:
            leig = eigen[1]  # ! this is eigen[0] if update_ic_eigen() ### also diff values
            if maxoln != 0:
                leig = eigen[0]
            if leig < 0. and maxoln == 0:
                lambda1 = -leig
            else:
                lambda1 = 0.01
        else:
            leig = eigen[0]
            if leig < 0:
                lambda1 = -leig+0.015
            else:
                lambda1 = 0.005
        if abs(lambda1) < 0.005:
            lambda1 = 0.005

        return lambda1

    def get_constraint_vectors(self, molecule, opt_type, ictan=None):
        # nconstraints = self.get_nconstraints(opt_type)
        opt_type = OptimizationTypes(opt_type)

        if opt_type == OptimizationTypes.Unconstrained:
            constraints = None
        elif opt_type in {
            OptimizationTypes.ICTAN,
            OptimizationTypes.Climbing,
            OptimizationTypes.Beales
        }:
            constraints = ictan
        elif opt_type == OptimizationTypes.MECI:
            self.logger.log_print("MECI")
            dgrad_U = block_matrix.dot(molecule.coord_basis, molecule.difference_gradient)
            dvec_U = block_matrix.dot(molecule.coord_basis, molecule.derivative_coupling)
            constraints = np.hstack((dgrad_U, dvec_U))
        elif opt_type in {
            OptimizationTypes.Seam,
            OptimizationTypes.TransitionStateSeam
        }:
            dgrad_U = block_matrix.dot(molecule.coord_basis, molecule.difference_gradient)
            dvec_U = block_matrix.dot(molecule.coord_basis, molecule.derivative_coupling)
            constraints = np.hstack((ictan, dgrad_U, dvec_U))
        else:
            raise NotImplementedError
        return constraints

    def get_constraint_steps(self, molecule, opt_type, g):
        # nconstraints = self.get_nconstraints(opt_type)
        n = len(g)
        # TODO Raise Error for CartesianCoordinates

        # TODO 4/24/2019 block matrix/distributed constraints
        constraint_steps = np.zeros((n, 1))
        opt_type = OptimizationTypes(opt_type)

        # 6/5 climb works with block matrix distributed constraints
        # => ictan climb
        if opt_type == OptimizationTypes.Climbing:
            gts = np.dot(g.T, molecule.constraints[:, 0])
            if gts.ndim > 0:
                gts = gts[0]
            # stepsize=np.linalg.norm(constraint_steps)
            max_step = 0.05/self.SCALE_CLIMB
            if gts > np.abs(max_step):
                gts = np.sign(gts)*max_step
                # constraint_steps = constraint_steps*max_step/stepsize
            self.logger.log_print(" gts {gts:1.4f}", gts=gts,)
            constraint_steps = gts*molecule.constraints[:, 0]
            constraint_steps = constraint_steps[:, np.newaxis]
        # => MECI
        elif opt_type == OptimizationTypes.MECI:
            dq = self.dgrad_step(molecule)
            constraint_steps[:, 0] = dq*molecule.constraints[:, 0]

        elif opt_type == OptimizationTypes.Seam:
            dq = self.dgrad_step(molecule)
            constraint_steps[:, 0] = dq*molecule.constraints[:, 1]
        # => seam climb
        elif opt_type == OptimizationTypes.TransitionStateSeam:
            gts = np.dot(g.T, molecule.constraints[:, 0])

            # climbing step
            max_step = 0.05/self.SCALE_CLIMB
            if gts > np.abs(max_step):
                gts = np.sign(gts)*max_step
                # constraint_steps = constraint_steps*max_step/stepsize
            self.logger.log_print(" gts {gts:1.4f}", gts=gts)
            constraint_steps = gts*molecule.constraints[:, 0]
            constraint_steps = constraint_steps[:, np.newaxis]

            # to CI step
            dq = self.dgrad_step(molecule)
            constraint_steps[:, 0] += dq*molecule.constraints[:, 1]

        return constraint_steps

    def dgrad_step(self, molecule):
        """ takes a linear step along dgrad"""

        norm_dg = np.linalg.norm(molecule.difference_gradient)
        self.logger.log_print(
            " norm_dg is {norm_dg:1.4f}",
            norm_dg=norm_dg
        )
        self.logger.log_print(
            " dE is {dE:1.4f}",
            dE=molecule.difference_energy
        )

        dq = -molecule.difference_energy/units.KCAL_MOL_PER_AU/norm_dg
        if dq < self.max_step/5:
            dq = -self.max_step/5
        if dq < -0.075:
            dq = -0.075

        return dq

    def walk_up(self, g, n):
        """ walk up the n'th DLC"""
        # assert isinstance(g[n],float), "gradq[n] is not float!"
        # if self.print_level>0:
        #    self.logger.log_print(' gts: {:1.4f}'.format(self.gradq[n,0]))
        # self.buf.write(' gts: {:1.4f}'.format(self.gradq[n,0]))
        SCALEW = 1.0
        SCALE = self.SCALEQN
        dq = g[n, 0]/SCALE
        # dq = np.dot(g.T,molecule.constraints)*molecule.constraints

        self.logger.log_print(" walking up the {n} coordinate = {dq:1.4f}", n=n, dq=dq)
        if abs(dq) > self.MAXAD/SCALEW:
            dq = np.sign(dq)*self.MAXAD/SCALE
        return dq

    def step_controller(self, step, ratio, gradrms, pgradrms, dEpre, opt_type, dE_iter):
        # => step controller controls DMAX/DMIN <= #

        opt_type = OptimizationTypes(opt_type)
        if opt_type in {OptimizationTypes.TransitionState, OptimizationTypes.Climbing}:
            if ratio < 0. and abs(dEpre) > 0.05:
                self.logger.log_print("sign problem, decreasing DMAX", log_level=self.logger.LogLevel.MoreDebug)
                self.max_step /= 1.35
            elif (ratio < 0.75 or ratio > 1.5):  # and abs(dEpre)>0.05:
                self.logger.log_print(" decreasing DMAX", log_level=self.logger.LogLevel.MoreDebug)
                if step < self.max_step:
                    self.max_step = step/1.1
                else:
                    self.max_step = self.max_step/1.2

            elif ratio > 0.85 and ratio < 1.3:

                # if step>self.max_step and gradrms<(pgradrms*1.35):
                #    self.logger.log_print(" increasing DMAX")
                #    self.max_step *= 1.1
                if gradrms > (pgradrms + 0.0005):
                    self.logger.log_print(' decreasing DMAX, gradrms increased', log_level=self.logger.LogLevel.MoreDebug)
                    self.max_step -= self.max_step/10.
                elif gradrms < (pgradrms + 0.0005):
                    if self.max_step < 0.05:
                        self.logger.log_print([
                            ' increased DMAX, gradrms decreased',
                            '{gradrms}',
                            '{pgradrms}',
                            " increasing DMAX"
                        ],
                            gradrms=gradrms,
                            pgradrms=pgradrms,
                            log_level=self.logger.LogLevel.MoreDebug
                        )
                        self.max_step = self.max_step*1.1
                    elif gradrms < (pgradrms-0.0005) and ratio > 0.9 and ratio < 1.1:
                        self.max_step = self.max_step*1.1

            if self.max_step > 0.25:
                self.max_step = 0.25
        else:
            if dE_iter > 0.001 and opt_type in {OptimizationTypes.Unconstrained, OptimizationTypes.ICTAN}:
                self.logger.log_print(" decreasing DMAX", log_level=self.logger.LogLevel.MoreDebug)
                if step < self.max_step:
                    self.max_step = step/1.5
                else:
                    self.max_step = self.max_step/1.5
            elif (ratio < 0.25 or ratio > 1.5) and abs(dEpre) > 0.05:
                self.logger.log_print(" decreasing DMAX", log_level=self.logger.LogLevel.MoreDebug)
                if step < self.max_step:
                    self.max_step = step/1.1
                else:
                    self.max_step = self.max_step/1.2
            elif ratio > 0.75 and ratio < 1.25 and step > self.max_step and gradrms < (pgradrms*1.35):
                self.logger.log_print(" increasing DMAX", log_level=self.logger.LogLevel.MoreDebug)
                self.max_step = self.max_step*1.1 + 0.01
            if self.max_step > 0.25:
                self.max_step = 0.25

        if self.max_step < self.DMIN:
            self.max_step = self.DMIN
        # print(" DMAX %1.2f" % self.max_step)


    def maxol_w_Hess(self, overlap):
        # Max overlap metrics
        absoverlap = np.abs(overlap)
        path_overlap = np.max(absoverlap)
        path_overlap_n = np.argmax(absoverlap)
        return path_overlap, path_overlap_n
