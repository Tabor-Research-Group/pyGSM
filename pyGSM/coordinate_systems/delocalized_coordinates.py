from __future__ import print_function
from ..utilities import block_matrix

# standard library imports
from sys import exit
from time import time

# third party
# from .networkx_loader import nx
# from collections import OrderedDict, defaultdict
from numpy.linalg import multi_dot
import numpy as np

# local application imports
from .internal_coordinates import InternalCoordinates
from .primitive_internals import PrimitiveInternalCoordinates
# from . topology import Topology
from .slots import *

class DelocalizedInternalCoordinates(InternalCoordinates):

    def __init__(self,
                 atoms,
                 xyz,
                 primitives,
                 constraints
                 ):
        super().__init__(
            atoms,
            xyz
        )

        # self.options = options
        self.constraints = constraints
        self.atoms = atoms
        self.natoms = len(self.atoms)

        self.Prims = primitives
        # print "in constructor",len(self.Prims.Internals)

        xyz = np.asanyarray(xyz).flatten()
        self.Vecs, self.Internals = self.build_dlc(self.Prims, xyz)
        # print("vecs after build")
        # print(self.Vecs)

    @classmethod
    def make_primitives(cls, primitives, **kwargs):
        # The DLC contains an instance of primitive internal coordinates.
        if primitives is None:
            print(" making primitives ")
            t1 = time()
            prims = PrimitiveInternalCoordinates(**kwargs)
            dt = time() - t1
            print(" Time to make prims %.3f" % dt)
        else:
            print(" setting primitives from options!")
            # print(" warning: not sure if a deep copy prims")
            # self.Prims=self.options['primitives']
            t0 = time()
            prims = primitives.copy()

            print(" num of primitives {}".format(len(prims.Internals)))
            dt = time() - t0
            print(" Time to copy prims %.3f" % dt)
            prims.clearCache()
        return prims

    def clearCache(self):
        super().clearCache()
        self.Prims.clearCache()

    def __repr__(self):
        return self.Prims.__repr__()

    def update(self, other):
        return self.Prims.update(other.Prims)

    def join(self, other):
        return self.Prims.join(other.Prims)

    def copy(self, xyz):
        return type(self)(
            self.atoms,
            xyz,
            self.Prims,
            self.constraints
        )

    def addConstraint(self, cPrim, cVal, xyz):
        self.Prims.addConstraint(cPrim, cVal, xyz)

    def getConstraints_from(self, other):
        self.Prims.getConstraints_from(other.Prims)

    def haveConstraints(self):
        return len(self.Prims.cPrims) > 0

    def getConstraintViolation(self, xyz):
        return self.Prims.getConstraintViolation(xyz)

    def printConstraints(self, xyz, thre=1e-5):
        self.Prims.printConstraints(xyz, thre=thre)

    def getConstraintTargetVals(self):
        return self.Prims.getConstraintTargetVals()

    def applyConstraints(self, xyz):
        """
        Pass in Cartesian coordinates and return new coordinates that satisfy the constraints exactly.
        This is not used in the current constrained optimization code that uses Lagrange multipliers instead.
        """
        xyz1 = xyz.copy()
        niter = 0
        while True:
            dQ = np.zeros(len(self.Internals), dtype=float)
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Look up the index of the DLC that corresponds to the constraint
                iDLC = self.cDLC[ic]
                # Calculate the further change needed in this constrained variable
                dQ[iDLC] = (self.Prims.cVals[ic] - c.value(xyz1))/self.Vecs[iPrim, iDLC]
                if c.isPeriodic:
                    Plus2Pi = dQ[iDLC] + 2*np.pi
                    Minus2Pi = dQ[iDLC] - 2*np.pi
                    if np.abs(dQ[iDLC]) > np.abs(Plus2Pi):
                        dQ[iDLC] = Plus2Pi
                    if np.abs(dQ[iDLC]) > np.abs(Minus2Pi):
                        dQ[iDLC] = Minus2Pi
            # print "applyConstraints calling newCartesian (%i), |dQ| = %.3e" % (niter, np.linalg.norm(dQ))
            xyz2 = self.newCartesian(xyz1, dQ, verbose=False)
            if np.linalg.norm(dQ) < 1e-6:
                return xyz2
            if niter > 1 and np.linalg.norm(dQ) > np.linalg.norm(dQ0):
                # logger.warning("\x1b[1;93mWarning: Failed to apply Constraint\x1b[0m")
                return xyz1
            xyz1 = xyz2.copy()
            niter += 1
            dQ0 = dQ.copy()

    def newCartesian_withConstraint(self, xyz, dQ, thre=0.1, verbose=False):
        xyz2 = self.newCartesian(xyz, dQ, verbose)
        constraintSmall = len(self.Prims.cPrims) > 0
        for ic, c in enumerate(self.Prims.cPrims):
            w = c.w if type(c) in [RotationA, RotationB, RotationC] else 1.0
            current = c.value(xyz)/w
            reference = self.Prims.cVals[ic]/w
            diff = (current - reference)
            if np.abs(diff-2*np.pi) < np.abs(diff):
                diff -= 2*np.pi
            if np.abs(diff+2*np.pi) < np.abs(diff):
                diff += 2*np.pi
            if np.abs(diff) > thre:
                constraintSmall = False
        if constraintSmall:
            xyz2 = self.applyConstraints(xyz2)
        return xyz2

    def wilsonB(self, xyz):
        Bp = self.Prims.wilsonB(xyz)
        return block_matrix.dot(block_matrix.transpose(self.Vecs), Bp)

    #def calcGrad(self, xyz, gradx):
    #    #q0 = self.calculate(xyz)
    #    Ginv = self.GInverse(xyz)
    #    Bmat = self.wilsonB(xyz)

    #    if self.frozen_atoms is not None:
    #        for a in [3*i for i in self.frozen_atoms]:
    #            gradx[a:a+3,0]=0.

    #    # Internal coordinate gradient
    #    # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx)
    #    nifty.click()
    #    #Gq = multi_dot([Ginv, Bmat, gradx])
    #    Bg = block_matrix.dot(Bmat,gradx)
    #    Gq = block_matrix.dot( Ginv, Bg)
    #    #print("time to do block mult %.3f" % nifty.click())
    #    #Gq = np.dot(np.multiply(np.diag(Ginv)[:,None],Bmat),gradx)
    #    #print("time to do efficient mult %.3f" % nifty.click())
    #    return Gq

    @classmethod
    def build_dlc(cls, prims, xyz, C=None):
        """
        Build the delocalized internal coordinates (DLCs) which are linear
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.

        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N.

        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz     : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units
        C       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        nifty.click()
        # print(" Beginning to build G Matrix")
        G = prims.GMatrix(xyz)  # in primitive coords
        time_G = nifty.click()
        # print(" Timings: Build G: %.3f " % (time_G))

        tmpvecs = []
        for A in G.matlist:
            L, Q = np.linalg.eigh(A)
            LargeVals = 0
            LargeIdx = []
            for ival, value in enumerate(L):
                # print("val=%.4f" %value,end=' ')
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    LargeIdx.append(ival)
            # print('\n')
            # print("LargeVals %i" % LargeVals)
            tmpvecs.append(Q[:, LargeIdx])

        vecs = block_matrix(tmpvecs)
        # print(" shape of DLC")
        # print(self.Vecs.shape)

        time_eig = nifty.click()
        print(" Timings: Build G: %.3f Eig: %.3f" % (time_G, time_eig))

        internals = ["DLC %i" % (i+1) for i in range(vecs.shape[1])]

        # #NOTE: this was bugged anyway, clearly a dead code path
        # # Vecs has number of rows equal to the number of primitives, and
        # # number of columns equal to the number of delocalized internal coordinates.
        # if self.haveConstraints():
        #     assert self.cVec is None, "can't have vector constraint and cprim."
        #     self.cVec = self.form_cVec_from_cPrims()

        if C is not None:
            # orthogonalize
            if (C[:] == 0.).all():
                raise RuntimeError
            Cn = math_utils.orthogonalize(C)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?

            cVecs = block_matrix.dot(block_matrix.dot(vecs, block_matrix.transpose(vecs)), Cn)

            # normalize C_U
            try:
                # print(cVecs.T)
                cVecs = math_utils.orthogonalize(cVecs)
            except Exception as e:
                raise ValueError("failed to orthogonalize constrain vectors")
                # print(cVecs)
                # print("error forming cVec")
                # exit(-1)

            # project constraints into vectors
            vecs = block_matrix.project_constraint(vecs, cVecs)
            # print(" shape of DLC")
            # print(self.Vecs.shape)
        return vecs, internals

    def build_dlc_conjugate(self, xyz, C=None):
        """
        Build the delocalized internal coordinates (DLCs) which are linear
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.

        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N.

        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz     : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units
        C       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        print(" starting to build G prim")
        nifty.click()
        G = self.Prims.GMatrix(xyz)  # in primitive coords
        time_G = nifty.click()
        print(" Timings: Build G: %.3f " % (time_G))

        tmpvecs = []
        for A in G.matlist:
            L, Q = np.linalg.eigh(A)
            LargeVals = 0
            LargeIdx = []
            for ival, value in enumerate(L):
                # print("val=%.4f" %value,end=' ')
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    LargeIdx.append(ival)
            # print("LargeVals %i" % LargeVals)
            tmpvecs.append(Q[:, LargeIdx])
        self.Vecs = block_matrix(tmpvecs)
        time_eig = nifty.click()
        # print(" Timings: Build G: %.3f Eig: %.3f" % (time_G, time_eig))

        self.Internals = ["DLC %i" % (i+1) for i in range(len(LargeIdx))]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        # if self.haveConstraints():
        #     assert cVec is None, "can't have vector constraint and cprim."
        #     cVec = self.form_cVec_from_cPrims()

        # TODO use block diagonal
        if C is not None:
            # orthogonalize
            if (C[:] == 0.).all():
                raise RuntimeError
            G = block_matrix.full_matrix(self.Prims.GMatrix(xyz))
            Cn = math_utils.conjugate_orthogonalize(C, G)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?
            cVecs = block_matrix.dot(block_matrix.dot(self.Vecs, block_matrix.transpose(self.Vecs)), Cn)

            # normalize C_U
            try:
                cVecs = math_utils.conjugate_orthogonalize(cVecs, G)
            except:
                print(cVecs)
                print("error forming cVec")
                exit(-1)

            # project constraints into vectors
            self.Vecs = block_matrix.project_conjugate_constraint(self.Vecs, cVecs, G)

        return

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def largeRots(self):
        """ Determine whether a molecule has rotated by an amount larger than some threshold (hardcoded in Prims.largeRots()). """
        return self.Prims.largeRots()

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = block_matrix.dot(block_matrix.transpose(self.Vecs), PMDiff)
        return np.array(Answer).flatten()

    def calculate(self, coords):
        """ Calculate the DLCs given the Cartesian coordinates. """
        PrimVals = self.Prims.calculate(coords)
        Answer = block_matrix.dot(block_matrix.transpose(self.Vecs), PrimVals)
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        # raw_input()
        return np.array(Answer).flatten()

    def calcPrim(self, vecq):
        # To obtain the primitive coordinates from the delocalized internal coordinates,
        # simply multiply self.Vecs*Answer.T where Answer.T is the column vector of delocalized
        # internal coordinates. That means the "c's" in Equation 23 of Schlegel's review paper
        # are simply the rows of the Vecs matrix.
        return block_matrix.dot(self.Vecs, vecq)

    # overwritting the parent internalcoordinates GMatrix
    # which is an elegant way to use the derivatives
    # but there is a more efficient way to compute G
    # using the block diagonal properties of G and V
    def GMatrix(self, xyz):
        tmpvecs = []
        Gp = self.Prims.GMatrix(xyz)
        Vt = block_matrix.transpose(self.Vecs)
        for vt, G, v in zip(Vt.matlist, Gp.matlist, self.Vecs.matlist):
            tmpvecs.append(np.dot(np.dot(vt, G), v))
        return block_matrix(tmpvecs)

    def MW_GMatrix(self, xyz, mass):
        tmpvecs = []
        s3a = 0
        Bp = self.Prims.wilsonB(xyz)
        Vt = block_matrix.transpose(self.Vecs)

        s3a = 0
        for vt, b, v in zip(Vt.matlist, Bp.matlist, self.Vecs.matlist):
            e3a = s3a + b.shape[1]
            tmpvecs.append(np.linalg.multi_dot([vt, b/mass[s3a:e3a], b.T, v]))
            s3a = e3a
        return block_matrix(tmpvecs)

    def derivatives(self, coords):
        """ Obtain the change of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.derivatives(coords)
        Answer = np.zeros((self.Vecs.shape[1], PrimDers.shape[1], PrimDers.shape[2]), dtype=float)

        # block matrix tensor dot
        count = 0
        for block in self.Vecs.matlist:
            for i in range(block.shape[1]):
                for j in range(block.shape[0]):
                    Answer[count, :, :] += block[j, i] * PrimDers[j, :, :]
                count += 1

        # print(" block matrix way")
        # print(Answer)
        # tmp = block_matrix.full_matrix(self.Vecs)
        # Answer1 = np.tensordot(tmp, PrimDers, axes=(0, 0))
        # print(" np way")
        # print(Answer1)
        # return np.array(Answer1)

        return Answer

    def second_derivatives(self, coords):
        """ Obtain the second derivatives of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.second_derivatives(coords)

        # block matrix tensor dot
        Answer = np.zeros((self.Vecs.shape[1], coords.shape[0], 3, coords.shape[0], 3))
        count = 0
        for block in self.Vecs.matlist:
            for i in range(block.shape[1]):
                for j in range(block.shape[0]):
                    Answer[count, :, :, :, :] += block[j, i] * PrimDers[j, :, :, :, :]
                count += 1

        # print(" block matrix way.")
        # print(Answer[0])

        # tmp = block_matrix.full_matrix(self.Vecs)
        # Answer2 = np.tensordot(tmp, PrimDers, axes=(0, 0))
        # print(" np tensor dot with full mat")
        # print(Answer2[0])
        # return np.array(Answer2)

        return Answer

    def MW_GInverse(self, xyz, mass):
        xyz = xyz.reshape(-1, 3)
        # nifty.click()
        G = self.MW_GMatrix(xyz, mass)
        # time_G = nifty.click()
        tmpGi = [np.linalg.inv(g) for g in G.matlist]
        # time_inv = nifty.click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return block_matrix(tmpGi)

    def GInverse(self, xyz):
        return self.GInverse_EIG(xyz)

    # TODO this needs to be fixed
    def GInverse_diag(self, xyz):
        t0 = time()
        G = self.GMatrix(xyz)
        # print(G)
        dt = time() - t0
        # print(" total time to get GMatrix %.3f" % dt)
        d = np.diagonal(G)
        # print(d)
        return np.diag(1./d)

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1, 3)
        # nifty.click()
        G = self.GMatrix(xyz)
        # time_G = nifty.click()
        # Gi = np.linalg.inv(G)
        tmpGi = [np.linalg.inv(g) for g in G.matlist]
        # time_inv = nifty.click()
        # print("G-time: %.3f Inv-time: %.3f" % (time_G, time_inv))
        return block_matrix(tmpGi)

    def repr_diff(self, other):
        return self.Prims.repr_diff(other.Prims)

    def guess_hessian(self, coords):
        """ Build the guess Hessian, consisting of a diagonal matrix
        in the primitive space and changed to the basis of DLCs. """
        Hprim = self.Prims.guess_hessian(coords)
        return multi_dot([self.Vecs.T, Hprim, self.Vecs])

    def resetRotations(self, xyz):
        """ Reset the reference geometries for calculating the orientational variables. """
        self.Prims.resetRotations(xyz)

    def create2dxyzgrid(self, xyz, xvec, yvec, nxx, nyy, mag):
        '''
        xvec and yvec are some delocalized coordinate basis vector (or some linear combination of them)
        nx and ny are the number of grid points
        mag is the step along the delocalized coordinate basis. Don't recommend using greater than 0.5
        returns an xyz grid to calculate energies on (see potential_energy_surface modules).

        '''

        x = np.linspace(-mag, mag, nxx)
        y = np.linspace(-mag, mag, nyy)
        xv, yv = np.meshgrid(x, y)
        xyz1 = xyz.flatten()
        xyzgrid = np.zeros((xv.shape[0], xv.shape[1], xyz1.shape[0]))
        print(self.Vecs.shape)
        print(xvec.shape)

        # find what linear combination of DLC basis xvec and yvec is
        proj_xvec = block_matrix.dot(block_matrix.transpose(self.Vecs), xvec)
        proj_yvec = block_matrix.dot(block_matrix.transpose(self.Vecs), yvec)

        #proj_xvec = block_matrix.dot(self.Vecs,xvec)
        #proj_yvec = block_matrix.dot(self.Vecs,yvec)

        print(proj_xvec.T)
        print(proj_yvec.T)
        print(proj_xvec.shape)

        rc = 0
        for xrow, yrow in zip(xv, yv):
            cc = 0
            for xx, yy in zip(xrow, yrow):

                # first form the vector in the grid as the linear comb of the projected vectors
                dq = xx * proj_xvec + yy*proj_yvec
                print(dq.T)
                print(dq.shape)

                # convert to xyz and save to xyzgrid
                xyzgrid[rc, cc, :] = self.newCartesian(xyz, dq).flatten()
                cc += 1
            rc += 1

        return xyzgrid


if __name__ == '__main__' and __package__ is None:
    pass
