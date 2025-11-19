from __future__ import print_function
from ..utilities import block_matrix, math_utils

# standard library imports
from sys import exit
from time import time

# third party
# from .networkx_loader import nx
# from collections import OrderedDict, defaultdict
from numpy.linalg import multi_dot
import numpy as np

# local application imports
from .internal_coordinates import InternalCoordinates, register_coordinate_system
from .primitive_internals import PrimitiveInternalCoordinates
# from . topology import Topology
from .slots import *

@register_coordinate_system("delocalized")
class DelocalizedInternalCoordinates(InternalCoordinates):

    def __init__(self,
                 atoms,
                 xyz,
                 primitives,
                 constraints=None
                 ):
        #TODO: make all of this inherited from the primitives
        #      currently could clash
        super().__init__(
            atoms,
            xyz,
            logger=primitives.logger,
            constraints=constraints
        )

        # self.options = options
        self.atoms = atoms
        self.natoms = len(self.atoms)

        self.Prims = primitives
        # print "in constructor",len(self.Prims.Internals)

        xyz = np.asanyarray(xyz).flatten()
        self.Vecs, self.Internals = self.build_dlc(self.Prims, xyz, constraints=self.constraints, logger=self.logger)
        # print("vecs after build")
        # print(self.Vecs)

    def update_dlc(self, new_xyz, constraints=None):
        #TODO: make this not an in-place operation
        if constraints is not None:
            self.constraints = constraints
        self.Vecs, self.Internals = self.build_dlc(self.Prims, new_xyz, logger=self.logger, constraints=self.constraints)
        return self.Vecs

    def get_state_dict(self):
        return dict(
            atoms=self.atoms,
            xyz=self.xyz,
            primitives=self.Prims
        )
    def modify(self, bonds=None, primitives=None, **changes):
        base_state = self.get_state_dict()
        if primitives is None:
            primitives = base_state['primitives']
        if bonds is not None:
            primitives = primitives.modify(bonds=bonds)
        return type(self)(
            **dict(base_state, primitives=primitives, **changes)
        )
    def copy(self):
        return self.modify(primitives=self.Prims.copy())

    @classmethod
    def make_primitives(cls, primitives, **kwargs):
        # The DLC contains an instance of primitive internal coordinates.
        if primitives is None:
            prims = PrimitiveInternalCoordinates(**kwargs)
        else:
            # print(" warning: not sure if a deep copy prims")
            # self.Prims=self.options['primitives']
            prims = primitives.copy()
            prims.clearCache()
        return prims

    def clearCache(self):
        super().clearCache()
        self.Prims.clearCache()

    # def __repr__(self):
    #     return self.Prims.__repr__()

    def update(self, other):
        return self.Prims.update(other.Prims)

    def join(self, other):
        return self.Prims.join(other.Prims)

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

    nonzero_gel_cutoff = 1e-6
    @classmethod
    def build_dlc(cls, prims, xyz, *, logger, constraints=None):
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
        constraints       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        # print(" Beginning to build G Matrix")
        G = prims.GMatrix(xyz)  # in primitive coords
        # print(" Timings: Build G: %.3f " % (time_G))

        tmpvecs = []
        for A in G.matlist:
            L, Q = np.linalg.eigh(A)
            LargeIdx = np.where(np.abs(L) > cls.nonzero_gel_cutoff)[0]
            # LargeVals = 0
            # LargeIdx = []
            # for ival, value in enumerate(L):
            #     # print("val=%.4f" %value,end=' ')
            #     if np.abs(value) > 1e-6:
            #         LargeVals += 1
            #         LargeIdx.append(ival)
            # # print('\n')
            # # print("LargeVals %i" % LargeVals)
            tmpvecs.append(Q[:, LargeIdx])

        vecs = block_matrix(tmpvecs)
        # print(" shape of DLC")
        # print(self.Vecs.shape)

        internals = ["DLC %i" % (i+1) for i in range(vecs.shape[1])]

        # #NOTE: this was bugged anyway, clearly a dead code path
        # # Vecs has number of rows equal to the number of primitives, and
        # # number of columns equal to the number of delocalized internal coordinates.
        # if self.haveConstraints():
        #     assert self.cVec is None, "can't have vector constraint and cprim."
        #     self.cVec = self.form_cVec_from_cPrims()

        if constraints is not None:
            # orthogonalize
            constraints = np.asanyarray(constraints)
            if np.sum(np.abs(constraints)) < 1e-12:
                raise ValueError("empty constraint passed")
            Cn = math_utils.orthogonalize(constraints)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?

            cVecs = block_matrix.dot(block_matrix.dot(vecs, block_matrix.transpose(vecs)), Cn)

            # normalize C_U
            try:
                # print(cVecs.T)
                cVecs = math_utils.orthogonalize(cVecs)
            except ValueError:
                raise ValueError("failed to orthogonalize constraint vectors")
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

        self.logger.print(" starting to build G prim")
        G = self.Prims.GMatrix(xyz)  # in primitive coords

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
                self.logger.print(cVecs)
                self.logger.print("error forming cVec")
                raise

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

    def guess_hessian(self, coords, bonds=None):
        """ Build the guess Hessian, consisting of a diagonal matrix
        in the primitive space and changed to the basis of DLCs. """
        Hprim = self.Prims.guess_hessian(coords, bonds)
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
        self.logger.log_print(self.Vecs.shape)
        self.logger.log_print(xvec.shape)

        # find what linear combination of DLC basis xvec and yvec is
        proj_xvec = block_matrix.dot(block_matrix.transpose(self.Vecs), xvec)
        proj_yvec = block_matrix.dot(block_matrix.transpose(self.Vecs), yvec)

        #proj_xvec = block_matrix.dot(self.Vecs,xvec)
        #proj_yvec = block_matrix.dot(self.Vecs,yvec)

        self.logger.log_print(proj_xvec.T)
        self.logger.log_print(proj_yvec.T)
        self.logger.log_print(proj_xvec.shape)

        rc = 0
        for xrow, yrow in zip(xv, yv):
            cc = 0
            for xx, yy in zip(xrow, yrow):

                # first form the vector in the grid as the linear comb of the projected vectors
                dq = xx * proj_xvec + yy*proj_yvec
                self.logger.log_print(dq.T)
                self.logger.log_print(dq.shape)

                # convert to xyz and save to xyzgrid
                xyzgrid[rc, cc, :] = self.newCartesian(xyz, dq).flatten()
                cc += 1
            rc += 1

        return xyzgrid


if __name__ == '__main__' and __package__ is None:
    pass
