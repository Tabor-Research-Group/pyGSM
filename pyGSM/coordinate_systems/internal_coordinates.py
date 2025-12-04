#!/usr/bin/env python
import abc
import enum
# standard library imports
import time

# third party
from collections import OrderedDict
import numpy as np
from numpy.linalg import multi_dot

from ..utilities import elements, block_matrix, Devutils as dev
from .topology import EdgeGraph, guess_bonds

ELEMENT_TABLE = elements.ElementData()

CacheWarning = False

__all__ = [
    "InternalCoordinates",
    "get_coordinate_type_name",
    "register_coordinate_system"
]

coordinate_type_registry = {}
def get_coordinate_type_name(system):
    for name, base_type in coordinate_type_registry.items():
        if isinstance(system, base_type):
            return name
    else:
        return None
def register_coordinate_system(name):
    def decorate(cls):
        coordinate_type_registry[name] = cls
        return cls
    return decorate

class InternalCoordinates(metaclass=abc.ABCMeta):

    def __init__(self, atoms, xyz, bonds=None, constraints=None, logger=None):
        if atoms is not None and isinstance(atoms[0], str):
            atoms = [ELEMENT_TABLE.from_symbol(a) for a in atoms]
        self.atoms = atoms
        self.xyz = xyz
        self._input_topology = bonds
        self._topology = None
        self.constraints = constraints
        self.stored_wilsonB = OrderedDict()
        self.logger = dev.Logger.lookup(logger)

    def get_state_dict(self):
        return dict(
            atoms=self.atoms,
            xyz=self.xyz,
            bonds=(
                self._input_topology
                    if self._topology is None else
                self._topology
            ),
            logger=self.logger
        )
    def modify(self, **changes):
        return type(self)(
            **dict(self.get_state_dict(), **changes)
        )
    def copy(self):
        return self.modify()

    @property
    def topology(self):
        if self._topology is None:
            self._topology = self._make_bond_graph(self.atoms, self.xyz, self._input_topology)
        return self._topology
    @classmethod
    def _make_bond_graph(cls, atoms, coords, bonds):
        if bonds is None:
            bonds = guess_bonds(atoms, coords)
        if not isinstance(bonds, EdgeGraph):
            if len(bonds[0]) > 2:
                edge_types = [b[2] for b in bonds]
            else:
                edge_types = 1
            bonds = EdgeGraph(np.arange(len(atoms)), [b[:2] for b in bonds], edge_types)
        return bonds

    @abc.abstractmethod
    def get_opts_for_report(self) -> dict:
        ...

    @abc.abstractmethod
    def calcDiff(self, xyz2, xyz1):
        ...

    @abc.abstractmethod
    def guess_hessian(self, coords, bonds=None):
        ...

    @abc.abstractmethod
    def derivatives(self, xyz):
        ...
    @abc.abstractmethod
    def second_derivatives(self, xyz):
        ...

    def clearCache(self):
        self.stored_wilsonB = OrderedDict()

    def wilsonB(self, xyz, use_cache=False):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        if use_cache:
            raise NotImplementedError("caching slower than derivs")
            global CacheWarning
            t0 = time.time()
            xhash = hash(xyz.tostring())
            ht = time.time() - t0
            if xhash in self.stored_wilsonB:
                ans = self.stored_wilsonB[xhash]
                return ans
            WilsonB = self.wilsonB(xyz, use_cache=False)
            self.stored_wilsonB[xhash] = WilsonB
            if len(self.stored_wilsonB) > 1000 and not CacheWarning:
                self.logger.log_print("\x1b[91mWarning: more than 100 B-matrices stored, memory leaks likely\x1b[0m")
                CacheWarning = True
        else:
            WilsonB = self.compute_bmatrix(xyz)
        return WilsonB

    def compute_bmatrix(self, xyz):
        WilsonB = []
        Der = self.derivatives(xyz)
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        return np.array(WilsonB)

    def GMatrix(self, xyz, u=None):
        """
        Given Cartesian coordinates xyz, return the G-matrix
        given by G = BuBt where u is an arbitrary matrix (default to identity)
        """
        # t0 = time.time()
        Bmat = self.wilsonB(xyz)
        if isinstance(Bmat, block_matrix):
            if u is not None:
                raise ValueError("block matrix mutl with `u` not supported")
            BuBt = block_matrix.dot(Bmat, block_matrix.transpose(Bmat))
        else:
            if u is None:
                BuBt = np.dot(Bmat, Bmat.T)
            else:
                BuBt = np.dot(Bmat, np.dot(u, Bmat.T))
        # t2 = time.time()
        # t10 = t1-t0
        # t21 = t2-t1
        # print("time to form B-matrix %.3f" % t10)
        # print("time to mat-mult B %.3f" % t21)
        return BuBt

    def GInverse_SVD(self, xyz):
        xyz = xyz.reshape(-1, 3)
        # Perform singular value decomposition
        # nifty.click()
        loops = 0
        while True:
            try:
                G = self.GMatrix(xyz)
                # time_G = nifty.click()
                U, S, VT = np.linalg.svd(G)
                # time_svd = nifty.click()
            except np.linalg.LinAlgError:
                self.logger.log_print("\x1b[1;91m SVD fails, perturbing coordinates and trying again\x1b[0m")
                xyz = xyz + 1e-2*np.random.random(xyz.shape)
                loops += 1
                if loops == 10:
                    raise RuntimeError('SVD failed too many times')
                continue
            break
        # print "Build G: %.3f SVD: %.3f" % (time_G, time_svd),
        V = VT.T
        UT = U.T
        Sinv = np.zeros_like(S)
        LargeVals = 0
        for ival, value in enumerate(S):
            # print "%.5e % .5e" % (ival,value)
            if np.abs(value) > 1e-6:
                LargeVals += 1
                Sinv[ival] = 1/value
        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        Sinv = np.diag(Sinv)
        Inv = multi_dot([V, Sinv, UT])
        return Inv

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1, 3)
        # nifty.click()
        G = self.GMatrix(xyz)
        # time_G = nifty.click()
        Gi = np.linalg.inv(G)
        # time_inv = nifty.click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return Gi

    def GInverse(self, xyz):
        # 9/2019 CRA what is the difference in performace/stability for SVD vs regular inverse?
        return self.GInverse_EIG(xyz)

    def calcGrad(self, xyz, gradx, frozen_atoms=None):
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)

        res = block_matrix.dot(Ginv, block_matrix.dot(Bmat, gradx))

        # Internal coordinate gradient
        return res

    def calcHess(self, xyz, gradx, hessx):
        """
        Compute the internal coordinate Hessian. 
        Expects Cartesian coordinates to be provided in a.u.
        """
        # xyz = xyz.flatten()
        # self.calculate(xyz)
        # Ginv = self.GInverse(xyz)
        # Bmat = self.wilsonB(xyz)
        # Gq = self.calcGrad(xyz, gradx)
        # deriv2 = self.second_derivatives(xyz)
        # Bmatp = deriv2.reshape(deriv2.shape[0], xyz.shape[0], xyz.shape[0])
        # Hx_BptGq = hessx - np.einsum('pmn,p->mn', Bmatp, Gq)
        # Hq = np.einsum('ps,sm,mn,nr,rq', Ginv, Bmat, Hx_BptGq, Bmat.T, Ginv, optimize=True)
        # return Hq

        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Ginv = block_matrix.full_matrix(Ginv)
        Bmat = self.wilsonB(xyz)
        Bmat = block_matrix.full_matrix(Bmat)

        # np.einsum('pmn,p->mn',Bmatp,Gq)
        BptGq = self.calcCg(xyz,gradx)
        Hx_BptGq = hessx - BptGq 

        Hq = np.einsum('ps,sm,mn,nr,rq', Ginv, Bmat, Hx_BptGq, Bmat.T, Ginv, optimize=True)
        return Hq

    def readCache(self, xyz, dQ):
        if not hasattr(self, 'stored_xyz'):
            return None
        if np.linalg.norm(self.stored_xyz - xyz) < 1e-10:
            if np.linalg.norm(self.stored_dQ - dQ) < 1e-10:
                return self.stored_newxyz
        return None

    def writeCache(self, xyz, dQ, newxyz):
        # xyz = xyz.flatten()
        # dQ = dQ.flatten()
        # newxyz = newxyz.flatten()
        self.stored_xyz = xyz.copy()
        self.stored_dQ = dQ.copy()
        self.stored_newxyz = newxyz.copy()
  
    def newCartesian(self, xyz, dQ, frozen_atoms=None, verbose=True):
        #TODO: make this actually return something...
        cached = self.readCache(xyz, dQ)
        if cached is not None:
            # print "Returning cached result"
            return cached
        xyz1 = xyz.copy()
        dQ1 = dQ.flatten()
        # Iterate until convergence:
        microiter = 0
        ndqs = []
        ndqt = 100.
        rmsds = []
        self.bork = False
        # Damping factor
        damp = 1.0
        
        # Function to exit from loop
        def finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1):
            if ndqt > 1e-1:
                self.logger.log_print(
                    " Failed to obtain coordinates after %i microiterations (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt),
                    log_level=self.logger.LogLevel.Debug
                )
                self.bork = True
                self.writeCache(xyz, dQ, xyz_iter1)
                return xyzsave.reshape((-1, 3))
            elif ndqt > 1e-3:
                self.logger.log_print(
                    " Approximate coordinates obtained after %i microiterations (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt),
                    log_level=self.logger.LogLevel.Debug
                )
            else:
                self.logger.log_print(
                    " Cartesian coordinates obtained after %i microiterations (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt),
                    log_level=self.logger.LogLevel.Debug
                )
            self.writeCache(xyz, dQ, xyzsave)
            return xyzsave.reshape((-1, 3))

        fail_counter = 0
        while True:
            microiter += 1
            Bmat = self.wilsonB(xyz1)
            Ginv = self.GInverse(xyz1)

            # Get new Cartesian coordinates
            dxyz = damp*block_matrix.dot(block_matrix.transpose(Bmat), block_matrix.dot(Ginv, dQ1))

            if frozen_atoms is not None:
                for a in [3*i for i in frozen_atoms]:
                    dxyz[a:a+3] = 0.

            xyz2 = xyz1 + dxyz.reshape((-1, 3))
            if microiter == 1:
                xyzsave = xyz2.copy()
                xyz_iter1 = xyz2.copy()
            # Calculate the actual change in internal coordinates
            dQ_actual = self.calcDiff(xyz2, xyz1)
            rmsd = np.sqrt(np.mean((np.array(xyz2-xyz1).flatten())**2))
            ndq = np.linalg.norm(dQ1-dQ_actual)
            if len(ndqs) > 0:
                if ndq > ndqt:
                    self.logger.log_print(
                        " Iter: {microiter} Err-dQ (Best) = {ndq:.5e} ({ndqt:.5e}) RMSD: {rmsd:.5e} Damp: {damp:.5e} (Bad)",
                        microiter=microiter, ndq=ndq, ndqt=ndqt, rmsd=rmsd, damp=damp,
                        log_level=self.logger.LogLevel.Debug
                    )
                    damp /= 2
                    fail_counter += 1
                    # xyz2 = xyz1.copy()
                else:
                    self.logger.log_print(
                        " Iter: {microiter} Err-dQ (Best) = {ndq:.5e} ({ndqt:.5e}) RMSD: {rmsd:.5e} Damp: {damp:.5e} (Good)",
                        microiter=microiter, ndq=ndq, ndqt=ndqt, rmsd=rmsd, damp=damp,
                        log_level=self.logger.LogLevel.Debug
                    )
                    fail_counter = 0
                    damp = min(damp*1.2, 1.0)
                    rmsdt = rmsd
                    ndqt = ndq
                    xyzsave = xyz2.copy()
            else:
                self.logger.log_print(
                    " Iter: {microiter} Err-dQ = {ndq:.5e} RMSD: {rmsd:.5e} Damp: {damp:.5e}",
                    microiter=microiter, ndq=ndq, rmsd=rmsd, damp=damp,
                    log_level=self.logger.LogLevel.Debug
                )
                rmsdt = rmsd
                ndqt = ndq
            ndqs.append(ndq)
            rmsds.append(rmsd)
            # Check convergence / fail criteria
            if rmsd < 1e-6 or ndq < 1e-6:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if fail_counter >= 5:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if microiter == 50:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            # Figure out the further change needed
            dQ1 = dQ1 - dQ_actual
            xyz1 = xyz2.copy()
