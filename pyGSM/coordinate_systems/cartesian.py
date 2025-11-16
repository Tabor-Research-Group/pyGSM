import numpy as np
from .internal_coordinates import InternalCoordinates
from .primitive_internals import PrimitiveInternalCoordinates
from .slots import *

class CartesianCoordinates(InternalCoordinates):
    """
    Cartesian coordinate system, written as a kind of internal coordinate class.  
    This one does not support constraints, because that requires adding some 
    primitive internal coordinates.
    """
    def __init__(self, atoms, xyz, topology=None, internals=None, form_topology=False, **etc):
        super(CartesianCoordinates, self).__init__(atoms, xyz)
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.natoms = len(self.atoms)
        if internals is None:
            internals = PrimitiveInternalCoordinates(
                atoms,
                xyz,
                topology,
                form_topology=form_topology,
                **etc
            )
            for i in range(self.natoms):
                internals.add(CartesianX(i, w=1.0))
                internals.add(CartesianY(i, w=1.0))
                internals.add(CartesianZ(i, w=1.0))
        self.Prims = internals
        #self.Prims = PrimitiveInternalCoordinates(options.copy())

        #if 'constraints' in kwargs and kwargs['constraints'] is not None:
        #    raise RuntimeError('Do not use constraints with Cartesian coordinates')

        self.Vecs = np.eye(3*self.natoms)

    def copy(self):
        return type(self)(
            self.atoms,
            self.xyz,
            topology=self.Prims.topology,
            internals=self.Prims.copy(),
            logger=self.logger
        )

    def guess_hessian(self, xyz, bonds=None):
        return 0.5*np.eye(len(xyz.flatten()))


    def calcGrad(self, xyz, gradx, frozen_atoms=None):
        #TODO: handle freezing
        return gradx

    def newCartesian(self,xyz, dq, verbose=True, frozen_atoms=None):
        return xyz+np.reshape(dq,(-1,3))
    
    def calculate(self,coords):
        return coords

    def calcDiff(self, xyz2, xyz1):
        raise NotImplementedError("shouldn't hit this code path for Cartesian coordinates")

    def derivatives(self, xyz):
        raise NotImplementedError("shouldn't hit this code path for Cartesian coordinates")

    def second_derivatives(self, xyz):
        raise NotImplementedError("shouldn't hit this code path for Cartesian coordinates")

    def addConstraint(self, cPrim, cVal, xyz):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def haveConstraints(self):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")