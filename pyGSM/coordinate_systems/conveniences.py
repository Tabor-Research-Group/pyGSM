
from .cartesian import CartesianCoordinates


def construct_coordinate_system(atoms, xyz,
                                bonds=None,
                                primitives=None,
                                topology=None
                                ):
    if primitives is None:
        ...
    return CartesianCoordinates()