
from .internal_coordinates import InternalCoordinates
from .cartesian import CartesianCoordinates
from .primitive_internals import PrimitiveInternalCoordinates
from .delocalized_coordinates import DelocalizedInternalCoordinates
from . import slots
from ..utilities import Devutils as dev
from .topology import guess_bonds

__all__ = [
    "construct_internal",
    "construct_coordinate_system"
]

def infer_spec(coord_data):
    if len(coord_data) == 2:
        a,b = coord_data
        return {
            "type":"distance",
            "a":a,
            "b":b
        }
    elif len(coord_data) == 3:
        a, b, c = coord_data
        return {
            "type": "angle",
            "a": a,
            "b": b,
            "c": c
        }
    elif len(coord_data) == 3:
        a, b, c, d = coord_data
        return {
            "type": "dihedral",
            "a": a,
            "b": b,
            "c": c,
            "d": d
        }
    else:
        raise ValueError(f"can't interpret internal coordinate spec {coord_data}")
def construct_internal(spec):
    if isinstance(spec, slots.PrimitiveCoordinate):
        return spec
    if not hasattr(spec, 'items'): # lazy check for dict-like behavior
        spec = infer_spec(spec)
    spec = spec.copy()
    cls = spec.pop('type')
    if isinstance(cls, str):
        cls = slots.coordinate_mapping[cls]
    return cls(**spec)

coordinate_type_definitions = {
    "TRIC": {
        "internals": "auto",
        "connect": False,
        "addtr": True,
        "addcart": False
    },
    "DLC": {
        "primitives": "auto",
        "connect": True,
        "addtr": True,
        "addcart": False
    },
    "HDLC": {
        "primitives": "auto",
        "connect": False,
        "addtr": False,
        "addcart": True
    }
}

def construct_coordinate_system(atoms, xyz,
                                bonds='auto',
                                primitives=None,
                                internals=None,
                                coordinate_type=None,
                                **opts
                                ) -> InternalCoordinates:
    if coordinate_type is not None:
        new_opts = coordinate_type_definitions[coordinate_type]
        opts = dict(new_opts, **opts)
        if primitives is None:
            primitives = opts.pop("primitives", None)
        if internals is None:
            internals = opts.pop("internals", None)

    if internals is not None:
        form_topology = dev.str_is(internals, 'auto')
        if form_topology:
            if dev.str_is(bonds, 'auto'):
                bonds = guess_bonds(atoms, xyz)
            elif bonds is None:
                raise ValueError("can't construct automatic internals without bonds")
            internals = None # handled within `PrimitiveInternalCoordinates`, for better or worse
        else:
            internals = [construct_internal(i) for i in internals]
        return PrimitiveInternalCoordinates(
            atoms,
            xyz,
            bonds,
            internals=internals,
            form_topology=form_topology,
            **opts
        )
    elif primitives is not None:
        form_topology = dev.str_is(primitives, 'auto')
        if form_topology:
            if dev.str_is(bonds, 'auto'):
                bonds = guess_bonds(atoms, xyz)
                primitives = None # see above
            elif bonds is None:
                raise ValueError("can't construct automatic primitives without bonds")
        if not hasattr(primitives, "GMatrix"):
            if primitives is not None:
                primitives = [construct_internal(i) for i in primitives]
            primitives = PrimitiveInternalCoordinates(
                atoms,
                xyz,
                bonds,
                internals=primitives,
                **opts
            )
        return DelocalizedInternalCoordinates(atoms, xyz, primitives)
    else:
        return CartesianCoordinates(atoms, xyz, **opts)