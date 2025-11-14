
from .base_lot import LoT

__all__ = [
    "lot_loaders",
    "construct_lot",
    "register_lot"
]

def _load_ase(**opts):
    from .ase import ASELoT
    return ASELoT(**opts)

def _load_xtb(**opts):
    from .xtb_lot import xTB_lot
    return xTB_lot(**opts)

lot_loaders = {
    "ase": _load_ase,
    "xtb": _load_xtb,
}
def register_lot(lot_name, lot_type):
    lot_loaders[lot_name] = lot_type

def construct_lot(lot_name, **opts):
    if isinstance(lot_name, LoT): return LoT
    if isinstance(lot_name, str):
        loader = lot_loaders.get(lot_name)
        if loader is None:
            raise ValueError(f"unknown level of theory {lot_name}")
        return loader(**opts)
    else:
        raise NotImplementedError("pure level of theory objects from functions not implemented")