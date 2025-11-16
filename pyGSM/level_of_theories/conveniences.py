
from .base_lot import LoT
import importlib

__all__ = [
    "lot_loader_registry",
    "construct_lot",
    "register_lot_loader"
]

lot_loader_registry = {}
def register_lot_loader(name):
    def decorate(func):
        lot_loader_registry[name] = func
        return func
    return decorate
def load_lot(name, **opts):
    loader = lot_loader_registry.get(name)
    if loader is None:
        loader = _resolve_lot(name)
    return loader(**opts)

def _resolve_lot(lot_name):
    est_package = importlib.import_module("pyGSM.level_of_theories." + lot_name.lower())
    lot_class = getattr(est_package, lot_name)
    return lot_class

@register_lot_loader("ase")
def _load_ase_lot(
        *,
        calculator,
        **lot_options
):
    from .ase import ASELoT

    return ASELoT(calculator, **lot_options)

@register_lot_loader("aimnet")
def _load_aimnet(*,
                 calculator=None,
                 charge=0,
                 constraints_file=None,
                 **lot_options
                 ):
    from .ase import ASELoT
    from aimnet2calc import AIMNet2ASE

    if calculator is None:
        calculator = AIMNet2ASE('aimnet2', charge=charge)
    if constraints_file is not None:
        constraints = ASELoT.read_constraints_file(constraints_file)
    else:
        constraints = [[None]]

    _load_ase_lot(calculator=calculator, constraints=constraints, **lot_options)

@register_lot_loader("xtb_lot")
@register_lot_loader("xtb")
def _load_xTb(**lot_options):
    from .xtb_lot import xTB_lot
    return xTB_lot(**lot_options)

def construct_lot(lot_name, allow_imports=True, **opts):
    if isinstance(lot_name, LoT): return LoT
    if isinstance(lot_name, str):
        loader = lot_loader_registry.get(lot_name)
        if loader is None:
            if allow_imports:
                loader = _resolve_lot(lot_name)
            else:
                raise ValueError(f"unknown level of theory {lot_name}")
        return loader(**opts)
    else:
        raise NotImplementedError("pure level of theory objects from functions not implemented")