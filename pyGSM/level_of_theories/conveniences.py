
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
        constraints_file=None,
        constraints_forces=None,
        **lot_options
):
    from .ase import ASELoT

    # filter out unused option used by only file based LoTs
    for file_opts in ['lot_inp_file']:
        lot_options.pop(file_opts, None)

    if constraints_file is not None:
        if constraints_forces is not None:
            raise ValueError("got both `constraints_file` and `constraints_forces`")
        constraints_forces = ASELoT.read_constraints_file(constraints_file)
    else:
        constraints_forces = [[None]]

    return ASELoT(calculator, constraints_forces=constraints_forces, **lot_options)

@register_lot_loader("aimnet")
def _load_aimnet(*,
                 calculator=None,
                 charge=0,
                 **lot_options
                 ):
    from aimnet2calc import AIMNet2ASE

    if calculator is None:
        calculator = AIMNet2ASE('aimnet2', charge=charge)

    return _load_ase_lot(calculator=calculator, **lot_options)

@register_lot_loader("xtb_lot")
@register_lot_loader("xtb")
def _load_xTb(**lot_options):
    from .xtb_lot import xTB_lot
    return xTB_lot(**lot_options)

def construct_lot(lot_name, allow_imports=True,
                  *,
                  states=None,
                  adiabatic_index=(0,),
                  multiplicity=(1,),
                  lot_options=None,
                  **opts):
    if isinstance(lot_name, LoT): return LoT
    if lot_options is not None:
        opts = dict(lot_options, **opts)

    if states is None:
        states = [(int(m), int(s)) for m, s in zip(multiplicity, adiabatic_index)]
    if isinstance(lot_name, str):
        loader = lot_loader_registry.get(lot_name)
        if loader is None:
            if allow_imports:
                loader = _resolve_lot(lot_name)
            else:
                raise ValueError(f"unknown level of theory {lot_name}")
        return loader(states=states, **opts)
    else:
        raise NotImplementedError("pure level of theory objects from functions not implemented")