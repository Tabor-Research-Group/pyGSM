from dataclasses import dataclass, asdict, field, fields
from typing import List, Optional, Literal, Dict, Any
import json

from ..growing_string_methods import GrowthType

GSMType = Literal["DE_GSM", "SE_GSM"] #, "SE_Cross"]
# PESType = Literal["PES", "Avg_PES", "Penalty_PES"]
CoordType = Literal["TRIC", "DLC", "HDLC"]
LineSearch = Literal["NoLineSearch", "backtrack"]
OptimizerType = Literal["eigenvector_follow","conjugate_gradient","lbfgs","beales_cg"]
PackageType = Literal["aimnet", "xTB_lot", "ase"]
GrowthDirection = Literal[0, 1, 2]
TSOptimizationType = Literal[0, 1, 2]

# TODO add validation for some fields.  Maybe try using pydantic instead of dataclasses?
#      MAB (11/15/2025): we'll let the types themselves do validation
@dataclass
class CoordinateSystemOptions:
    # coords / topology
    coordinate_type: CoordType = "TRIC"
    internals: Literal['auto']|list[tuple[int, int]|tuple[int, int, int]|tuple[int, int, int, int]] = None
    primitives:Literal['auto']|list[tuple[int, int]|tuple[int, int, int]|tuple[int, int, int, int]] = None
    bonds: tuple[int, int]|Literal['auto'] = 'auto'
    hybrid_idx_start_stop: Any = None
    # hybrid_coord_idx_file: Optional[str] = None       # Is this used?
    # prim_idx_file: Optional[str] = None               # Is this used?
@dataclass
class LevelOfTheoryOptions:
    EST_Package: PackageType = "aimnet"
    lot_inp_file: Optional[str] = None  # Package specific?
    lot_options: Dict[str,Any] = None
    adiabatic_index: List[int] = field(default_factory=lambda: [0])
    multiplicity: List[int] = field(default_factory=lambda: [1])
    states: list[tuple[int, int]] = None
    gradient_states: list[tuple[int, int]] = None
    coupling_states: list[tuple[int, int]] = None

    # xTB / ASE options
    # xTB_Hamiltonian: str = "GFN2-xTB"
    # xTB_accuracy: float = 1.0
    # xTB_electronic_temperature: float = 300.0
    # solvent: Optional[str] = None       #Solvent to use (xTB calculations only)
    # xyz_output_format: str = "molden"   # Other options???
    # ase_class: Optional[str] = None
    # ase_kwargs: Optional[Dict[str, Any]] = None

    # External Force
    # constraints_forces = None # need to figure out what type this should be
    # constraints_file: Optional[str] = None

@dataclass
class MoleculeOptions:
    atoms: list[str] = None
    coords: list[list[float]] = None
    charge: int = 0
    xyzfile: str = None

@dataclass
class OptimizerOptions:
    optimizer: OptimizerType = "eigenvector_follow"
    linesearch: LineSearch = "NoLineSearch"
    DMAX: float = 0.1
@dataclass
class GrowingStringMethodOptions:
    mode: GSMType|GrowthType = "DE_GSM"
    num_nodes: int = None

    # run behavior
    growth_direction: GrowthDirection = 0       # Options: 0,1,2
    only_climb: bool = False
    no_climb: bool = False
    reactant_geom_fixed: bool = False
    product_geom_fixed: bool = False
    reparametrize: bool = False
    start_climb_immediately: bool = False
    only_drive: bool = False

    BDIST_RATIO: float = 0.5
    ADD_NODE_TOL: float = 0.01
    DQMAG_MAX: float = 0.8      # Controls SE_GSM stepsize when adding nodes
    CONV_TOL: float = 1e-6
    conv_Ediff: float = 100.0
    conv_dE: float = 1.0
    conv_gmax: float = 100.0

    # interpolation
    interp_method: str = "DLC"

    isomer_specification: Any = None #TODO: provide actual type hints for these
    isomers_file: Optional[str] = None

    # other stuff
    # dont_analyze_ICs: bool = True   # This doesn't seem used    # TODO probably makes more sense to have "analyze_ICs" and default False
    rtype: TSOptimizationType = None
    setup_DE_from_SE: bool = False  # If True, will setup DE_GSM from SE_GSM run

    max_gsm_iters: int = 10000
    max_opt_steps: Optional[int] = None

    def __post_init__(self):
        self._set_mode()
        self._set_num_nodes()
        self._set_rtype()
        self._set_max_opt_steps()

        if self.mode == "SE_GSM" and self.isomers_file is None:
            raise ValueError("SE_GSM mode needs an isomers file.")

    def _set_mode(self):
        self.mode = GrowthType(self.mode)

    def _set_num_nodes(self):
        if self.num_nodes is None:
            if self.mode == "SE_GSM":
                self.num_nodes = 20
            elif self.mode == "DE_GSM":
                self.num_nodes = 9

    def _set_rtype(self):
        if self.rtype is None:
            if self.only_climb:
                self.rtype = 1
            elif self.no_climb:
                self.rtype = 0
            else:
                self.rtype = 2

    def _set_max_opt_steps(self):
        if self.max_opt_steps is None:
            if self.mode == "SE_GSM":
                self.max_opt_steps = 20
            elif self.mode == "DE_GSM":
                self.max_opt_steps = 3

@dataclass
class RunnerSettings:
    ID: int = 0
    logger: bool|str = None

    # parallel
    mp_cores: int = 1

    scratch_dir: str|None = None
    restart_file: Optional[str] = None

# full_option_type_list = (
#     CoordinateSystemOptions,
#     LevelOfTheoryOptions,
#     OptimizerOptions,
#     GrowingStringMethodOptions,
#     RunnerSettings
# )
# def _build_dataclass_field_map(option_type_mapping={},  # add mutable state intentionally
#                                option_types=None):
#     ## don't add confusing 'features'
#     # if option_type_mapping is None: # to build a subversion
#     #     option_type_mapping = {}
#     if option_types is None:
#         option_types = full_option_type_list
#     if len(option_type_mapping) == 0:
#         for cls in option_types:
#             for k in fields(cls):
#                 prev_cls = option_type_mapping.get(k.name)
#                 if prev_cls is not None:
#                     raise ValueError(f"duplicate option '{k.name}' for {cls} and {prev_cls}")
#                 option_type_mapping[k.name] = cls
#     return option_type_mapping
def filter_options(core_class, **opts):
    field_split = {}
    cls_names = {}
    prev = set()
    for base_field in fields(core_class):
        cls = base_field.type
        cls_names[base_field.name] = cls
        field_names = {f.name for f in fields(cls)}
        inter_keys = opts.keys() & field_names
        dupe = prev & inter_keys
        if len(dupe) > 0:
            raise ValueError(f"duplicate keys (implementation bug) {dupe}")
        prev.update(inter_keys)
        field_split[cls] = {k:opts[k] for k in opts.keys() & field_names}

    if len(opts.keys() - prev) > 0:
        raise ValueError(f"got unknown options {list(opts.keys())}")
    return cls_names, field_split

@dataclass
class GSMConfig:
    runner_settings: RunnerSettings
    gsm_settings: GrowingStringMethodOptions
    molecule_settings: MoleculeOptions
    optimizer_settings: OptimizerOptions
    evaluator_settings: LevelOfTheoryOptions
    coordinate_system_settings: CoordinateSystemOptions

    @classmethod
    def from_dict(cls, d: Dict):
        cls_names, options_split = filter_options(cls, **d)
        subopts = {
            k:cls(**options_split[cls]) for k, cls in cls_names.items()
        }
        return GSMConfig(**subopts)
    
    @staticmethod
    def from_json(json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return GSMConfig.from_dict(data)
    
    @staticmethod
    def from_file(filepath: str):
        raise NotImplementedError("from_file method not implemented yet.")

    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def help():
        raise NotImplementedError("help method not implemented yet.")
