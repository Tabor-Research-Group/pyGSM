from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict, Any
import json

GSMType = Literal["DE_GSM", "SE_GSM"] #, "SE_Cross"]
PESType = Literal["PES", "Avg_PES", "Penalty_PES"]
CoordType = Literal["TRIC", "DLC", "HDLC"]
LineSearch = Literal["NoLineSearch", "backtrack"]
OptimizerType = Literal["eigenvector_follow","conjugate_gradient","lbfgs","beales_cg"]
PackageType = Literal["QChem", "Orca", "Molpro", 
                      "PyTC", "TeraChemCloud", "OpenMM", 
                      "DFTB", "TeraChem", "BAGEL", 
                      "xTB_lot", "ase"]

# TODO add validation for some fields.  Maybe try using pydantic instead of dataclasses?

@dataclass
class GSMConfig:
    # TODO Add docstrings for each field?  
    # essentials
    xyzfile: Optional[str] = None
    config_file: Optional[str] = None
    mode: GSMType = "DE_GSM"
    EST_Package: PackageType = "ase"
    lot_inp_file: Optional[str] = None  # Package specific?

    # common
    ID: int = 0
    num_nodes: int = None
    PES_type: PESType = "PES"   # Pretty sure the other options won't be used
    adiabatic_index: List[int] = field(default_factory=lambda: [0])
    multiplicity: List[int] = field(default_factory=lambda: [1])
    states: List[int] = None
    charge: int = 0

    # optimizer / stepping
    optimizer: OptimizerType = "eigenvector_follow"
    opt_print_level: int = 1    # 2 Prints everything in opt
    gsm_print_level: int = 1    
    linesearch: LineSearch = "NoLineSearch"
    DMAX: float = 0.1
    ADD_NODE_TOL: float = 0.01
    DQMAG_MAX: float = 0.8      # Controls SE_GSM stepsize when adding nodes
    BDIST_RATIO: float = 0.5
    CONV_TOL: float = 1e-6
    conv_Ediff: float = 100.0
    conv_dE: float = 1.0
    conv_gmax: float = 100.0
    max_gsm_iters: int = 10000
    max_opt_steps: Optional[int] = None

    # run behavior
    growth_direction: int = 0       # Options: 0,1,2
    only_climb: bool = False
    no_climb: bool = False
    reactant_geom_fixed: bool = False
    product_geom_fixed: bool = False
    restart_file: Optional[str] = None
    reparametrize: bool = False
    start_climb_immediately: bool = False
    only_drive: bool = False

    # coords / topology
    coordinate_type: CoordType = "TRIC"
    isomers_file: Optional[str] = None
    # hybrid_coord_idx_file: Optional[str] = None       # Is this used?
    # prim_idx_file: Optional[str] = None               # Is this used?

    # xTB / ASE options
    xTB_Hamiltonian: str = "GFN2-xTB"
    xTB_accuracy: float = 1.0
    xTB_electronic_temperature: float = 300.0
    solvent: Optional[str] = None       #Solvent to use (xTB calculations only)
    xyz_output_format: str = "molden"   # Other options???
    ase_class: Optional[str] = None
    ase_kwargs: Optional[Dict[str, Any]] = None

    # External Force
    constraints_file: Optional[str] = None

    # parallel
    nproc: Optional[int] = 1
    mp_cores: int = 1

    # interpolation
    interp_method: str = "DLC"

    # other stuff
    # dont_analyze_ICs: bool = True   # This doesn't seem used    # TODO probably makes more sense to have "analyze_ICs" and default False
    rtype: Optional[int] = None
    setup_DE_from_SE: bool = False  # If True, will setup DE_GSM from SE_GSM run


    def __post_init__(self):
        self._set_rtype()
        self._set_max_opt_steps()
        self._set_num_nodes()
        self._set_states()
        if self.charge != 0:
            print("Warning: charge is not implemented for all level of theories. "
              "Make sure this is correct for your package.")
        
        # Some basic validation for single-ended GSM
        if self.mode == "SE_GSM":
            if self.isomers_file is None:
                raise ValueError("SE_GSM mode needs an isomers file.")

        # Some basic validation for double-ended GSM
        if self.mode == "DE_GSM":
            # if self.
            pass

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

    def _set_num_nodes(self):
        if self.num_nodes is None:
            if self.mode == "SE_GSM":
                self.num_nodes = 20
            elif self.mode == "DE_GSM":
                self.num_nodes = 9
    def _set_states(self):
        self.states = [(int(m), int(s)) for m, s in zip(self.multiplicity, self.adiabatic_index)]


    # Allow dictionary-like access -- for compatibility with existing code
    def __getitem__(self, key):
        return getattr(self, key)

    @staticmethod
    def from_dict(d: Dict):
        return GSMConfig(**d)
    
    @staticmethod
    def from_json(json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return GSMConfig(**data)
    
    @staticmethod
    def from_file(filepath: str):
        raise NotImplementedError("from_file method not implemented yet.")

    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def help():
        raise NotImplementedError("help method not implemented yet.")
