from typing import Callable

from . import GSM
from .de_gsm import DE_GSM
from .se_gsm import SE_GSM
from .se_cross import SE_Cross

__all__ = [
    "gsm_types",
    "construct_gsm"
]

gsm_types = {
    "DE_GSM": DE_GSM,
    "SE_GSM": SE_GSM,
    "SE_Cross": SE_Cross,
}
def infer_gsm_type(
        *,
        reactant=None,
        nodes=None,
        product=None,
        **etc
) -> str:
    if nodes is not None:
        if reactant is None:
            reactant = nodes[0]
        if product is None:
            product = nodes[-1]
    if reactant is None and product is None:
        raise ValueError("can't infer GSM type without `reactant`, `product`, or `nodes`")
    #TODO: extend this detection to SE_Cross too
    if product is None:
        return "DE_GSM"
    else:
        return "SE_GSM"
def construct_gsm(mode=None, restart_options=None, **opts) -> GSM:
    if mode is None:
        mode:str = infer_gsm_type(**opts)
    if isinstance(mode, str):
        loader:"type" = gsm_types[mode]
    else:
        mode: "Callable[[], GSM]"
        loader: "Callable[[], GSM]" = mode
    if (
            restart_options is not None
            and restart_options.get('is_restarted', True)
    ):
        return loader.from_restart(**opts)
    else:
        return loader(**opts)
    #
    # def build_GSM_obj(cfg: GSMConfig, reactant, product, driving_coordinates, optimizer):
    #     nifty.printcool("Building the GSM object")
    #     if cfg['mode'] == "DE_GSM":
    #         gsm = DE_GSM.from_options(
    #             reactant=reactant,
    #             product=product,
    #             nnodes=cfg['num_nodes'],
    #             CONV_TOL=cfg['CONV_TOL'],
    #             CONV_gmax=cfg['conv_gmax'],
    #             CONV_Ediff=cfg['conv_Ediff'],
    #             CONV_dE=cfg['conv_dE'],
    #             ADD_NODE_TOL=cfg['ADD_NODE_TOL'],
    #             growth_direction=cfg['growth_direction'],
    #             optimizer=optimizer,
    #             ID=cfg['ID'],
    #             print_level=cfg['gsm_print_level'],
    #             xyz_writer=XYZ_WRITERS[cfg['xyz_output_format']],
    #             mp_cores=cfg["mp_cores"],
    #             interp_method=cfg["interp_method"],
    #         )
    #     else:
    #         if cfg['mode'] == "SE_GSM":
    #             gsm_class = SE_GSM
    #         elif cfg['mode'] == "SE_Cross":
    #             gsm_class = SE_Cross
    #         else:
    #             raise NotImplementedError(f"GSM type: `{cfg['mode']}` not understood")
    #
    #         gsm = gsm_class.from_options(
    #             reactant=reactant,
    #             nnodes=cfg['num_nodes'],
    #             DQMAG_MAX=cfg['DQMAG_MAX'],
    #             BDIST_RATIO=cfg['BDIST_RATIO'],
    #             CONV_TOL=cfg['CONV_TOL'],
    #             ADD_NODE_TOL=cfg['ADD_NODE_TOL'],
    #             optimizer=optimizer,
    #             print_level=cfg['gsm_print_level'],
    #             driving_coords=driving_coordinates,
    #             ID=cfg['ID'],
    #             xyz_writer=XYZ_WRITERS[cfg['xyz_output_format']],
    #             mp_cores=cfg["mp_cores"],
    #             interp_method=cfg["interp_method"],
    #         )
    #     return gsm