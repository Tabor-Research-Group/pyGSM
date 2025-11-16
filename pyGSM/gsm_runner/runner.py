import os
import dataclasses as dc
import numpy as np

from .gsm_config import GSMConfig
from . import core as core
from .. import growing_string_methods as GSM
from ..utilities import nifty, manage_xyz, Devutils as dev

__all__ = [
    "GSMRunner"
]

@dc.dataclass
class GSMResults:
    nodes: list[np.ndarray] = None

class GSMRunner:
    def __init__(self,
                 gsm:GSM.GSM,
                 *,
                 max_gsm_iters:int,
                 max_opt_steps:int=None,
                 rtype:GSM.TSOptimizationStrategy=None,
                 scratch_dir=None):
        self.gsm = gsm
        self.scratch_dir = scratch_dir
        self.max_gsm_iters = max_gsm_iters
        self.max_opt_steps = max_opt_steps
        self.rtype = GSM.TSOptimizationStrategy(rtype)

    @classmethod
    def from_config(cls, cfg: GSMConfig):
        run_opts = cfg.runner_settings
        eval_settings = cfg.evaluator_settings
        gsm_opts = cfg.gsm_settings
        logger = dev.Logger.lookup(run_opts.logger)

        # nifty.printcool_dictionary(cfg.to_dict(), title="GSM Configuration")

        mols = core.load_mols(cfg) #TODO: allow direct loading of mols
        optimizer = core.load_optimizer(cfg)
        evaluator = core.create_lot(cfg, mols[0])

        reactant, product, driving_coordinates = core.setup_topologies(cfg)

        gsm = core.build_GSM_obj(cfg, reactant, product, driving_coordinates, optimizer)

        if cfg['restart_file'] is not None:
            gsm.setup_from_geometries(geoms, reparametrize=cfg['reparametrize'],
                                      start_climb_immediately=cfg['start_climb_immediately'])

        return cls(
            gsm,
            max_gsm_iters=run_opts.max_gsm_iters,
            max_opt_steps=run_opts.max_opt_steps,
            rtype=gsm_opts.rtype,
            scratch_dir=run_opts.scratch_dir
        )

    def run(self):#, max_iters=None, max_opt_steps=None, rtype=None):
        res = self.gsm.go_gsm(
            self.max_gsm_iters,
            self.max_opt_steps,
            rtype=self.rtype
        )

        geoms = [node.xyz for node in self.gsm.nodes]
        return GSMResults(nodes=geoms)

    @classmethod
    def run_simple(cls, **opts):
        runner = cls.from_config(GSMConfig.from_dict(opts))
        return runner


# def run_gsm(cfg: GSMConfig, verbose=True):
#
#     gsm.go_gsm(cfg['max_gsm_iters'], cfg['max_opt_steps'], cfg['rtype'])
#
#     core.post_processing(gsm,
#                     have_TS=True,
#                     )
#     manage_xyz.write_xyz(f'TSnode_{gsm.ID}.xyz', gsm.nodes[gsm.TSnode].geometry)
#
#     if cfg['mode'] == 'SE_GSM' and cfg['setup_DE_from_SE']:
#         core.SE_output_to_DE_input()
#
#     core.cleanup_scratch(gsm.ID)
#
#     return
# def run_simple(*, verbose=True, **kwargs):
#     return run_gsm(
#         GSMConfig(**kwargs),
#         verbose=verbose
#     )

if __name__ == "__main__":
    pass