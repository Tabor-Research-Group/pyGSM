import os
import dataclasses as dc
import numpy as np

from .gsm_config import GSMConfig
from . import core as core
from .. import growing_string_methods as GSM
from ..molecule import Molecule
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
                 rtype:int|GSM.TSOptimizationStrategy=None,
                 scratch_dir=None):
        self.gsm = gsm
        self.scratch_dir = scratch_dir
        self.max_gsm_iters = max_gsm_iters
        if max_opt_steps is None:
            max_opt_steps = self.gsm.default_max_opt_steps
        self.max_opt_steps = max_opt_steps
        self.rtype = GSM.TSOptimizationStrategy(rtype)

    @classmethod
    def from_config(cls, cfg: GSMConfig, validate=True):
        run_opts = cfg.runner_settings
        logger = dev.Logger.lookup(run_opts.logger)

        mols = core.load_mols(cfg, logger=logger) #TODO: allow direct loading of mols
        if validate:
            for m in mols:
                if m is not None:
                    cls.check_gsm_mol(m)

        optimizer = core.construct_optimizer(cfg, logger=logger)
        evaluator = core.construct_lot(cfg, mols[0], logger=logger)

        gsm = core.construct_gsm(cfg,
                                 mols=mols,
                                 evaluator=evaluator,
                                 optimizer=optimizer,
                                 logger=logger)

        return cls(
            gsm,
            max_gsm_iters=run_opts.max_gsm_iters,
            max_opt_steps=run_opts.max_opt_steps,
            rtype=run_opts.rtype,
            scratch_dir=run_opts.scratch_dir
        )

    @classmethod
    def check_gsm_mol(cls, mol:Molecule):
        if not mol.using_dlcs:
            raise ValueError("GSM methods are only defined for molecules using delocalized internal coordinates")

    @classmethod
    def check_gsm_object(self, gsm):
        for node in gsm.nodes:
            if node is not None:
                self.check_gsm_mol(node)

    def run(self, validate=True):#, max_iters=None, max_opt_steps=None, rtype=None):
        if validate:
            self.check_gsm_object(self.gsm)
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
        return runner.run()


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