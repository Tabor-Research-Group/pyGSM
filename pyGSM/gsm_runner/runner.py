import dataclasses
import os
import dataclasses as dc
import numpy as np
import typing
import warnings

from .gsm_config import GSMConfig
from . import core as core
from .. import growing_string_methods as GSM
from ..molecule import Molecule
from ..utilities import XYZWriter, OutputManager, Devutils as dev

__all__ = [
    "GSMRunner"
]

@dc.dataclass
class GSMResults:
    nodes: list[np.ndarray] = None
    output: typing.Any = None
    gsm: GSM.GSM = None

class GSMRunner:
    def __init__(self,
                 gsm:GSM.GSM,
                 *,
                 max_gsm_iters:int,
                 ID=None,
                 mp_cores=None,
                 max_opt_steps:int=None,
                 rtype:int|GSM.TSOptimizationStrategy=None,
                 setup_DE_from_SE=False):
        self.gsm = gsm
        self.ID = ID
        self.mp_cores = mp_cores
        self.max_gsm_iters = max_gsm_iters
        if max_opt_steps is None:
            max_opt_steps = self.gsm.default_max_opt_steps
        self.max_opt_steps = max_opt_steps
        self.rtype = GSM.TSOptimizationStrategy(rtype)
        self.setup_DE_from_SE = setup_DE_from_SE # unused for now

    @classmethod
    def from_config(cls, cfg: GSMConfig, validate=True):
        run_opts = cfg.runner_settings
        run_dict = dataclasses.asdict(run_opts)
        logger = dev.Logger.lookup(run_dict.pop('logger'))
        for dead_opt in ['only_climb', 'no_climb', 'only_drive']:
            del run_dict[dead_opt]

        mols = core.load_mols(cfg, logger=logger) #TODO: allow direct loading of mols
        if validate:
            for m in mols:
                if m is not None:
                    cls.check_gsm_mol(m)

        optimizer = core.construct_optimizer(cfg, logger=logger)
        evaluator = core.construct_lot(cfg, mols[0], logger=logger)

        xyz_format = run_dict.pop('xyz_format')
        scratch_writer = XYZWriter(OutputManager.lookup(run_dict.pop('scratch_dir')), xyz_format)
        output_writer = XYZWriter(OutputManager.lookup(run_dict.pop('output_dir')), xyz_format)

        gsm = core.construct_gsm(cfg,
                                 mols=mols,
                                 evaluator=evaluator,
                                 optimizer=optimizer,
                                 scratch_writer=scratch_writer,
                                 output_writer=output_writer,
                                 logger=logger)

        return cls(
            gsm,
            **run_dict
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

    def prep_gsm(self, gsm):
        #TODO: don't modify in place
        gsm.mp_cores = self.mp_cores
        return gsm

    def run(self, validate=True):#, max_iters=None, max_opt_steps=None, rtype=None):
        if validate:
            self.check_gsm_object(self.gsm)

        gsm = self.prep_gsm(self.gsm)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = gsm.run(
                self.max_gsm_iters,
                self.max_opt_steps,
                rtype=self.rtype
            )

        geoms = [node.xyz for node in gsm.nodes]
        return GSMResults(nodes=geoms, output=res, gsm=gsm)

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