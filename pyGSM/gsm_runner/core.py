import os
import numpy as np
import glob
import dataclasses

from ..level_of_theories import construct_lot as _construct_lot
from ..optimizers import construct_optimizer as _construct_optimizer
from ..growing_string_methods import construct_gsm as _construct_gsm
from ..utilities import manage_xyz, nifty, Devutils as dev
from ..molecule import Molecule

from .gsm_config import GSMConfig

__all__ = [
    "load_mols",
    "construct_lot",
    "construct_optimizer",
    "construct_gsm"
]

def load_structures(cfg: GSMConfig):
    mol_opts = cfg.molecule_settings
    geoms = mol_opts.coords
    restart_file = cfg.restart_settings.restart_file
    if geoms is not None:
        if restart_file is not None: raise ValueError("can't have explicit coords and a `restart_file`")
        atoms = mol_opts.atoms
        if mol_opts.atoms is None: raise ValueError("can't have coords without atoms")
    else:
        if mol_opts.xyzfile is not None:
            if restart_file is not None: raise ValueError("can't have an `xyzfile` and a `restart_file`")
            raw_geoms = manage_xyz.read_xyzs(mol_opts.xyzfile)
        elif restart_file is not None:
            raw_geoms = manage_xyz.read_molden_geoms(restart_file)
        else:
            raise ValueError("no `coords`, `xyzfile`, or `restart_file` provided")
        atoms = manage_xyz.get_atoms(raw_geoms[0])
        geoms = np.array([manage_xyz.xyz_to_np(g) for g in raw_geoms])
    return atoms, geoms

def construct_mols(cfg:GSMConfig, *, atoms, coords):
    coords = np.asarray(coords)
    smol = coords.ndim == 2
    if smol: coords = coords[np.newaxis]
    mol_opts = dataclasses.asdict(cfg.molecule_settings)
    coord_opts = dataclasses.asdict(cfg.coordinate_system_settings)
    mol_opts = {
        # structures have been preloaded, we _might_ need to keep this up to date with gsm_config.py
        k:mol_opts[k] for k in mol_opts.keys() - {
            "atoms",
            "coords",
            "xyzfile"
    }}
    cs_base_opts = {
        k:coord_opts.pop(k)
        for k in ['internals', 'primitives', 'bonds', 'coordinate_type']
    }
    mols = [
        Molecule.from_xyz(
            atoms,
            xyz,
            coordinate_system_options=coord_opts,
            **mol_opts,
            **cs_base_opts,
        )
        for xyz in coords
    ]
    if smol:
        mols = mols[0]
    return mols

def load_mols(cfg):
    #TODO: add in support for a `trajectory_file` argument
    #      that can store charges and other potentially relevant information
    atoms, geoms = load_structures(cfg)
    return construct_mols(cfg, atoms=atoms, coords=geoms)

def construct_lot(cfg: GSMConfig, mol):
    lot_opts = dataclasses.asdict(cfg.evaluator_settings)

    package = lot_opts.pop('EST_Package')
    # common options for LoTs
    lot_options = dict(lot_opts,
        atoms=mol.atoms,
        # xyz=mol.xyz,
        charge=mol.charge,
    )

    # actual LoT choice
    return _construct_lot(package, **lot_options)

def construct_optimizer(cfg: GSMConfig):
    opt_settings = dataclasses.asdict(cfg.optimizer_settings)
    only_climb = cfg.runner_settings.only_climb # I think this is in the wrong spot...
    name = opt_settings.pop('optimizer')
    if only_climb:
        opt_settings['update_hess_in_bg'] = False

    return _construct_optimizer(name, **opt_settings)

def construct_gsm(cfg:GSMConfig, *, mols, evaluator, optimizer):
    gsm_opts = dataclasses.asdict(cfg.gsm_settings)
    for o in ['reactant_geom_fixed', 'product_geom_fixed']:
        gsm_opts.pop(o)
    driving_coords = gsm_opts.pop("driving_coords")
    isomers_file = gsm_opts.pop("isomers_file")
    if driving_coords is None:
        driving_coords = isomers_file
    elif isomers_file is not None:
        raise ValueError("got values for both `driving_coords` and `isomers_file`")
    return _construct_gsm(
        optimizer=optimizer,
        nodes=mols,
        evaluator=evaluator,
        restart_options=dataclasses.asdict(cfg.restart_settings),
        tolerances=dataclasses.asdict(cfg.tolerance_settings),
        driving_coords=driving_coords,
        **gsm_opts
    )


def cleanup_scratch(ID):
    directory = './scratch'
    growthiter_files = glob.glob(os.path.join(directory, f'growth_iters_{ID:03d}_*.xyz'))
    for f in growthiter_files:
        os.remove(f)
    optiter_files = glob.glob(os.path.join(directory, f'opt_iters_{ID:03d}_*.xyz'))
    for f in optiter_files:
        os.remove(f)
    # cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(ID)
    # os.system(cmd)
    # cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(ID)
    # os.system(cmd)
    ##cmd = "rm scratch/initial_ic_reparam_{:03d}_{:03d}.xyz".format()
    # if cfg['EST_Package']=="DFTB":
    #    for i in range(self.gsm.nnodes):
    #        cmd = 'rm -rf scratch/{}'.format(i)
    #        os.system(cmd)


def get_nproc():
    # THIS FUNCTION DOES NOT RETURN "USABLE" CPUS
    try:
        return os.cpu_count()
    except (ImportError, NotImplementedError):
        pass
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass
    raise Exception('Can not determine number of CPUs on this system')


def post_processing(gsm, have_TS=True):
    plot(fx=gsm.energies, x=range(len(gsm.energies)), title=gsm.ID)

    ICs = []
    ICs.append(gsm.nodes[0].primitive_internal_coordinates)

    # TS energy
    if have_TS:
        minnodeR = np.argmin(gsm.energies[:gsm.TSnode])
        TSenergy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
        print(" TS energy: %5.4f" % TSenergy)
        print(" absolute energy TS node %5.4f" % gsm.nodes[gsm.TSnode].energy)
        minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode:])
        print(" min reactant node: %i min product node %i TS node is %i" % (minnodeR, minnodeP, gsm.TSnode))

        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[gsm.TSnode].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open('IC_data_{:04d}.txt'.format(gsm.ID), 'w') as f:
            f.write("Internals \t minnodeR: {} \t TSnode: {} \t minnodeP: {}\n".format(minnodeR, gsm.TSnode, minnodeP))
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

    else:
        minnodeR = 0
        minnodeP = gsm.nR
        print(" absolute energy end node %5.4f" % gsm.nodes[gsm.nR].energy)
        print(" difference energy end node %5.4f" % gsm.nodes[gsm.nR].difference_energy)
        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open('IC_data_{}.txt'.format(gsm.ID), 'w') as f:
            f.write("Internals \t Beginning: {} \t End: {}".format(minnodeR, gsm.TSnode, minnodeP))
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\n".format(*x))

    # Delta E
    deltaE = gsm.energies[minnodeP] - gsm.energies[minnodeR]
    print(" Delta E is %5.4f" % deltaE)

def SE_output_to_DE_input():
    try:
        f = open('opt_converged_000.xyz')
    except:
        f = open('grown_string_000.xyz')
        
    Coords = []
    n_of_atoms = []
    Energies = []
    while True:
        line = f.readline()
        if line == '[Geometries] (XYZ)\n':
            tmp_coord = []
            n_of_atoms = f.readline()
            tmp_coord.append(n_of_atoms)
            empty_line = f.readline()
            tmp_coord.append(empty_line)
            for i in range(int(n_of_atoms)):
                coord_line = f.readline()
                tmp_coord.append(coord_line)
            Coords.append(tmp_coord)

        elif line == n_of_atoms:
            tmp_coord = []
            tmp_coord.append(n_of_atoms)
            empty_line = f.readline()
            tmp_coord.append(empty_line)
            for i in range(int(n_of_atoms)):
                coord_line = f.readline()
                tmp_coord.append(coord_line)
            Coords.append(tmp_coord)

        elif line == 'energy\n':
            for i in range(len(Coords)):
                energy = f.readline()
                Energies.append(float(energy))
        elif line == 'max-step\n':
            break

    Energies = np.array(Energies)
    TS_idx = np.argmax(Energies)
    Reactant_idx = np.where(Energies == min(Energies[TS_idx:]))[0][0]
    Product_idx = 0
    Revised_energies = Energies - Energies[Reactant_idx]
    TS_energy = Revised_energies[TS_idx]
    Product_energy = Revised_energies[Product_idx]

    f = open('conf_continue.xyz','w')
    for i in range(len(Coords[Reactant_idx])):
        f.write(Coords[Reactant_idx][i])
    for i in range(len(Coords[Product_idx])):
        f.write(Coords[Product_idx][i])
    f.close()
