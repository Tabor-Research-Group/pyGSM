import os
import numpy as np
import glob
import dataclasses

from ..coordinate_systems import (
    DelocalizedInternalCoordinates, Distance, PrimitiveInternalCoordinates, construct_coordinate_system
)
from ..level_of_theories import construct_lot
from ..optimizers import construct_optimizer
from ..utilities import manage_xyz, nifty, Devutils as dev
from ..molecule import Molecule

from .gsm_config import GSMConfig

def load_structures(cfg: GSMConfig):
    mol_opts = cfg.molecule_settings
    geoms = mol_opts.coords
    restart_file = cfg.runner_settings.restart_file
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

def create_lot(cfg: GSMConfig, mol):
    lot_opts = cfg.evaluator_settings

    # common options for LoTs
    lot_options = dict(
        atoms=mol.atoms,
        xyz=mol.xyz,
        lot_inp_file=lot_opts.lot_inp_file,
        states=lot_opts.states,
        gradient_states=lot_opts.gradient_states,
        coupling_states=lot_opts.coupling_states,
        nproc=lot_opts.nproc,
        charge=mol.charge,
        lot_options=lot_opts.lot_options
    )

    # actual LoT choice
    return construct_lot(lot_opts.EST_Package, **lot_options)

def load_optimizer(cfg: GSMConfig):
    opt_settings = dataclasses.asdict(cfg.optimizer_settings)
    only_climb = cfg.gsm_settings.only_climb
    name = opt_settings.pop('optimizer')
    if only_climb:
        opt_settings['update_hess_in_bg'] = False

    return construct_optimizer(name, **opt_settings)

def construct_mols(cfg:GSMConfig, *, atoms, coords, evaluator):
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
    mols = [
        Molecule.from_xyz(
            atoms,
            xyz,
            energy_evaluator=evaluator,
            **mol_opts,
            **coord_opts
        )
        for xyz in coords
    ]
    if smol:
        mols = mols[0]
    return mols

def setup_topologies(cfg: GSMConfig, **mol_args):
    '''
    Returns (reactant: Molecule, product: Molecule, driving_coordinates: list)
    if SE_GSM: returns (reactant, None, driving_coordinates)    
    if DE_GSM, returns (reactant, product, None)
    '''

    coord_settings = cfg.coordinate_system_settings
    mols = construct_mols(cfg, **mol_args)
    print(mols)

    raise Exception(...)


    reactant = Molecule.from_xyz()
    nifty.printcool("Building the reactant object with {}".format(coord_settings.coordinate_type))

    mol_1 = Molecule
    Form_Hessian = True if cfg['optimizer'] == 'eigenvector_follow' else False

    # Build the topology
    nifty.printcool("Building the topology")
    xyz1 = coords[0]
    top1 = Topology.build_topology(
        xyz1,
        atoms,
    )
    if cfg['mode'] == 'DE_GSM':
        xyz2 = manage_xyz.xyz_to_np(geoms[-1])
        top2 = Topology.build_topology(
            xyz2,
            atoms,
            # hybrid_indices=hybrid_indices,
            # prim_idx_start_stop=prim_indices,
        )

        # Add bonds to top1 that are present in top2
        # It's not clear if we should form the topology so the bonds
        # are the same since this might affect the Primitives of the xyz1 (slightly)
        # Later we stil need to form the union of bonds, angles and torsions
        # However, I think this is important, the way its formulated, for identifiyin
        # the number of fragments and blocks, which is used in hybrid TRIC.
        for bond in top2.edges():
            if bond in top1.edges:
                pass
            elif (bond[1], bond[0]) in top1.edges():
                pass
            else:
                print(" Adding bond {} to top1".format(bond))
                if bond[0] > bond[1]:
                    top1.add_edge(bond[0], bond[1])
                else:
                    top1.add_edge(bond[1], bond[0])

    if cfg['mode'] == 'SE_GSM' or cfg['mode'] == 'SE_Cross':
        driving_coordinates = read_isomers_file(cfg['isomers_file'])

        driving_coord_prims = []
        for dc in driving_coordinates:
            prim = get_driving_coord_prim(dc)
            if prim is not None:
                driving_coord_prims.append(prim)

        for prim in driving_coord_prims:
            if type(prim) == Distance:
                bond = (prim.atoms[0], prim.atoms[1])
                if bond in top1.edges:
                    pass
                elif (bond[1], bond[0]) in top1.edges():
                    pass
                else:
                    print(" Adding bond {} to top1".format(bond))
                    top1.add_edge(bond[0], bond[1])

    nifty.printcool("Building Primitive Internal Coordinates")
    connect = False
    addtr = False
    addcart = False
    if cfg['coordinate_type'] == "DLC":
        connect = True
    elif cfg['coordinate_type'] == "TRIC":
        addtr = True
    elif cfg['coordinate_type'] == "HDLC":
        addcart = True
    p1 = PrimitiveInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        topology=top1,
    )
    p1.newMakePrimitives(xyz1)
    print(" done making primitives p1")

    if cfg['mode'] == 'DE_GSM':
        nifty.printcool("Building Primitive Internal Coordinates 2")
        p2 = PrimitiveInternalCoordinates.from_options(
            xyz=xyz2,
            atoms=atoms,
            addtr=addtr,
            addcart=addcart,
            connect=connect,
            topology=top1,  # Use the topology of 1 because we fixed it above
        )

        p2.newMakePrimitives(xyz2)
        print(" done making primitives p2")

        nifty.printcool("Forming Union of Primitives")
        # Form the union of primitives
        p1.add_union_primitives(p2)

        print("check {}".format(len(p1.Internals)))

    elif cfg['mode'] == 'SE_GSM' or cfg['mode'] == 'SE_Cross':
        for dc in driving_coord_prims:
            if type(dc) != Distance:  # Already handled in topology
                if dc not in p1.Internals:
                    print("Adding driving coord prim {} to Internals".format(dc))
                    p1.append_prim_to_block(dc)

    nifty.printcool("Building Delocalized Internal Coordinates")
    coord_obj1 = DelocalizedInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=p1,
    )

    nifty.printcool("Building the reactant")
    reactant = Molecule.from_options(
        geom=geoms[0],
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=Form_Hessian,
        # frozen_atoms=frozen_indices,
    )

    # SE_GSM returns here
    if cfg['mode'] == 'SE_GSM' or cfg['mode'] == 'SE_Cross':
        return reactant, None, driving_coordinates

    if cfg['mode'] == 'DE_GSM':
        nifty.printcool("Building the product object")
        xyz2 = manage_xyz.xyz_to_np(geoms[-1])
        product = Molecule.copy_from_options(
            reactant,
            xyz=xyz2,
            new_node_id=cfg['num_nodes'] - 1,
            copy_wavefunction=False,
        )
        return reactant, product, None

def build_GSM_obj(cfg: GSMConfig, reactant, product, driving_coordinates, optimizer):
    nifty.printcool("Building the GSM object")
    if cfg['mode'] == "DE_GSM":
        gsm = DE_GSM.from_options(
            reactant=reactant,
            product=product,
            nnodes=cfg['num_nodes'],
            CONV_TOL=cfg['CONV_TOL'],
            CONV_gmax=cfg['conv_gmax'],
            CONV_Ediff=cfg['conv_Ediff'],
            CONV_dE=cfg['conv_dE'],
            ADD_NODE_TOL=cfg['ADD_NODE_TOL'],
            growth_direction=cfg['growth_direction'],
            optimizer=optimizer,
            ID=cfg['ID'],
            print_level=cfg['gsm_print_level'],
            xyz_writer=XYZ_WRITERS[cfg['xyz_output_format']],
            mp_cores=cfg["mp_cores"],
            interp_method=cfg["interp_method"],
        )
    else:
        if cfg['mode'] == "SE_GSM":
            gsm_class = SE_GSM
        elif cfg['mode'] == "SE_Cross":
            gsm_class = SE_Cross
        else:
            raise NotImplementedError(f"GSM type: `{cfg['mode']}` not understood")

        gsm = gsm_class.from_options(
            reactant=reactant,
            nnodes=cfg['num_nodes'],
            DQMAG_MAX=cfg['DQMAG_MAX'],
            BDIST_RATIO=cfg['BDIST_RATIO'],
            CONV_TOL=cfg['CONV_TOL'],
            ADD_NODE_TOL=cfg['ADD_NODE_TOL'],
            optimizer=optimizer,
            print_level=cfg['gsm_print_level'],
            driving_coords=driving_coordinates,
            ID=cfg['ID'],
            xyz_writer=XYZ_WRITERS[cfg['xyz_output_format']],
            mp_cores=cfg["mp_cores"],
            interp_method=cfg["interp_method"],
        )
    return gsm

def read_isomers_file(isomers_file):
    with open(isomers_file) as f:
        tmp = filter(None, (line.rstrip() for line in f))
        lines = []
        for line in tmp:
            lines.append(line)

    driving_coordinates = []

    if lines[0] == "NEW":
        start = 1
    else:
        start = 0

    for line in lines[start:]:
        dc = []
        twoInts = False
        threeInts = False
        fourInts = False
        for i, elem in enumerate(line.split()):
            if i == 0:
                dc.append(elem)
                if elem == "ADD" or elem == "BREAK":
                    twoInts = True
                elif elem == "ANGLE":
                    threeInts = True
                elif elem == "TORSION" or elem == "OOP":
                    fourInts = True
                elif elem == "ROTATE":
                    threeInts = True
            else:
                if twoInts and i > 2:
                    dc.append(float(elem))
                elif twoInts and i > 3:
                    dc.append(float(elem))  # add break dist
                elif threeInts and i > 3:
                    dc.append(float(elem))
                elif fourInts and i > 4:
                    dc.append(float(elem))
                else:
                    dc.append(int(elem))
        driving_coordinates.append(dc)

    nifty.printcool("driving coordinates {}".format(driving_coordinates))
    return driving_coordinates


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
