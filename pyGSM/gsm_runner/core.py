from aimnet2calc import AIMNet2ASE
import argparse
import importlib
import json
import os
import textwrap
import numpy as np
import glob
from pyGSM.gsm_runner.gsm_config import GSMConfig


from pyGSM.coordinate_systems import DelocalizedInternalCoordinates, Distance, PrimitiveInternalCoordinates, Topology
from pyGSM.growing_string_methods import DE_GSM, SE_Cross, SE_GSM
from pyGSM.level_of_theories.ase import ASELoT
from pyGSM.level_of_theories.xtb_lot import xTB_lot
from pyGSM.optimizers import beales_cg, conjugate_gradient, eigenvector_follow, lbfgs
from pyGSM.potential_energy_surfaces import Avg_PES, PES, Penalty_PES
from pyGSM.utilities import elements, manage_xyz, nifty
from pyGSM.utilities.manage_xyz import XYZ_WRITERS
from pyGSM.molecule import Molecule
from pyGSM.utilities.cli_utils import get_driving_coord_prim, plot


def _initializing(cfg: GSMConfig):

    # XYZ
    if cfg["restart_file"]:
        geoms = manage_xyz.read_molden_geoms(cfg["restart_file"])
    else:
        geoms = manage_xyz.read_xyzs(cfg['xyzfile'])

    # LOT
    nifty.printcool("Build the {} level of theory (LOT) object".format(cfg['EST_Package']))
    lot = create_lot(cfg, geoms[0])

    nifty.printcool("Building the {} objects".format(cfg['PES_type']))
    pes = choose_pes(lot, cfg)

    return geoms, lot, pes

def create_lot(cfg: GSMConfig, geom):
    # decision making
    #cfg['states'] = [(int(m), int(s)) for m, s in zip(cfg["multiplicity"], cfg["adiabatic_index"])]     # doing this with GSMConfig __postinit__
    do_coupling = cfg['PES_type'] == "Avg_PES"
    coupling_states = cfg['states'] if cfg['PES_type'] == "Avg_PES" else []

    # common options for LoTs
    lot_options = dict(
        ID=cfg['ID'],
        lot_inp_file=cfg['lot_inp_file'],
        states=cfg['states'],
        gradient_states=cfg['states'],
        coupling_states=coupling_states,
        geom=geom,
        nproc=cfg["nproc"],
        charge=cfg["charge"],
        do_coupling=do_coupling,
    )

    # actual LoT choice
    lot_name = cfg["EST_Package"]
    if lot_name.lower() == "ase":

        # de-serialise the JSON argument given
        # ase_kwargs = dict(json.loads(cfg.get("ase_kwargs", "{}")))
        calc = AIMNet2ASE('aimnet2',charge=0)
        if cfg['constraints_file'] != None:
            constraints = read_constraints_file(cfg['constraints_file'])
        else:
            constraints = [[None]]
        return ASELoT.from_options(calc, constraints, **lot_options)

    if lot_name == "xTB_lot":
        # raise NotImplementedError("xTB_lot has been disabled temporarily.")
        return xTB_lot.from_options(
            xTB_Hamiltonian=cfg['xTB_Hamiltonian'],
            xTB_accuracy=cfg['xTB_accuracy'],
            xTB_electronic_temperature=cfg['xTB_electronic_temperature'],
            solvent=cfg['solvent'],
            **lot_options,
        )
    else:
        est_package = importlib.import_module("pyGSM.level_of_theories." + lot_name.lower())
        lot_class = getattr(est_package, lot_name)
        return lot_class.from_options(**lot_options)

def choose_pes(lot, cfg: GSMConfig):
    if cfg['PES_type'] == 'PES':
        pes = PES.from_options(
            lot=lot,
            ad_idx=cfg['adiabatic_index'][0],
            multiplicity=cfg['multiplicity'][0],
            # FORCE=cfg['FORCE'],
            # RESTRAINTS=cfg['RESTRAINTS'],
        )
    else:
        pes1 = PES.from_options(
            lot=lot, multiplicity=cfg['states'][0][0],
            ad_idx=cfg['states'][0][1],
            # FORCE=cfg['FORCE'],
            # RESTRAINTS=cfg['RESTRAINTS'],
        )
        pes2 = PES.from_options(
            lot=lot,
            multiplicity=cfg['states'][1][0],
            ad_idx=cfg['states'][1][1],
            # FORCE=cfg['FORCE'],
            # RESTRAINTS=cfg['RESTRAINTS'],
        )
        if cfg['PES_type'] == "Avg_PES":
            pes = Avg_PES(PES1=pes1, PES2=pes2, lot=lot)
        elif cfg['PES_type'] == "Penalty_PES":
            pes = Penalty_PES(PES1=pes1, PES2=pes2, lot=lot, sigma=cfg['sigma'])
        else:
            raise NotImplementedError

    return pes


def choose_optimizer(cfg: GSMConfig):
    update_hess_in_bg = True
    if cfg["only_climb"] or cfg['optimizer'] == "lbfgs":
        update_hess_in_bg = False

    # choose the class
    if cfg['optimizer'] == "conjugate_gradient":
        opt_class = conjugate_gradient
    elif cfg['optimizer'] == "eigenvector_follow":
        opt_class = eigenvector_follow
    elif cfg['optimizer'] == "lbfgs":
        opt_class = lbfgs
    elif cfg['optimizer'] == "beales_cg":
        opt_class = beales_cg
    else:
        raise NotImplementedError(f"Optimizer `{cfg['optimizer']}` not implemented")

    optimizer = opt_class.from_options(
        print_level=cfg['opt_print_level'],
        Linesearch=cfg['linesearch'],
        update_hess_in_bg=update_hess_in_bg,
        conv_Ediff=cfg['conv_Ediff'],
        conv_dE=cfg['conv_dE'],
        conv_gmax=cfg['conv_gmax'],
        DMAX=cfg['DMAX'],
        # opt_climb=True if cfg["only_climb"] else False,
    )

    return optimizer

def setup_topologies(cfg: GSMConfig, geoms = None, pes = None):
    '''
    Returns (reactant: Molecule, product: Molecule, driving_coordinates: list)
    if SE_GSM: returns (reactant, None, driving_coordinates)    
    if DE_GSM, returns (reactant, product, None)
    '''
    if geoms is None or pes is None:
        geoms, lot, pes = _initializing(cfg)

    nifty.printcool("Building the reactant object with {}".format(cfg['coordinate_type']))
    Form_Hessian = True if cfg['optimizer'] == 'eigenvector_follow' else False

    # Build the topology
    nifty.printcool("Building the topology")
    atom_symbols = manage_xyz.get_atoms(geoms[0])
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geoms[0])
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

def read_constraints_file(constraints_file):
    with open(constraints_file) as f:
        tmp = filter(None, (line.rstrip() for line in f))
        lines = []
        for line in tmp:
            lines.append(line)
    constraints = []
    for line in lines:
        idx1 = int(line.split()[0])
        idx2 = int(line.split()[1])
        value = float(line.split()[2])
        constraints.append([idx1, idx2, value])
    return constraints


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
