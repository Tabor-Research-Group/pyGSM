import argparse
import textwrap
from . import gsm_api, gsm_config


# TODO this is old.  check config.GSMConfig for valid params
def parse_arguments(verbose=True):
    parser = argparse.ArgumentParser(
        description="Reaction path transition state and photochemistry tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
                Example of use:
                --------------------------------
                gsm -mode DE_GSM -xyzfile yourfile.xyz -package QChem -lot_inp_file qstart -ID 1
                ''')
    )
    parser.add_argument('-config_file', type=str, help='Path to a configuration file in JSON format', required=False)
    parser.add_argument('-xyzfile', help='XYZ file containing reactant and, if DE-GSM, product.', required=True)
    parser.add_argument('-isomers', help='driving coordinate file', type=str, required=False)
    parser.add_argument('-mode', default="DE_GSM", help='GSM Type (default: %(default)s)',
                        choices=["DE_GSM", "SE_GSM", "SE_Cross"], type=str, required=True)
    parser.add_argument('-only_drive', action='store_true', help='')
    parser.add_argument('-package', default="QChem", type=str,
                        help="Electronic structure theory package (default: %(default)s)",
                        choices=["QChem", "Orca", "Molpro", "PyTC", "TeraChemCloud", "OpenMM", "DFTB", "TeraChem",
                                 "BAGEL", "xTB_lot", "ase"])
    parser.add_argument('-lot_inp_file', type=str, default=None,
                        help='external file to specify calculation e.g. qstart,gstart,etc. Highly package specific.',
                        required=False)
    parser.add_argument('-ID', default=0, type=int, help='string identification number (default: %(default)s)',
                        required=False)
    parser.add_argument('-num_nodes', type=int, default=11,
                        help='number of nodes for string (defaults: 9 DE-GSM, 20 SE-GSM)', required=False)
    parser.add_argument('-pes_type', type=str, default='PES', help='Potential energy surface (default: %(default)s)',
                        choices=['PES', 'Avg_PES', 'Penalty_PES'])
    # parser.add_argument('-adiabatic_index', nargs="*", type=int, default=[0],
    #                     help='Adiabatic index (default: %(default)s)', required=False)
    parser.add_argument('-multiplicity', nargs="*", type=int, default=[1], help='Multiplicity (default: %(default)s)')
    # parser.add_argument('-FORCE_FILE', type=str, default=None,
    #                     help='Constant force between atoms in AU,e.g. [(1,2,0.1214)]. Negative is tensile, positive is compresive')
    # parser.add_argument('-RESTRAINT_FILE', type=str, default=None, help='Harmonic translational restraints')
    parser.add_argument('-optimizer', type=str, default='eigenvector_follow',
                        help='The optimizer object. (default: %(default)s Recommend LBFGS for large molecules >1000 atoms)',
                        required=False)
    parser.add_argument('-opt_print_level', type=int, default=1,
                        help='Printout for optimization. 2 prints everything in opt.', required=False)
    parser.add_argument('-gsm_print_level', type=int, default=1, help='Printout for gsm. 1 prints ?', required=False)
    parser.add_argument('-xTB_Hamiltonian', type=str, default='GFN2-xTB', help='xTB hamiltonian', choices=["GFN2-xTB", "GFN1-xTB"], required=False)
    parser.add_argument('-xTB_accuracy', type=float, default=1.0, help='xTB accuracy', required=False)
    parser.add_argument('-xTB_electronic_temperature', type=float, default=300.0, help='xTB electronic temperature', required=False)
    parser.add_argument('-xyz_output_format', type=str, default="molden", help='Format of the produced XYZ files', required=False)
    parser.add_argument('-solvent', type=str, help='Solvent to use (xTB calculations only)', required=False)
    parser.add_argument('-linesearch', type=str, default='NoLineSearch', help='default: %(default)s',
                        choices=['NoLineSearch', 'backtrack'])
    parser.add_argument('-coordinate_type', type=str, default='TRIC', help='Coordinate system (default %(default)s)',
                        choices=['TRIC', 'DLC', 'HDLC'])
    parser.add_argument('-ADD_NODE_TOL', type=float, default=0.01,
                        help='Convergence tolerance for adding new node (default: %(default)s)', required=False)
    parser.add_argument('-DQMAG_MAX', type=float, default=0.8,
                        help='Maximum step size in single-ended mode (default: %(default)s)', required=False)
    parser.add_argument('-BDIST_RATIO', type=float, default=0.5,
                        help='Reaction completion convergence in SE modes (default: %(default)s)')
    parser.add_argument('-CONV_TOL', type=float, default=0.000001,
                        help='Convergence tolerance for optimizing nodes (default: %(default)s)', required=False)
    parser.add_argument('-growth_direction', type=int, default=0,
                        help='Direction adding new nodes (default: %(default)s)', choices=[0, 1, 2])
    parser.add_argument('-reactant_geom_fixed', action='store_true',
                        help='Fix reactant geometry i.e. do not pre-optimize')
    parser.add_argument('-product_geom_fixed', action='store_true',
                        help='Fix product geometry i.e. do not pre-optimize')
    parser.add_argument('-nproc', type=int, default=1,
                        help='Processors for calculation. Python will detect OMP_NUM_THREADS, only use this if you want to force the number of processors')
    parser.add_argument('-charge', type=int, default=0, help='Total system charge (default: %(default)s)')
    parser.add_argument('-max_gsm_iters', type=int, default=10000,
                        help='The maximum number of GSM cycles (default: %(default)s)')
    parser.add_argument('-max_opt_steps', type=int,
                        help='The maximum number of node optimizations per GSM cycle (defaults: 3 DE-GSM, 20 SE-GSM)')
    parser.add_argument('-only_climb', action='store_true', help="Only use climbing image to optimize TS")
    parser.add_argument('-no_climb', action='store_true', help="Don't climb to the TS")
    # parser.add_argument('-optimize_mesx', action='store_true', help='optimize to the MESX')
    # parser.add_argument('-optimize_meci', action='store_true', help='optimize to the MECI')
    parser.add_argument('-restart_file', help='restart file', type=str)
    parser.add_argument('-mp_cores', type=int, default=1,
                        help="Use python multiprocessing to parallelize jobs on a single compute node. Set OMP_NUM_THREADS, ncpus accordingly.")
    parser.add_argument('-dont_analyze_ICs', action='store_false',
                        help="Don't post-print the internal coordinates primitives and values")  # defaults to true
    parser.add_argument('-hybrid_coord_idx_file', type=str, default=None,
                        help="A filename containing a list of  indices to use in hybrid coordinates. 0-Based indexed")
    # parser.add_argument('-frozen_coord_idx_file', type=str, default=None,
    #                     help="A filename containing a list of  indices to be frozen. 0-Based indexed")
    parser.add_argument('-conv_Ediff', default=100., type=float, help='Energy difference convergence of optimization.')
    parser.add_argument('-conv_dE', default=1., type=float, help='State difference energy convergence')
    parser.add_argument('-conv_gmax', default=100., type=float, help='Max grad rms threshold')
    parser.add_argument('-DMAX', default=.1, type=float, help='')
    parser.add_argument('-sigma', default=1., type=float,
                        help='The strength of the difference energy penalty in Penalty_PES')
    parser.add_argument('-prim_idx_file', type=str,
                        help="A filename containing a list of indices to define fragments. 0-Based indexed")
    parser.add_argument('-reparametrize', action='store_true', help='Reparametrize restart string equally along path')
    parser.add_argument('-interp_method', default='DLC', type=str, help='')
    # parser.add_argument('-bonds_file', type=str, help="A file which contains the bond indices (0-based)")
    parser.add_argument('-start_climb_immediately', action='store_true', help='Start climbing immediately when restarting.')
    parser.add_argument('-constraints_file', type=str, required=False, default=None)

    # ASE calculator's options
    group_ase = parser.add_argument_group('ASE', 'ASE calculator options')
    group_ase.add_argument('--ase-class', type=str,
                           help='ASE calculator import path, eg. "ase.calculators.lj.LennardJones"')
    group_ase.add_argument('--ase-kwargs', type=str, help='ASE calculator keyword args, as JSON dictionary, '
                           'eg. {"param_filename":"path/to/file.xml"}')

    args = parser.parse_args()

    return args

# TODO this is old.  check config.GSMConfig for valid params
def parseargs_to_dict(args) -> dict: 
    inpfileq = {
        # LOT
        'lot_inp_file': args.lot_inp_file,
        'xyzfile': args.xyzfile,
        'EST_Package': args.package,
        'reactant_geom_fixed': args.reactant_geom_fixed,
        'nproc': nproc,
        'states': None,
        'xTB_Hamiltonian': args.xTB_Hamiltonian,
        'xTB_accuracy': args.xTB_accuracy,
        'xTB_electronic_temperature': args.xTB_electronic_temperature,
        'solvent': args.solvent,

        # PES
        'PES_type': args.pes_type,
        'adiabatic_index': args.adiabatic_index,
        'multiplicity': args.multiplicity,
        'charge': args.charge,
        # 'FORCE_FILE': args.FORCE_FILE,
        # 'RESTRAINT_FILE': args.RESTRAINT_FILE,
        # 'FORCE': None,
        # 'RESTRAINTS': None,

        # optimizer
        'optimizer': args.optimizer,
        'opt_print_level': args.opt_print_level,
        'linesearch': args.linesearch,
        'DMAX': args.DMAX,

        #output
        'xyz_output_format': args.xyz_output_format,

        # molecule
        'coordinate_type': args.coordinate_type,
        'hybrid_coord_idx_file': args.hybrid_coord_idx_file,
        'frozen_coord_idx_file': args.frozen_coord_idx_file,
        'prim_idx_file': args.prim_idx_file,

        # GSM
        'gsm_type': args.mode,  # SE_GSM, SE_Cross
        'num_nodes': args.num_nodes,
        'isomers_file': args.isomers,
        'ADD_NODE_TOL': args.ADD_NODE_TOL,
        'CONV_TOL': args.CONV_TOL,
        'conv_Ediff': args.conv_Ediff,
        'conv_dE': args.conv_dE,
        'conv_gmax': args.conv_gmax,
        'BDIST_RATIO': args.BDIST_RATIO,
        'DQMAG_MAX': args.DQMAG_MAX,
        'growth_direction': args.growth_direction,
        'ID': args.ID,
        'product_geom_fixed': args.product_geom_fixed,
        'gsm_print_level': args.gsm_print_level,
        'max_gsm_iters': args.max_gsm_iters,
        'max_opt_steps': args.max_opt_steps,
        # 'use_multiprocessing': args.use_multiprocessing,
        # 'sigma': args.sigma,

        # newly added args that did not live here yet
        'only_climb': args.only_climb,
        'restart_file': args.restart_file,
        'no_climb': args.no_climb,
        # 'optimize_mesx': args.optimize_mesx,
        # 'optimize_meci': args.optimize_meci,
        # 'bonds_file': args.bonds_file,
        'mp_cores': args.mp_cores,
        'interp_method': args.interp_method,
        'only_drive': args.only_drive,
        'reparametrize': args.reparametrize,
        'dont_analyze_ICs': args.dont_analyze_ICs,
        'start_climb_immediately' : args.start_climb_immediately,

        # ASE
        'ase_class': args.ase_class,
        'ase_kwargs': args.ase_kwargs,

        # External Force
        'constraints_file':args.constraints_file,
    }

    # if verbose:
    #     nifty.printcool_dictionary(inpfileq, title='Parsed GSM Keys : Values')

    # set default num_nodes
    # if inpfileq['num_nodes'] is None:
    #     if inpfileq['gsm_type'] == "DE_GSM":
    #         inpfileq['num_nodes'] = 9
    #     else:
    #         inpfileq['num_nodes'] = 20

    # checks on parameters
    # if inpfileq['PES_type'] != "PES":
    #     assert len(inpfileq["adiabatic_index"]) > 1, "need more states"
    #     assert len(inpfileq["multiplicity"]) > 1, "need more spins"
    # if inpfileq["charge"] != 0:
    #     print("Warning: charge is not implemented for all level of theories. "
    #           "Make sure this is correct for your package.")

    return inpfileq

def main():
    args = parse_arguments()
    args_dict = parseargs_to_dict(args)
    if args_dict['config_file'] is not None:
        cfg = gsm_config.GSMConfig.from_json(args_dict['config_file'])
    else:
        cfg = gsm_config.GSMConfig(**args_dict)  # or gsm_config.GSMConfig.from_dict(args_dict)
    run = gsm_api.run_gsm(cfg, verbose=True)

if __name__ == "__main__":
    main()