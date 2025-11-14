from pyGSM.gsm_runner.gsm_config import GSMConfig
import pyGSM.gsm_runner.core as core
import os
from pyGSM.utilities import nifty, manage_xyz

class GSMResults:
    pass

def run_gsm(cfg: GSMConfig, verbose=True):
    if verbose:
        print_msg()

    # TODO I'm not convinced nproc is doing anything.  Check later.
    if cfg.nproc > 1:
        force_num_procs = True
        if verbose:
            print("forcing number of processors to be {}!!!".format(cfg.nproc))
    else:
        force_num_procs = False
    if force_num_procs:
        nproc = cfg.nproc
    else:
        # nproc = get_nproc()
        try:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        except:
            nproc = 1
        if verbose:
            print(" Using {} processors".format(nproc))
    
    if verbose:
        nifty.printcool_dictionary(cfg.to_dict(), title="GSM Configuration")

    geoms, lot, pes = core._initializing(cfg)

    optimizer = core.choose_optimizer(cfg)

    reactant, product, driving_coordinates = core.setup_topologies(cfg, geoms, pes)

    gsm = core.build_GSM_obj(cfg, reactant, product, driving_coordinates, optimizer)

    if not cfg['reactant_geom_fixed']:
        path = os.path.join(os.getcwd(), 'scratch/{:03}/{}/'.format(cfg["ID"], 0))
        nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
        print(reactant.geometry)
        optimizer.optimize(
            molecule=reactant,
            refE=reactant.energy,
            opt_steps=1000,
            path=path
        )

    if not cfg['product_geom_fixed'] and cfg.mode == 'DE_GSM':
        path = os.path.join(os.getcwd(), 'scratch/{:03}/{}/'.format(cfg["ID"], cfg["num_nodes"] -1))
        nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
        print('product geometry:\n',product.geometry)
        optimizer.optimize(
            molecule=product,
            refE=product.energy,
            opt_steps=1000,
            path=path
        )

    if cfg['restart_file'] is not None:
        gsm.setup_from_geometries(geoms, reparametrize=cfg['reparametrize'], start_climb_immediately=cfg['start_climb_immediately'])

    gsm.go_gsm(cfg['max_gsm_iters'], cfg['max_opt_steps'], cfg['rtype'])

    core.post_processing(gsm,
                    have_TS=True,
                    )
    manage_xyz.write_xyz(f'TSnode_{gsm.ID}.xyz', gsm.nodes[gsm.TSnode].geometry)

    if cfg['mode'] == 'SE_GSM' and cfg['setup_DE_from_SE']:
        core.SE_output_to_DE_input()

    core.cleanup_scratch(gsm.ID)
    
    return 

def print_msg():
    msg = """
    __        __   _                            _        
    \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___  
     \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \ 
      \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
       \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/ 
                                    ____ ____  __  __ 
                       _ __  _   _ / ___/ ___||  \/  |
                      | '_ \| | | | |  _\___ \| |\/| |
                      | |_) | |_| | |_| |___) | |  | |
                      | .__/ \__, |\____|____/|_|  |_|
                      |_|    |___/                    
#==========================================================================#
#| If this code has benefited your research, please support us by citing: |#
#|                                                                        |# 
#| Aldaz, C.; Kammeraad J. A.; Zimmerman P. M. "Discovery of conical      |#
#| intersection mediated photochemistry with growing string methods",     |#
#| Phys. Chem. Chem. Phys., 2018, 20, 27394                               |#
#| http://dx.doi.org/10.1039/c8cp04703k                                   |#
#|                                                                        |# 
#| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
#| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
#| http://dx.doi.org/10.1063/1.4952956                                    |#
#==========================================================================#


    """
    print(msg)

if __name__ == "__main__":
    pass