from pyGSM.gsm_runner.gsm_config import GSMConfig
import pyGSM.gsm_runner.core as core
import os
from pyGSM.utilities import nifty, manage_xyz

class GSMResults:
    pass

def run_gsm(cfg: GSMConfig, verbose=True):

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

if __name__ == "__main__":
    pass