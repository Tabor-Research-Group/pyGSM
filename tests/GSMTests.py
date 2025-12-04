import os, sys

import numpy as np

dev_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
root = os.path.join(dev_root, 'pyGSM')
sys.path.insert(0, root)
# os.chdir(root)

import os.path
import unittest
import itertools
import pprint
import tempfile as tf

__all__ = [
    "GSMTests"
]

def test_data(filename):
    return os.path.join(root, "tests", "data", filename)

class GSMTests(unittest.TestCase):

    @unittest.skip
    def test_ConstructMolecules(self):
        from pyGSM.molecule import Molecule
        from pyGSM.coordinate_systems import construct_coordinate_system
        from pyGSM.utilities import manage_xyz
        geom = manage_xyz.read_xyzs(test_data('diels_alder.xyz'))[0]

        # print(geom)
        coords = construct_coordinate_system(
            manage_xyz.get_atoms(geom),
            manage_xyz.xyz_to_np(geom),
            primitives='auto'
        )

        coords.GMatrix(coords.xyz)

        mol = Molecule.from_file(
            test_data('diels_alder.xyz'),
            energy_evaluator="ase",
            energy_evaluator_options={'calculator':'aimnet2ase'},
            coordinate_type='TRIC'
        )

    def test_DEGSM(cls):
        from pyGSM.gsm_runner import GSMRunner
        test_dir = os.path.join(root, "tests", "test_results")
        os.makedirs(test_dir, exist_ok=True)
        os.chdir(test_dir)

        try:
            os.remove('log.txt')
        except OSError:
            pass
        # try:
        #     os.remove('checkpoint.hdf5')
        # except OSError:
        #     pass

        GSMRunner.run_simple(
            xyzfile=test_data('diels_alder.xyz'),
            # EST_Package='aimnet',
            EST_Package='rdkit',
            # logger='log.txt',
            logger=True,
            # logger=dev.Logger(log_level=dev.LogLevel.MoreDebug),
            output_dir='.'
        )

if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.GeometryTests')