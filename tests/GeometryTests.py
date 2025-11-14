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
    "GeometryTests"
]

def test_data(filename):
    return os.path.join(root, "tests", "data", filename)

class GeometryTests(unittest.TestCase):

    # @unittest.skip
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

if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.GeometryTests')