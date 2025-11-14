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
        from pyGSM.utilities import manage_xyz
        geom = manage_xyz.read_xyzs(test_data('diels_alder.xyz'))[0]
        print(geom)

if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.GeometryTests')