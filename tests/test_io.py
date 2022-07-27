import unittest
import os
from unittest.mock import patch
import shutil
import numpy as np
import json

from pymatgen.core.structure import Structure

from shakenbreak.io import read_espresso_structure
from shakenbreak.analysis import _calculate_atomic_disp

# TODO: Add tests for structure parsing fuctions (castep, cp2k & fhi-aims)
# For cp2k, castep & fhi-aims, this relies on ase.io so should work fine,
# but sanity check


class IoTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    def test_read_espresso_structure(self):
        """Test `read_espresso_structure` function."""
        structure_from_cif = Structure.from_file(
            os.path.join(self.DATA_DIR, "quantum_espresso/espresso_structure.cif")
        )
        structure_from_espresso_output = read_espresso_structure(
            os.path.join(self.DATA_DIR, "quantum_espresso/vac_1_Cd_0/Bond_Distortion_30.0%/espresso.out")
        )
        self.assertTrue(
            _calculate_atomic_disp(structure_from_cif, structure_from_espresso_output)[
                0
            ]
            < 0.01
        )
