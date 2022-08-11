import unittest
import os
from unittest.mock import patch
import shutil
import numpy as np
import json

from pymatgen.core.structure import Structure

from shakenbreak.io import read_espresso_structure, parse_qe_input, parse_fhi_aims_input
from shakenbreak.analysis import _calculate_atomic_disp

# TODO: Add tests for structure parsing functions (castep, cp2k & fhi-aims)
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

    def test_parse_qe_input(self):
        "Test parse_qe_input() function."
        path = os.path.join(self.DATA_DIR, "quantum_espresso/qe.in")
        params = parse_qe_input(path)
        test_params = {
            'CONTROL': {
                'title': "Si bulk",
                'calculation': "relax",
                'restart_mode': "from_scratch",
                'outdir': "Si_example/",
                'pseudo_dir': "/home/ireaml/pw/pseudo/"
            },
            'SYSTEM': {
                'ecutwfc': 18.0,
                'ecutrho': 72.0
            },
            'ELECTRONS': {'conv_thr': '1d-7'},
            'ATOMIC_SPECIES': {'Si': '1.00 Si.vbc'},
        }
        self.assertEqual(
            test_params,
            params
        )

    def test_parse_fhi_aims_input(self):
        "Test parse_fhi_aims_input() function."
        path = os.path.join(self.DATA_DIR, "fhi_aims/control.in")
        params = parse_fhi_aims_input(path)
        test_params = {
            'xc': 'hse',
            'charge': 0.0,
            'spin': 'collinear',
            'default_initial_moment': 0.0,
            'sc_iter_limit': 50.0,
            'relax_geometry': ['bfgs', 0.001]
        }
        self.assertEqual(
            test_params,
            params
        )