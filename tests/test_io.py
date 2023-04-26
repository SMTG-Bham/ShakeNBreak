import os
import unittest
import warnings

from pymatgen.core.structure import Structure

from shakenbreak.analysis import _calculate_atomic_disp
from shakenbreak.io import (
    parse_fhi_aims_input,
    parse_qe_input,
    read_castep_structure,
    read_cp2k_structure,
    read_espresso_structure,
    read_fhi_aims_structure,
    read_vasp_structure,
)


class IoTestCase(unittest.TestCase):
    """ "Test functions in shakenbreak.io.
    Note that io.parse_energies is tested via test_cli.py"""

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")

    def test_read_vasp_structure(self):
        """Test read_vasp_structure() function."""
        with warnings.catch_warnings(record=True) as w:
            output = read_vasp_structure("fake_file")
            warning_message = (
                "fake_file file doesn't exist, storing as 'Not converged'. Check "
                "path & relaxation"
            )
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))
            self.assertEqual(output, "Not converged")

        with warnings.catch_warnings(record=True) as w:
            output = read_vasp_structure(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_sub_1_In_on_Cd_1.yaml")
            )
            warning_message = (
                f"Problem obtaining structure from: "
                f"{os.path.join(self.VASP_CDTE_DATA_DIR, 'CdTe_sub_1_In_on_Cd_1.yaml')}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))
            self.assertEqual(output, "Not converged")

        with warnings.catch_warnings(record=True) as w:
            output = read_vasp_structure(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_POSCAR")
            )
            V_Cd_struc = Structure.from_file(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_POSCAR")
            )
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 0)
            self.assertEqual(output, V_Cd_struc)

    def test_read_espresso_structure(self):
        """Test `read_espresso_structure` function."""
        structure_from_cif = Structure.from_file(
            os.path.join(self.DATA_DIR, "quantum_espresso/espresso_structure.cif")
        )
        structure_from_espresso_output = read_espresso_structure(
            os.path.join(
                self.DATA_DIR,
                "quantum_espresso/vac_1_Cd_0/Bond_Distortion_30.0%/espresso.out",
            )
        )
        self.assertTrue(
            _calculate_atomic_disp(structure_from_cif, structure_from_espresso_output)[
                0
            ]
            < 0.01
        )

    def test_read_cp2k_structure(self):
        "Test read_cp2k_structure() function."
        path = os.path.join(self.DATA_DIR, "cp2k/cp2k.restart")
        structure = read_cp2k_structure(path)
        test_structure = Structure.from_file(os.path.join(self.DATA_DIR, "cp2k/POSCAR"))
        self.assertEqual(test_structure, structure)

    def test_read_castep_structure(self):
        "Test read_castep_structure() function."
        path = os.path.join(self.DATA_DIR, "castep/Si2-cellopt-aniso.castep")
        structure = read_castep_structure(path)
        test_structure = Structure.from_file(
            os.path.join(self.DATA_DIR, "castep/POSCAR")
        )
        self.assertTrue(_calculate_atomic_disp(test_structure, structure)[0] < 0.01)

    def test_read_fhi_aims_structure(self):
        "Test read_cp2k_structure() function."
        # Test parsing from output file
        path = os.path.join(self.DATA_DIR, "fhi_aims/Si_opt.out")
        structure = read_fhi_aims_structure(path, format="aims-output")
        test_structure = Structure.from_file(
            os.path.join(self.DATA_DIR, "fhi_aims/POSCAR")
        )
        self.assertTrue(_calculate_atomic_disp(test_structure, structure)[0] < 0.01)
        # Test parsing geometry.in file
        path = os.path.join(self.DATA_DIR, "fhi_aims/geometry.in")
        structure = read_fhi_aims_structure(path, format="aims")
        test_structure = Structure.from_file(
            os.path.join(self.DATA_DIR, "fhi_aims/POSCAR_geom")
        )
        self.assertTrue(_calculate_atomic_disp(test_structure, structure)[0] < 0.01)

    def test_parse_qe_input(self):
        "Test parse_qe_input() function."
        path = os.path.join(self.DATA_DIR, "quantum_espresso/qe.in")
        params = parse_qe_input(path)
        test_params = {
            "CONTROL": {
                "title": "Si bulk",
                "calculation": "relax",
                "restart_mode": "from_scratch",
                "outdir": "Si_example/",
                "pseudo_dir": "/home/ireaml/pw/pseudo/",
            },
            "SYSTEM": {"ecutwfc": 18.0, "ecutrho": 72.0},
            "ELECTRONS": {"conv_thr": "1d-7"},
            "ATOMIC_SPECIES": {"Si": "1.00 Si.vbc"},
        }
        self.assertEqual(test_params, params)

    def test_parse_fhi_aims_input(self):
        "Test parse_fhi_aims_input() function."
        path = os.path.join(self.DATA_DIR, "fhi_aims/control.in")
        params = parse_fhi_aims_input(path)
        test_params = {
            "xc": "pbe",
            "charge": 0.0,
            "spin": "collinear",
            "default_initial_moment": 0.0,
            "sc_iter_limit": 100.0,
            "relax_geometry": ["bfgs", 0.001],
        }
        self.assertEqual(test_params, params)
