import unittest
import os
import pickle
import copy
from unittest.mock import patch
import shutil
import warnings

import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure, Element
from shakenbreak import analysis, champion_defects_rerun, energy_lowering_distortions


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class EnergyLoweringDistortionsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        self.V_Cd_distortion_data = analysis.open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
        )
        self.organized_V_Cd_distortion_data = analysis.organize_data(
            self.V_Cd_distortion_data
        )
        self.V_Cd_distortion_data_no_unperturbed = analysis.open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt")
        )
        self.organized_V_Cd_distortion_data_no_unperturbed = analysis.organize_data(
            self.V_Cd_distortion_data_no_unperturbed
        )
        self.In_Cd_1_distortion_data = analysis.open_file(
            os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt")
        )  # note this was rattled with the old, non-Monte Carlo rattling (ASE's atoms.rattle())
        self.organized_In_Cd_1_distortion_data = analysis.organize_data(
            self.In_Cd_1_distortion_data
        )

    def tearDown(self):
        # removed generated folders
        if_present_rm(os.path.join(self.DATA_DIR, "Int_Cd_2_1"))

    def test_get_deep_distortions(self):
        """Test get_deep_distortions() function"""
        os.mkdir(self.DATA_DIR + "/Int_Cd_2_1")  # without data, to test warnings
        defect_charges_dict = champion_defects_rerun.read_defects_directories(
            self.DATA_DIR
        )
        with patch("builtins.print") as mock_print, warnings.catch_warnings(
            record=True
        ) as w:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            low_energy_defects_dict = energy_lowering_distortions.get_deep_distortions(
                defect_charges_dict, self.DATA_DIR
            )
            mock_print.assert_any_call("\nvac_1_Cd")
            mock_print.assert_any_call(
                "vac_1_Cd_0: Energy difference between minimum, found with -0.55 bond distortion, "
                "and unperturbed: -0.76 eV.\n"
            )
            mock_print.assert_any_call("Deep distortion found for vac_1_Cd_0")
            mock_print.assert_any_call(
                "Energy lowering distortion found for vac_1_Cd with charge "
                "0. Adding to low_energy_defects dictionary."
            )
            mock_print.assert_any_call("\nInt_Cd_2")
            mock_print.assert_any_call(
                f"Path {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt does not exist"
            )
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            warning_message = (
                f"No data parsed from {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt, "
                f"returning None"
            )
            self.assertIn(warning_message, str(w[0].message))

            self.assertEqual(len(low_energy_defects_dict), 1)
            self.assertIn("vac_1_Cd", low_energy_defects_dict)
            self.assertEqual(low_energy_defects_dict["vac_1_Cd"]["charges"], [0])
            np.testing.assert_almost_equal(
                low_energy_defects_dict["vac_1_Cd"]["energy_diff"], -0.7551820700000178
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"]["bond_distortion"], -0.55
            )
            distorted_structure = Structure.from_file(
                self.DATA_DIR + "/vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR"
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"]["structure"], distorted_structure
            )

        with patch("builtins.print") as mock_print:
            low_energy_defects_dict = energy_lowering_distortions.get_deep_distortions(
                defect_charges_dict, self.DATA_DIR, min_e_diff=0.8
            )
            mock_print.assert_any_call("\nvac_1_Cd")
            mock_print.assert_any_call(
                "vac_1_Cd_0: Energy difference between minimum, found with -0.55 bond distortion, "
                "and unperturbed: -0.76 eV.\n"
            )
            mock_print.assert_any_call(
                "No energy lowering distortion with energy difference greater than min_e_diff = 0.80 eV found for vac_1_Cd with charge 0."
            )
            mock_print.assert_any_call("\nInt_Cd_2")
            mock_print.assert_any_call(
                f"Path {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt does not exist"
            )
            mock_print.assert_any_call(
                "No energy lowering distortion with energy difference greater than min_e_diff = "
                "0.80 eV found for Int_Cd_2 with charge 1."
            )
            self.assertEqual(low_energy_defects_dict, {})

        # test stol kwarg, print messages
