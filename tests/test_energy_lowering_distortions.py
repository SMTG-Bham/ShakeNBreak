import unittest
import os
from unittest.mock import patch
import shutil
import warnings

import numpy as np

from pymatgen.core.structure import Structure, Element
from shakenbreak import analysis, champion_defects_rerun, energy_lowering_distortions


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class EnergyLoweringDistortionsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    def tearDown(self):
        # removed generated folders
        for defect_dir in ["Int_Cd_2_1", "vac_1_Cd_1", "vac_1_Cd_2"]:
            if_present_rm(os.path.join(self.DATA_DIR, defect_dir))

    def test_get_deep_distortions(self):
        """Test get_deep_distortions() function"""
        # create fake distortion folders for testing functionality:
        for defect_dir in ["Int_Cd_2_1", "vac_1_Cd_1", "vac_1_Cd_2"]:
            os.mkdir(self.DATA_DIR + f"/{defect_dir}")
        # Int_Cd_2_1 without data, to test warnings
        V_Cd_1_txt = f"""Bond_Distortion_-7.5%
        -205.700
        Unperturbed
        -205.800"""
        with open(os.path.join(self.DATA_DIR, "vac_1_Cd_1/vac_1_Cd_1.txt"), "w") as fp:
            fp.write(V_Cd_1_txt)
        V_Cd_2_txt = f"""Bond_Distortion_-35.0%
        -206.000
        Unperturbed
        -205.800"""
        with open(os.path.join(self.DATA_DIR, "vac_1_Cd_2/vac_1_Cd_2.txt"), "w") as fp:
            fp.write(V_Cd_2_txt)

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
                "and unperturbed: -0.76 eV."
            )
            mock_print.assert_any_call(
                "Energy lowering distortion found for vac_1_Cd with charge "
                "0. Adding to low_energy_defects dictionary."
            )
            mock_print.assert_any_call(
                "No energy lowering distortion with energy difference greater than min_e_diff = "
                "0.05 eV found for vac_1_Cd with charge 1."
            )
            mock_print.assert_any_call(
                "vac_1_Cd_2: Energy difference between minimum, found with -0.35 bond distortion, "
                "and unperturbed: -0.20 eV."
            )
            mock_print.assert_any_call(
                f"Problem parsing final, low-energy structure for -35.0% bond distortion of "
                f"vac_1_Cd_2 at {self.DATA_DIR}/vac_1_Cd_2/Bond_Distortion_-35.0%/CONTCAR. "
                f"This species will be skipped and will not be included in low_energy_defects ("
                f"check relaxation calculation and folder)."
            )
            mock_print.assert_any_call("\nInt_Cd_2")
            mock_print.assert_any_call(
                f"Path {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt does not exist"
            )
            mock_print.assert_any_call(
                "No data parsed for Int_Cd_2_1. This species will be skipped and will not be "
                "included in the low_energy_defects charge state lists (and so energy lowering "
                "distortions found for other charge states will not be applied for this species)."
            )
            # self.assertNotIn("No energy lowering distortion with energy difference greater than
            # min_e_diff = "
            #   "0.05 eV found for Int_Cd_2 with charge 1.", mock_print.call_args_list ))
            # TODO: need to iterate over / flatten list of lists of list – how?
            # check this is skipped if no data
            mock_print.assert_any_call(
                "\nComparing and pruning defect structures across charge states..."
            )
            mock_print.assert_any_call(
                "Problem parsing structures for vac_1_Cd_1. This species will be skipped and will "
                "not be included in low_energy_defects (check relaxation folders with CONTCARs "
                "are present)."
            )
            self.assertEqual(
                len(w), 28
            )  # No Int_Cd_2_1 data (1), no V_Cd_1 structures (26), all V_Cd_1 'unconverged'
            for warning in w:
                self.assertEqual(warning.category, UserWarning)
            warning_message = (
                f"No data parsed from {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt, "
                f"returning None"
            )
            self.assertIn(warning_message, str(w[0].message))

            self.assertEqual(len(low_energy_defects_dict), 1)
            self.assertIn("vac_1_Cd", low_energy_defects_dict)
            self.assertEqual(len(low_energy_defects_dict["vac_1_Cd"]), 1)
            self.assertEqual(low_energy_defects_dict["vac_1_Cd"][0]["charges"], [0])
            np.testing.assert_almost_equal(
                low_energy_defects_dict["vac_1_Cd"][0]["energy_diffs"],
                [-0.7551820700000178],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][0]["bond_distortions"], [-0.55]
            )
            distorted_structure = Structure.from_file(
                self.DATA_DIR + "/vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR"
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][0]["structures"],
                [distorted_structure],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][0]["excluded_charges"], set()
            )

        # test verbose=False output:
        with patch("builtins.print") as mock_print:
            low_energy_defects_dict = energy_lowering_distortions.get_deep_distortions(
                defect_charges_dict, self.DATA_DIR, verbose=False
            )  # same call as before, just with verbose=False
            # self.assertNotIn("vac_1_Cd_0: Energy difference between minimum, found with -0.55
            # bond distortion, and unperturbed: -0.76 eV." mock_print.call_args_list ))
            # self.assertNotIn("vac_1_Cd_2: Energy difference between minimum, found with -0.35
            # bond distortion, and unperturbed: -0.20 eV."
            # mock_print.call_args_list ))
            # TODO: need to iterate over / flatten list of lists of list – how?
            # check this is skipped if no data

        # test min_e_diff kwarg:
        with patch("builtins.print") as mock_print, warnings.catch_warnings(
            record=True
        ) as w:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            low_energy_defects_dict = energy_lowering_distortions.get_deep_distortions(
                defect_charges_dict, self.DATA_DIR, min_e_diff=0.8
            )
            mock_print.assert_any_call("\nvac_1_Cd")
            mock_print.assert_any_call(
                "vac_1_Cd_0: Energy difference between minimum, found with -0.55 bond distortion, "
                "and unperturbed: -0.76 eV."
            )
            mock_print.assert_any_call(
                "No energy lowering distortion with energy difference greater than min_e_diff = "
                "0.80 eV found for vac_1_Cd with charge 0."
            )
            mock_print.assert_any_call(
                "No energy lowering distortion with energy difference greater than min_e_diff = "
                "0.80 eV found for vac_1_Cd with charge 1."
            )
            mock_print.assert_any_call("\nInt_Cd_2")
            mock_print.assert_any_call(
                f"Path {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt does not exist"
            )
            self.assertEqual(len(w), 1)  # No Int_Cd_2_1 data (1)
            self.assertEqual(warning.category, UserWarning)
            warning_message = (
                f"No data parsed from {self.DATA_DIR}/Int_Cd_2_1/Int_Cd_2_1.txt, "
                f"returning None"
            )
            self.assertIn(warning_message, str(w[0].message))
            self.assertEqual(low_energy_defects_dict, {})

        # test behaviour with two _different_ energy lowering distortions for two different charge
        # states, and thus also testing structure matching routines:
        # use relaxed -20.0% distorted V_Cd structure for all fake V_Cd_1 and V_Cd_2 directories,
        # to test that structure matching should match with V_Cd_0 Unperturbed first (i.e. starts
        # with unperturbed, then rattled, then distortions
        for fake_distortion_dir in ["Bond_Distortion_-7.5%", "Unperturbed"]:
            os.mkdir(f"{self.DATA_DIR}/vac_1_Cd_1/{fake_distortion_dir}")
            shutil.copyfile(
                f"{self.DATA_DIR}/vac_1_Cd_0/Bond_Distortion_-20.0%/CONTCAR",
                f"{self.DATA_DIR}/vac_1_Cd_1/{fake_distortion_dir}/CONTCAR",
            )

        for fake_distortion_dir in ["Bond_Distortion_-35.0%", "Unperturbed"]:
            os.mkdir(f"{self.DATA_DIR}/vac_1_Cd_2/{fake_distortion_dir}")
            shutil.copyfile(
                f"{self.DATA_DIR}/vac_1_Cd_0/Bond_Distortion_-20.0%/CONTCAR",
                f"{self.DATA_DIR}/vac_1_Cd_2/{fake_distortion_dir}/CONTCAR",
            )

        with patch("builtins.print") as mock_print:
            low_energy_defects_dict = energy_lowering_distortions.get_deep_distortions(
                defect_charges_dict, self.DATA_DIR
            )  # same call as before
            # mock_print.assert_any_call(
            #     f"Problem parsing final, low-energy structure for -35.0% bond distortion of "
            #     f"vac_1_Cd_2 at {self.DATA_DIR}/vac_1_Cd_2/Bond_Distortion_-35.0%/CONTCAR. "
            #     f"This species will be skipped and will not be included in low_energy_defects ("
            #     f"check relaxation calculation and folder)."
            # ) TODO: Assert not called!
            mock_print.assert_any_call(
                "Ground-state structure found for vac_1_Cd with charges ["
                "2] has been also previously been found for charge state 0 "
                "(according to structure matching). Adding this charge to "
                "the corresponding entry in low_energy_defects[vac_1_Cd]."
            )
            mock_print.assert_any_call(
                "Ground-state structure found for vac_1_Cd with charges ["
                "2, 0] has been also previously been found for charge "
                "state 1 (according to structure matching). Adding this "
                "charge to the corresponding entry in low_energy_defects["
                "vac_1_Cd]."
            )
            self.assertEqual(len(low_energy_defects_dict), 1)
            self.assertIn("vac_1_Cd", low_energy_defects_dict)
            self.assertEqual(len(low_energy_defects_dict["vac_1_Cd"]), 2)
            self.assertEqual(low_energy_defects_dict["vac_1_Cd"][0]["charges"], [0])
            np.testing.assert_almost_equal(
                low_energy_defects_dict["vac_1_Cd"][0]["energy_diffs"],
                [-0.7551820700000178],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][0]["bond_distortions"], [-0.55]
            )
            distorted_structure = Structure.from_file(
                self.DATA_DIR + "/vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR"
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][0]["structures"],
                [distorted_structure],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][0]["excluded_charges"], {1, 2}
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["charges"], [2, 0, 1]
            )
            np.testing.assert_almost_equal(
                low_energy_defects_dict["vac_1_Cd"][1]["energy_diffs"],
                [-0.2, 0.0, 0.0],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["bond_distortions"],
                [-0.35, "Unperturbed", "Unperturbed"],
            )
            unperturbed_structure = Structure.from_file(
                f"{self.DATA_DIR}/vac_1_Cd_0/Unperturbed/CONTCAR"
            )
            distorted_structure = Structure.from_file(
                f"{self.DATA_DIR}/vac_1_Cd_0/Bond_Distortion_-20.0%/CONTCAR"
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["structures"],
                [distorted_structure, unperturbed_structure, distorted_structure],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["excluded_charges"], set()
            )

        # test case where the _same_ non-spontaneous energy lowering distortion was found for two
        # different charge states
        V_Cd_1_txt_w_distortion = f"""Bond_Distortion_-7.5%
        -206.700
        Unperturbed
        -205.800"""
        with open(os.path.join(self.DATA_DIR, "vac_1_Cd_1/vac_1_Cd_1.txt"), "w") as fp:
            fp.write(V_Cd_1_txt_w_distortion)
        with patch("builtins.print") as mock_print:
            low_energy_defects_dict = energy_lowering_distortions.get_deep_distortions(
                defect_charges_dict, self.DATA_DIR
            )  # same call as before
            mock_print.assert_any_call("Low-energy distorted structure for vac_1_Cd_2 already "
                                       "found with charge states [1], storing together.")
            self.assertEqual(len(low_energy_defects_dict["vac_1_Cd"]), 2)
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["charges"], [1, 2, 0]
            )
            np.testing.assert_almost_equal(
                low_energy_defects_dict["vac_1_Cd"][1]["energy_diffs"],
                [-0.9, -0.2, 0.0],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["bond_distortions"],
                [-0.075, -0.35, "Unperturbed"],
            )
            unperturbed_structure = Structure.from_file(
                f"{self.DATA_DIR}/vac_1_Cd_0/Unperturbed/CONTCAR"
            )
            distorted_structure = Structure.from_file(
                f"{self.DATA_DIR}/vac_1_Cd_0/Bond_Distortion_-20.0%/CONTCAR"
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["structures"],
                [distorted_structure, distorted_structure, unperturbed_structure],
            )
            self.assertEqual(
                low_energy_defects_dict["vac_1_Cd"][1]["excluded_charges"], set()
            )
        # all print messages and potential structure matching outcomes in `get_deep_distortions`
        # have now been tested in the above code

        # test stol kwarg, print messages
