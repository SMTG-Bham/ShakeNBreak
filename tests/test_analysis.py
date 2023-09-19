import os
import re
import shutil
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from monty.serialization import dumpfn, loadfn
from pandas.core.frame import DataFrame
from pymatgen.core.structure import Element, Structure

from shakenbreak import analysis


def if_present_rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class AnalyseDefectsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.VASP_TIO2_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/vac_1_Ti_0")
        self.EXAMPLE_RESULTS = os.path.join(self.DATA_DIR, "example_results")
        self.organized_V_Cd_distortion_data = loadfn(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.yaml")
        )
        self.organized_V_Cd_distortion_data_no_unperturbed = loadfn(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.yaml",
            )
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR"
            )
        )
        self.organized_In_Cd_1_distortion_data = loadfn(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_sub_1_In_on_Cd_1.yaml")
        )  # note this was rattled with the old, non-Monte Carlo rattling (ASE's atoms.rattle())
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR"
            )
        )
        self.V_Cd_minus0pt3_dimer_ground_state = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/Bond_Distortion_-30.0%/CONTCAR"
            )
        )
        self.V_Cd_unperturbed = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/Unperturbed/CONTCAR")
        )

    def tearDown(self):
        # restore the original file (after 'no unperturbed' tests):
        shutil.copy(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.yaml"),
            os.path.join(self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/vac_1_Cd_0.yaml"),
        )
        for i in [
            "fake_file.yaml",
            os.path.join(self.DATA_DIR, "vasp/distortion_metadata.json"),
        ]:
            if_present_rm(i)

        if_present_rm(f"{self.VASP_TIO2_DATA_DIR}/Unperturbed/OUTCAR")
        if_present_rm(f"{self.VASP_TIO2_DATA_DIR}/Bond_Distortion_-40.0%/OUTCAR")

    def copy_v_Ti_OUTCARs(self):
        """
        Copy the OUTCAR files from the `v_Ti_0` `example_results` directory to the `vac_1_Ti_0` `vasp`
        data directory
        """
        shutil.copyfile(
            f"{self.EXAMPLE_RESULTS}/v_Ti_0/Unperturbed/OUTCAR",
            f"{self.VASP_TIO2_DATA_DIR}/Unperturbed/OUTCAR"
        )
        shutil.copyfile(
            f"{self.EXAMPLE_RESULTS}/v_Ti_0/Bond_Distortion_-40.0%/OUTCAR",
            f"{self.VASP_TIO2_DATA_DIR}/Bond_Distortion_-40.0%/OUTCAR",
        )

    def test__format_distortion_names(self):
        self.assertEqual(
            "Unperturbed", analysis._format_distortion_names("Unperturbed")
        )
        self.assertEqual("Rattled", analysis._format_distortion_names("Rattled"))
        self.assertEqual(
            0.3, analysis._format_distortion_names("Bond_Distortion_30.0%")
        )
        self.assertEqual(
            "-20.0%_from_3",
            analysis._format_distortion_names("Bond_Distortion_-20.0%_from_3"),
        )
        self.assertEqual(
            "Rattled_from_-1", analysis._format_distortion_names("Rattled_from_-1")
        )
        self.assertEqual(
            "Label_not_recognized", analysis._format_distortion_names("Wally_McDoodle")
        )

    def test_get_gs_distortion(self):
        """Test get_gs_distortion() function."""
        gs_distortion = analysis.get_gs_distortion(self.organized_V_Cd_distortion_data)
        self.assertEqual(gs_distortion, (-0.7551820700000178, -0.55))

        # test In_Cd_1:
        gs_distortion = analysis.get_gs_distortion(
            self.organized_In_Cd_1_distortion_data
        )
        self.assertEqual(gs_distortion, (-0.006500369999997702, "Rattled"))

        # test with 'Unperturbed' not present:
        gs_distortion_no_unperturbed = analysis.get_gs_distortion(
            self.organized_V_Cd_distortion_data_no_unperturbed
        )
        self.assertEqual(gs_distortion_no_unperturbed, (None, -0.55))

    @patch("builtins.print")
    def test_sort_data(self, mock_print):
        """Test _sort_data() function."""
        # test V_Cd_distortion_data:
        gs_distortion = analysis.get_gs_distortion(self.organized_V_Cd_distortion_data)
        sorted_V_Cd_distortion_data = analysis._sort_data(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.yaml")
        )
        self.assertEqual(
            sorted_V_Cd_distortion_data,
            (self.organized_V_Cd_distortion_data, *gs_distortion),
        )
        mock_print.assert_called_once_with(
            "CdTe_vac_1_Cd_0_stdev_0.25: Energy difference between minimum, "
            + "found with -0.55 bond distortion, and unperturbed: -0.76 eV."
        )

        # test verbose = False
        with patch("builtins.print") as mock_not_verbose_print:
            sorted_V_Cd_distortion_data = analysis._sort_data(
                os.path.join(
                    self.VASP_CDTE_DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.yaml"
                ),
                verbose=False,
            )
            self.assertEqual(
                sorted_V_Cd_distortion_data,
                (self.organized_V_Cd_distortion_data, *gs_distortion),
            )  # check same output returned, just no printing:
            mock_not_verbose_print.assert_not_called()

        # test In_Cd_1:
        gs_distortion = analysis.get_gs_distortion(
            self.organized_In_Cd_1_distortion_data
        )
        with patch("builtins.print") as mock_In_Cd_print:
            sorted_In_Cd_1_distortion_data = analysis._sort_data(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_sub_1_In_on_Cd_1.yaml"),
            )
            mock_In_Cd_print.assert_not_called()
        self.assertEqual(
            sorted_In_Cd_1_distortion_data,
            (self.organized_In_Cd_1_distortion_data, *gs_distortion),
        )

        # test with 'Unperturbed' not present:
        gs_distortion_no_unperturbed = analysis.get_gs_distortion(
            self.organized_V_Cd_distortion_data_no_unperturbed
        )
        with patch("builtins.print") as mock_no_unperturbed_print:
            organized_V_Cd_distortion_data_no_unperturbed = analysis._sort_data(
                os.path.join(
                    self.VASP_CDTE_DATA_DIR,
                    "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.yaml",
                )
            )
            self.assertEqual(
                organized_V_Cd_distortion_data_no_unperturbed,
                (
                    self.organized_V_Cd_distortion_data_no_unperturbed,
                    *gs_distortion_no_unperturbed,
                ),
            )
            mock_no_unperturbed_print.assert_called_once_with(
                f"CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed: Unperturbed energy not found in "
                f"{os.path.join(self.VASP_CDTE_DATA_DIR, 'CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.yaml')}. "
                f"Lowest energy structure found with -0.55 bond distortion."
            )

        # test error catching:
        with warnings.catch_warnings(record=True) as w:
            output = analysis._sort_data("fake_file")
            warning_message = f"Path fake_file does not exist"
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))
            self.assertEqual(output, (None, None, None))

        with warnings.catch_warnings(record=True) as w:
            dumpfn({"Unperturbed": -378.66236832, "distortions": {}}, "fake_file.yaml")
            output = analysis._sort_data("fake_file.yaml")
            warning_message = (
                "No distortion results parsed from fake_file.yaml, returning None"
            )
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))
            self.assertEqual(output, (None, None, None))

    def test_analyse_defect_site(self):
        """Test analyse_defect_site() function."""
        # test V_Cd:
        with patch("builtins.print") as mock_print:
            output = analysis.analyse_defect_site(
                self.V_Cd_minus0pt5_struc_rattled, name="Test pop", vac_site=[0, 0, 0]
            )
            # mock_print.assert_any_call("==> ", "Test pop structural analysis ", " <==")
            mock_print.assert_any_call(
                "\033[1m" + "Test pop structural analysis " + "\033[0m"
            )
            self.assertEqual(
                mock_print.call_args_list[1][0][:2], ("Analysing site", Element("V"))
            )
            np.testing.assert_array_equal(
                mock_print.call_args_list[1][0][2],
                np.array([0, 0, 0]),
            )
            mock_print.assert_any_call(
                "Local order parameters (i.e. resemblance to given "
                "structural motif, via CrystalNN):"
            )
            mock_print.assert_called_with(
                "\nBond-lengths (in \u212B) to nearest neighbours: "
            )

            expected_V_Cd_crystalNN_coord_df = pd.DataFrame(
                {
                    "Coordination": {
                        0: "square co-planar",
                        1: "tetrahedral",
                        2: "rectangular see-saw-like",
                        3: "see-saw-like",
                        4: "trigonal pyramidal",
                    },
                    "Factor": {
                        0: round(0.21944138045080427, 2),
                        1: round(0.8789337233604618, 2),
                        2: round(0.11173135591518825, 2),
                        3: round(0.3876174560001795, 2),
                        4: round(0.3965691083975486, 2),
                    },
                }
            )
            pd.testing.assert_frame_equal(expected_V_Cd_crystalNN_coord_df, output[0])
            expected_V_Cd_crystalNN_bonding_df = pd.DataFrame(
                {
                    "Element": {0: "Te", 1: "Te", 2: "Te", 3: "Te"},
                    "Distance (\u212B)": {0: "1.42", 1: "1.42", 2: "2.62", 3: "3.01"},
                }
            )
            pd.testing.assert_frame_equal(expected_V_Cd_crystalNN_bonding_df, output[1])
            self.assertEqual(len(output), 2)

        # test Int_Cd_2:
        with patch("builtins.print") as mock_print:
            output = analysis.analyse_defect_site(
                self.Int_Cd_2_minus0pt6_NN_10_struc_rattled,
                name="Int_Cd_2",
                site_num=65,
            )
            # mock_print.assert_any_call("==> ", "Int_Cd_2 structural analysis ", " <==")
            mock_print.assert_any_call(
                "\033[1m" + "Int_Cd_2 structural analysis " + "\033[0m"
            )
            self.assertEqual(
                mock_print.call_args_list[1][0][:2], ("Analysing site", Element("Cd"))
            )
            np.testing.assert_array_equal(
                mock_print.call_args_list[1][0][2],
                np.array([0.8125, 0.1875, 0.8125]),
            )

            expected_Int_Cd_2_NN_10_crystalNN_coord_df = pd.DataFrame(
                {
                    "Coordination": {
                        0: "hexagonal planar",
                        1: "octahedral",
                        2: "pentagonal pyramidal",
                    },
                    "Factor": {
                        0: round(0.778391113850372, 2),
                        1: round(0.014891251252011014, 2),
                        2: round(0.058350214482398306, 2),
                    },
                }
            )
            pd.testing.assert_frame_equal(
                expected_Int_Cd_2_NN_10_crystalNN_coord_df, output[0]
            )
            expected_Int_Cd_2_NN_10_crystalNN_bonding_df = pd.DataFrame(
                {
                    "Element": {0: "Cd", 1: "Cd", 2: "Cd", 3: "Te", 4: "Te", 5: "Te"},
                    "Distance (\u212B)": {
                        0: "1.09",
                        1: "1.09",
                        2: "1.09",
                        3: "1.09",
                        4: "1.09",
                        5: "1.09",
                    },
                }
            )
            pd.testing.assert_frame_equal(
                expected_Int_Cd_2_NN_10_crystalNN_bonding_df, output[1]
            )
            self.assertEqual(len(output), 2)

        # test error catching:
        with self.assertRaises(ValueError):
            analysis.analyse_defect_site(
                self.Int_Cd_2_minus0pt6_NN_10_struc_rattled,
                name="Int_Cd_2",
                # no site_num or vac_coords specified
            )

    def test_get_structures(self):
        """Test get_structures() function."""
        # V_Cd_0 with defaults (reading from subdirectories):
        defect_structures_dict = analysis.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.VASP_CDTE_DATA_DIR
        )
        self.assertEqual(len(defect_structures_dict), 26)
        bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
        self.assertEqual(
            set(defect_structures_dict.keys()), set(bond_distortions + ["Unperturbed"])
        )
        relaxed_0pt5_V_Cd_structure = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_-50.0%/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)

        # V_Cd_0 with a defined subset (using `bond_distortions`):
        defect_structures_dict = analysis.get_structures(
            defect_species="vac_1_Cd_0",
            output_path=self.VASP_CDTE_DATA_DIR,
            bond_distortions=[-0.50, -0.25, 0],
        )
        self.assertEqual(
            len(defect_structures_dict), 4
        )  # 3 distortions plus unperturbed
        relaxed_0pt5_V_Cd_structure = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_-50.0%/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)

        # test exception for wrong defect species
        with self.assertRaises(FileNotFoundError) as e:
            wrong_path_error = FileNotFoundError(
                f"Path {self.VASP_CDTE_DATA_DIR}/vac_1_Cd_1 does not exist!"
            )
            analysis.get_structures(
                defect_species="vac_1_Cd_1",
                output_path=self.VASP_CDTE_DATA_DIR,  # wrong defect species
            )
            self.assertIn(wrong_path_error, e.exception)

        # test error catching:
        with self.assertRaises(FileNotFoundError) as e:
            wrong_path_error = FileNotFoundError(
                f"Path wrong_path/vac_1_Cd_0 does not exist!"
            )
            analysis.get_structures(
                defect_species="vac_1_Cd_0", output_path="wrong_path"
            )
            self.assertIn(wrong_path_error, e.exception)

        # test V_Cd_0 with ignoring High_Energy folder
        shutil.copytree(
            os.path.join(self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/Bond_Distortion_-40.0%"),
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/Bond_Distortion_-48.0%_High_Energy"
            ),
        )
        defect_structures_dict = analysis.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.VASP_CDTE_DATA_DIR
        )
        self.assertEqual(len(defect_structures_dict), 26)
        bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
        self.assertEqual(
            set(defect_structures_dict.keys()), set(bond_distortions + ["Unperturbed"])
        )
        relaxed_0pt5_V_Cd_structure = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_-50.0%/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)
        shutil.rmtree(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/Bond_Distortion_-48.0%_High_Energy"
            )
        )

    def test_get_energies(self):
        """Test get_energies() function."""
        # V_Cd_0 with defaults (reading from `vac_1_Cd_0/vac_1_Cd_0.yaml`):
        defect_energies_dict = analysis.get_energies(
            defect_species="vac_1_Cd_0", output_path=self.VASP_CDTE_DATA_DIR
        )
        energies_dict_keys_dict = {"distortions": None, "Unperturbed": None}
        self.assertEqual(defect_energies_dict.keys(), energies_dict_keys_dict.keys())
        bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
        self.assertEqual(
            list(defect_energies_dict["distortions"].keys()), bond_distortions
        )
        # test some specific energies:
        np.testing.assert_almost_equal(
            defect_energies_dict["distortions"][-0.4], -0.7548057600000107
        )
        np.testing.assert_almost_equal(
            defect_energies_dict["distortions"][-0.2], -0.003605090000007749
        )
        self.assertEqual(defect_energies_dict["Unperturbed"], 0)

        # test verbose = False
        with patch("builtins.print") as mock_not_verbose_print:
            defect_energies_dict = analysis.get_energies(
                defect_species="vac_1_Cd_0",
                output_path=self.VASP_CDTE_DATA_DIR,
                verbose=False,
            )
            # no print called:
            mock_not_verbose_print.assert_not_called()
            # check same outputs:
            self.assertEqual(
                defect_energies_dict.keys(), energies_dict_keys_dict.keys()
            )
            bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
            self.assertEqual(
                list(defect_energies_dict["distortions"].keys()), bond_distortions
            )
            # test a specific energy:
            np.testing.assert_almost_equal(
                defect_energies_dict["distortions"][-0.4], -0.7548057600000107
            )

        # V_Cd_0 with meV (reading from `vac_1_Cd_0/vac_1_Cd_0.yaml`):
        defect_energies_meV_dict = analysis.get_energies(
            defect_species="vac_1_Cd_0",
            output_path=self.VASP_CDTE_DATA_DIR,
            units="meV",
        )
        self.assertEqual(
            defect_energies_meV_dict.keys(), energies_dict_keys_dict.keys()
        )
        self.assertEqual(
            list(defect_energies_meV_dict["distortions"].keys()), bond_distortions
        )
        # test some specific energies:
        np.testing.assert_almost_equal(
            defect_energies_meV_dict["distortions"][-0.4], -754.8057600000107
        )
        np.testing.assert_almost_equal(
            defect_energies_meV_dict["distortions"][-0.2], -3.605090000007749
        )
        self.assertEqual(defect_energies_meV_dict["Unperturbed"], 0)

        # test if 'Unperturbed' is not present:
        shutil.copy(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.yaml",
            ),
            os.path.join(self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/vac_1_Cd_0.yaml"),
        )
        # Note we copy back to original in self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            warning_message = (
                "Unperturbed defect energy not found in energies file. Energies will be given "
                "relative to the lowest energy defect structure found."
            )
            defect_energies_dict = analysis.get_energies(
                defect_species="vac_1_Cd_0", output_path=self.VASP_CDTE_DATA_DIR
            )
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))

            energies_dict_keys_dict = {"distortions": None}
            self.assertEqual(
                defect_energies_dict.keys(), energies_dict_keys_dict.keys()
            )
            bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
            self.assertEqual(
                list(defect_energies_dict["distortions"].keys()), bond_distortions
            )
            # test some specific energies:
            np.testing.assert_almost_equal(
                defect_energies_dict["distortions"][-0.4], 0.00037631000000715176
            )
            np.testing.assert_almost_equal(
                defect_energies_dict["distortions"][-0.2], 0.75157698000001
            )
            self.assertFalse("Unperturbed" in defect_energies_dict)

    def test_calculate_struct_comparison(self):
        """Test calculate_struct_comparison() function."""
        # V_Cd_0 with defaults (reading from `vac_1_Cd_0` and `distortion_metadata.json`):
        defect_structures_dict = analysis.get_structures(
            defect_species="v_Cd_s0_0", output_path=self.EXAMPLE_RESULTS
        )
        with patch("builtins.print") as mock_print:
            max_dist_dict = analysis.calculate_struct_comparison(
                defect_structures_dict, verbose=True
            )
            mock_print.assert_called_with("Comparing structures to Unperturbed...")
        self.assertEqual(
            len(max_dist_dict), len(defect_structures_dict)
        )  # one for each
        self.assertEqual(
            max_dist_dict.keys(), defect_structures_dict.keys()
        )  # one for each
        np.testing.assert_almost_equal(max_dist_dict[-0.4], 0.8082011457587672)
        np.testing.assert_almost_equal(max_dist_dict[-0.2], 0.02518600797944396)
        np.testing.assert_almost_equal(
            max_dist_dict["Unperturbed"], 1.7500286730158273e-15
        )

        # V_Cd_0 with 'disp' (reading from `vac_1_Cd_0` and `distortion_metadata.json`):
        disp_dict = analysis.calculate_struct_comparison(defect_structures_dict, "disp")
        self.assertEqual(len(disp_dict), len(defect_structures_dict))  # one for each
        self.assertEqual(
            disp_dict.keys(), defect_structures_dict.keys()
        )  # one for each
        np.testing.assert_almost_equal(disp_dict[-0.4], 5.760478611114056)
        np.testing.assert_almost_equal(
            disp_dict[-0.2], 0.0
        )  # no displacements above threshold
        np.testing.assert_almost_equal(disp_dict["Unperturbed"], 0.0)

        # test with specified ref_structure as dict key:
        with patch("builtins.print") as mock_print:
            disp_dict = analysis.calculate_struct_comparison(
                defect_structures_dict, "disp", ref_structure=-0.4, verbose=True
            )
            mock_print.assert_called_with(
                "Comparing structures to -40.0% bond distorted structure..."
            )
        # spot check:
        self.assertEqual(round(disp_dict[-0.2], 3), 5.75)
        np.testing.assert_almost_equal(disp_dict[-0.4], 0)
        self.assertEqual(round(disp_dict["Unperturbed"], 3), 5.76)

        # test with specified ref_structure as Structure object:
        with patch("builtins.print") as mock_print:
            disp_dict = analysis.calculate_struct_comparison(
                defect_structures_dict,
                "disp",
                ref_structure=self.V_Cd_minus0pt5_struc_rattled,
                verbose=True,
            )
            mock_print.assert_called_with(
                "Comparing structures to specified ref_structure (Cd31 Te32)..."
            )
        # spot check:
        self.assertEqual(round(disp_dict[-0.2], 3), 25.851)
        self.assertEqual(round(disp_dict[-0.4], 3), 26.247)
        self.assertEqual(round(disp_dict["Unperturbed"], 3), 25.874)

        # test kwargs:
        max_dist_dict = analysis.calculate_struct_comparison(
            defect_structures_dict, "max_dist", stol=0.01
        )
        # spot check:
        self.assertEqual(round(max_dist_dict[-0.2], 3), 0.025)
        self.assertIsNone(max_dist_dict[-0.4])
        np.testing.assert_almost_equal(max_dist_dict["Unperturbed"], 0)

        disp_dict = analysis.calculate_struct_comparison(
            defect_structures_dict, "disp", stol=0.01, min_dist=0.01
        )
        # spot check:
        self.assertEqual(round(disp_dict[-0.2], 3), 0.121)
        self.assertIsNone(disp_dict[-0.4])
        self.assertTrue(np.isclose(disp_dict["Unperturbed"], 0))

        # test error catching:
        with self.assertRaises(KeyError) as e:
            wrong_key_error = KeyError(
                "Reference structure key 'Test pop' not found in defect_structures_dict."
            )
            analysis.calculate_struct_comparison(
                defect_structures_dict, ref_structure="Test pop"
            )
            self.assertIn(wrong_key_error, e.exception)

        with self.assertRaises(ValueError) as e:
            unconverged_error = ValueError(
                "Specified reference structure (Unperturbed) is not converged and "
                "cannot be used for structural comparison. Check structures or specify a "
                "different reference structure (ref_structure)."
            )
            unconverged_structures_dict = defect_structures_dict.copy()
            unconverged_structures_dict["Unperturbed"] = "Not converged"
            analysis.calculate_struct_comparison(
                unconverged_structures_dict,
            )
            self.assertIn(unconverged_error, e.exception)

        with self.assertRaises(TypeError) as e:
            wrong_type_error = TypeError(
                "ref_structure must be either a key from defect_structures_dict or a pymatgen "
                "Structure object. Got <class 'int'> instead."
            )
            analysis.calculate_struct_comparison(
                defect_structures_dict, ref_structure=1
            )
            self.assertIn(wrong_type_error, e.exception)

        with self.assertRaises(ValueError) as e:
            wrong_metric_error = ValueError(
                f"Invalid metric 'metwhat'. Must be one of 'disp' or 'max_dist'."
            )  # https://youtu.be/DmH1prySUpA
            analysis.calculate_struct_comparison(
                defect_structures_dict, metric="metwhat"
            )
            self.assertIn(wrong_metric_error, e.exception)

    def test_compare_structures(self):
        """Test compare_structures() function."""
        # V_Cd_0 with defaults (reading from `v_Cd_0` and `distortion_metadata.json`):
        defect_structures_dict = analysis.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.VASP_CDTE_DATA_DIR
        )
        defect_energies_dict = analysis.get_energies(
            defect_species="vac_1_Cd_0", output_path=self.VASP_CDTE_DATA_DIR
        )
        with patch("builtins.print") as mock_print:
            struct_comparison_df = analysis.compare_structures(
                defect_structures_dict, defect_energies_dict
            )
            mock_print.assert_called_with("Comparing structures to Unperturbed...")
        self.assertIsInstance(struct_comparison_df, pd.DataFrame)
        self.assertEqual(len(struct_comparison_df), len(defect_structures_dict))
        self.assertEqual(
            struct_comparison_df.columns.to_list(),
            [
                "Bond Distortion",
                "\u03A3{Displacements} (\u212B)",  # Sigma and Angstrom
                "Max Distance (\u212B)",  # Angstrom
                f"\u0394 Energy (eV)",  # Delta
            ],
        )
        self.assertEqual(
            set(struct_comparison_df["Bond Distortion"].to_list()),
            set(defect_structures_dict.keys()),
        )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 0.0, 0.025, 0.0]
        )
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 5.760, 0.808, -0.75]
        )
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 0.000, 0.000, 0.00]
        )

        # test with specified ref_structure as dict key:
        with patch("builtins.print") as mock_print:
            struct_comparison_df = analysis.compare_structures(
                defect_structures_dict, defect_energies_dict, ref_structure=-0.4
            )
            mock_print.assert_called_with(
                "Comparing structures to -40.0% bond distorted " "structure..."
            )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 5.75, 0.801, 0.0]
        )
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 0.0, 0.0, -0.75]
        )
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 5.76, 0.808, 0.0]
        )

        # test with specified ref_structure as Structure object:
        with patch("builtins.print") as mock_print:
            struct_comparison_df = analysis.compare_structures(
                defect_structures_dict,
                defect_energies_dict,
                ref_structure=self.V_Cd_minus0pt5_struc_rattled,
            )
            mock_print.assert_called_with(
                "Comparing structures to specified ref_structure (Cd31 Te32)..."
            )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 25.851, 1.29, 0.0]
        )
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 26.247, 0.999, -0.75]
        )
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 25.874, 1.314, 0.0]
        )

        # test kwargs:
        with warnings.catch_warnings(record=True) as w:
            struct_comparison_df = analysis.compare_structures(
                defect_structures_dict,
                defect_energies_dict,
                stol=0.01,
                units="meV",
                min_dist=0.01,
            )
            # spot check:
            self.assertEqual(
                struct_comparison_df.iloc[16].to_list(), [-0.2, 0.121, 0.025, 0.0]
            )
            # When a too tight stol is used, check the code retries with larger stol
            warning_message = (
                f"The specified tolerance {0.01} seems to be too tight as"
                " too many lattices could not be matched. Will retry with"
                f" larger tolerance ({0.01+0.4})."
            )
            # self.assertEqual(w[-1].category, UserWarning)
            self.assertTrue(
                any([warning_message in str(warning.message) for warning in w])
            )
            self.assertEqual(
                struct_comparison_df.iloc[8].to_list(), [-0.4, 8.31, 0.808, -0.75]
            )
            self.assertEqual(
                struct_comparison_df.iloc[-1].to_list(),
                ["Unperturbed", 0.000, 0.000, 0.00],
            )
            self.assertEqual(
                struct_comparison_df.columns.to_list(),
                [
                    "Bond Distortion",
                    "\u03A3{Displacements} (\u212B)",  # Sigma and Angstrom
                    "Max Distance (\u212B)",  # Angstrom
                    f"\u0394 Energy (meV)",  # Delta
                ],
            )

        # test comparing structures with specified ref_structure and no unperturbed:
        defect_structures_dict_no_unperturbed = defect_structures_dict.copy()
        defect_structures_dict_no_unperturbed.pop("Unperturbed")
        defect_energies_dict_no_unperturbed = defect_energies_dict.copy()
        defect_energies_dict_no_unperturbed.pop("Unperturbed")
        struct_comparison_df = analysis.compare_structures(
            defect_structures_dict_no_unperturbed,
            defect_energies_dict_no_unperturbed,
            ref_structure=-0.2,
        )
        # spot check:
        self.assertEqual(struct_comparison_df.iloc[16].to_list(), [-0.2, 0.0, 0.0, 0.0])
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 5.75, 0.801, -0.75]
        )
        self.assertTrue(
            "Unperturbed" not in struct_comparison_df["Bond Distortion"].to_list()
        )

        # test error catching:
        with self.assertRaises(KeyError) as e:
            wrong_key_error = KeyError(
                "Reference structure key 'Test pop' not found in defect_structures_dict."
            )
            analysis.compare_structures(
                defect_structures_dict, defect_energies_dict, ref_structure="Test pop"
            )
            self.assertIn(wrong_key_error, e.exception)

        with self.assertRaises(ValueError) as e:
            unconverged_error = ValueError(
                "Specified reference structure (Unperturbed) is not converged and cannot be used "
                "for structural comparison."
            )
            unconverged_structures_dict = defect_structures_dict.copy()
            unconverged_structures_dict["Unperturbed"] = "Not converged"
            analysis.compare_structures(
                unconverged_structures_dict, defect_energies_dict
            )
            self.assertIn(unconverged_error, e.exception)

        with self.assertRaises(TypeError) as e:
            wrong_type_error = TypeError(
                "ref_structure must be either a key from defect_structures_dict or a pymatgen "
                "Structure object. Got <class 'int'> instead."
            )
            analysis.compare_structures(
                defect_structures_dict, defect_energies_dict, ref_structure=1
            )
            self.assertIn(wrong_type_error, e.exception)

        # test 'all structures are not converged' warning:
        unconverged_structures_dict = defect_structures_dict.copy()
        for distortion_key in unconverged_structures_dict:
            unconverged_structures_dict[distortion_key] = "Not converged"
        with warnings.catch_warnings(record=True) as w:
            output = analysis.compare_structures(
                unconverged_structures_dict, defect_energies_dict
            )

            warning_message = (
                "All structures in defect_structures_dict are not converged. "
                "Returning None."
            )
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))
            self.assertEqual(output, None)

    @patch("builtins.print")
    def test_get_homoionic_bonds(self, mock_print):
        """Test get_homoionic_bonds() function"""
        with patch("builtins.print") as mock_print:
            bonds = analysis.get_homoionic_bonds(
                structure=self.V_Cd_minus0pt3_dimer_ground_state,
                element="Te",
                radius=2.9,
                verbose=False,
            )
            self.assertEqual(bonds, {"Te(32)": {"Te(41)": "2.75 A"}})
            mock_print.assert_not_called()

        with patch("builtins.print") as mock_unperturbed_print:
            bonds = analysis.get_homoionic_bonds(
                structure=self.V_Cd_unperturbed,
                element="Te",
                radius=2.9,
                verbose=True,
            )
            self.assertEqual(bonds, {})
            mock_unperturbed_print.assert_called_once_with(
                "No homoionic bonds found with a search radius of 2.9 A"
            )

        with warnings.catch_warnings(record=True) as w:
            bonds = analysis.get_homoionic_bonds(
                structure=self.V_Cd_minus0pt3_dimer_ground_state,
                element="Na",
                radius=2.9,
                verbose=False,
            )
            self.assertEqual(
                str(w[-1].message), "Your structure does not contain element Na!"
            )

    def test_get_site_magnetizations(self):
        """Test get_site_magnetizations() function"""
        self.copy_v_Ti_OUTCARs()
        # Non existent defect folder
        self.assertRaises(
            FileNotFoundError,
            analysis.get_site_magnetizations,
            defect_species="vac_1_Ti_-1",
            output_path=os.path.join(self.DATA_DIR, "vasp"),
            distortions=["Unperturbed", -0.4],
        )

        # User gives defect_site and threshold
        with patch("builtins.print") as mock_print:
            mags = analysis.get_site_magnetizations(
                defect_species="vac_1_Ti_0",
                defect_site=[0.0, 0.16666666666666669, 0.25],
                output_path=os.path.join(self.DATA_DIR, "vasp"),
                distortions=["Unperturbed", -0.4],
                threshold=0.3,
                orbital_projections=False,
                verbose=True,
            )
            mock_print.assert_any_call(
                "Analysing distortion Unperturbed. Total magnetization: 4.0"
            )
            mock_print.assert_any_call(
                "Analysing distortion -0.4. Total magnetization: -0.0"
            )
            mock_print.assert_any_call(
                "No significant magnetizations found for distortion: -0.4 \n"
            )

            pd.testing.assert_frame_equal(
                mags["Unperturbed"],
                DataFrame(
                    {
                        "Site": {
                            "O(35)": "O(35)",
                            "O(53)": "O(53)",
                            "O(62)": "O(62)",
                            "O(68)": "O(68)",
                        },
                        "Frac coords": {
                            "O(35)": [0.0, 0.167, 0.014],
                            "O(53)": [-0.0, 0.167, 0.486],
                            "O(62)": [0.165, 0.167, 0.292],
                            "O(68)": [0.835, 0.167, 0.292],
                        },
                        "Site mag": {
                            "O(35)": 1.458,
                            "O(53)": 1.478,
                            "O(62)": 1.522,
                            "O(68)": 1.521,
                        },
                        "Dist. (\u212B)": {
                            "O(35)": 2.26,
                            "O(53)": 2.26,
                            "O(62)": 1.91,
                            "O(68)": 1.91,
                        },
                    }
                ),
            )

        # Without defect site and with orbital projections
        with warnings.catch_warnings(record=True) as w:
            # copy distortion_metadata.json without TiO2 data into folder, to check warning
            shutil.copyfile(
                os.path.join(self.VASP_CDTE_DATA_DIR, "distortion_metadata.json"),
                os.path.join(self.DATA_DIR, "vasp/distortion_metadata.json"),
            )
            mags = analysis.get_site_magnetizations(
                defect_species="vac_1_Ti_0",
                output_path=os.path.join(self.DATA_DIR, "vasp"),
                distortions=["Unperturbed"],
                threshold=0.3,
                orbital_projections=True,
            )
            self.assertEqual(
                str(w[-1].message),
                "Could not find defect vac_1_Ti_0 in distortion_metadata.json file. Will not "
                "include distance between defect and sites with significant magnetization.",
            )
            pd.testing.assert_frame_equal(
                mags["Unperturbed"],
                DataFrame(
                    {
                        "Site": {
                            "O(35)": "O(35)",
                            "O(53)": "O(53)",
                            "O(62)": "O(62)",
                            "O(68)": "O(68)",
                        },
                        "Frac coords": {
                            "O(35)": [0.0, 0.167, 0.014],
                            "O(53)": [-0.0, 0.167, 0.486],
                            "O(62)": [0.165, 0.167, 0.292],
                            "O(68)": [0.835, 0.167, 0.292],
                        },
                        "Site mag": {
                            "O(35)": 1.458,
                            "O(53)": 1.478,
                            "O(62)": 1.522,
                            "O(68)": 1.521,
                        },
                        "s": {
                            "O(35)": 0.012,
                            "O(53)": 0.013,
                            "O(62)": 0.013,
                            "O(68)": 0.013,
                        },
                        "p": {
                            "O(35)": 0.717,
                            "O(53)": 0.726,
                            "O(62)": 0.748,
                            "O(68)": 0.747,
                        },
                        "d": {"O(35)": 0.0, "O(53)": 0.0, "O(62)": 0.0, "O(68)": 0.0},
                    }
                ),
            )

        # Non existent structure
        self.copy_v_Ti_OUTCARs()
        os.mkdir(f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%")
        shutil.copyfile(
            f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_-40.0%/OUTCAR",
            f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%/OUTCAR",
        )
        with warnings.catch_warnings(record=True) as w:
            mags = analysis.get_site_magnetizations(
                defect_species="vac_1_Ti_0",
                output_path=os.path.join(self.DATA_DIR, "vasp"),
                distortions=[
                    0.2,
                ],  # no CONTCAR in this distortion folder
                threshold=0.3,
                defect_site=[0.0, 0.16666666666666669, 0.25],
            )
        self.assertTrue(
            any(
                "Structure for vac_1_Ti_0 either not converged or not found. Skipping "
                "magnetisation analysis." in str(warning.message)
                for warning in w
            )
        )

        # ISPIN = 1 OUTCAR:
        shutil.copyfile(
            f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_-40.0%/CONTCAR",
            f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%/CONTCAR",
        )
        with open(f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%/OUTCAR") as f:
            outcar_string = f.read()
        ispin1_outcar_string = re.sub(
            "ISPIN  =      2    spin polarized calculation?", "ISPIN = 1", outcar_string
        )
        with open(
            f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%/OUTCAR", "w"
        ) as f:
            f.write(ispin1_outcar_string)
        with warnings.catch_warnings(record=True) as w:
            mags = analysis.get_site_magnetizations(
                defect_species="vac_1_Ti_0",
                output_path=os.path.join(self.DATA_DIR, "vasp"),
                distortions=[
                    0.2,
                ],
                threshold=0.3,
                defect_site=[0.0, 0.16666666666666669, 0.25],
            )
        self.assertTrue(
            any(
                f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%/OUTCAR is "
                f"from a "
                "non-spin-polarised calculation (ISPIN = 1), so magnetization analysis "
                "is not possible. Skipping." in str(warning.message)
                for warning in w
            )
        )
        if_present_rm(f"{self.DATA_DIR}/vasp/vac_1_Ti_0/Bond_Distortion_20.0%")

        # Non existent OUTCAR
        with warnings.catch_warnings(record=True) as w:
            mags = analysis.get_site_magnetizations(
                defect_species="vac_1_Ti_0",
                output_path=os.path.join(self.DATA_DIR, "vasp"),
                distortions=[
                    0.1,
                ],  # no OUTCAR in this distortion folder
                threshold=0.3,
                defect_site=[0.0, 0.16666666666666669, 0.25],
            )
            self.assertTrue(
                any(
                    "OUTCAR file not found in path" in str(warning.message)
                    for warning in w
                )
            )


if __name__ == "__main__":
    unittest.main()
