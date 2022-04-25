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
from shakenbreak import analyse_defects


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class AnalyseDefectsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        self.V_Cd_distortion_data = analyse_defects.open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
        )
        self.organized_V_Cd_distortion_data = analyse_defects.organize_data(
            self.V_Cd_distortion_data
        )
        self.V_Cd_distortion_data_no_unperturbed = analyse_defects.open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt")
        )
        self.organized_V_Cd_distortion_data_no_unperturbed = (
            analyse_defects.organize_data(self.V_Cd_distortion_data_no_unperturbed)
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR")
        )
        self.In_Cd_1_distortion_data = analyse_defects.open_file(
            os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt")
        )  # note this was rattled with the old, non-Monte Carlo rattling (ASE's atoms.rattle())
        self.organized_In_Cd_1_distortion_data = analyse_defects.organize_data(
            self.In_Cd_1_distortion_data
        )
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR")
        )

    def tearDown(self):
        # restore the original file (after 'no unperturbed' tests):
        shutil.copy(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt"),
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/BDM/vac_1_Cd_0.txt"),
        )

    @patch("builtins.print")
    def test_open_file(self, mock_print):
        """Test open_file() function."""
        returned_data = analyse_defects.open_file("fake_file")
        self.assertEqual(returned_data, [])
        mock_print.assert_called_once_with("Path fake_file does not exist")

        self.assertEqual(len(self.V_Cd_distortion_data), 52)
        self.assertListEqual(
            [
                "vac_1_Cd_0_0.0%_Bond_Distortion",
                "-205.72650569",
                "vac_1_Cd_0_-10.0%_Bond_Distortion",
            ],
            self.V_Cd_distortion_data[:3],
        )
        self.assertEqual("-205.72311458", self.V_Cd_distortion_data[-1])

        # test In_Cd_1 parsing:
        self.assertListEqual(
            self.In_Cd_1_distortion_data,
            [
                "sub_1_In_on_Cd_1_only_rattled",
                "-214.88259023",
                "sub_1_In_on_Cd_1_Unperturbed_Defect",
                "-214.87608986",
            ],
        )

    def test_organize_data(self):
        """Test organize_data() function."""
        empty_dict = {"distortions": [], "Unperturbed": []}
        self.assertTrue(isinstance(self.organized_V_Cd_distortion_data, dict))
        self.assertTrue(empty_dict.keys() == self.organized_V_Cd_distortion_data.keys())
        self.assertEqual(len(self.organized_V_Cd_distortion_data), 2)
        self.assertEqual(len(self.organized_V_Cd_distortion_data["distortions"]), 25)
        self.assertEqual(
            self.organized_V_Cd_distortion_data["Unperturbed"], -205.72311458
        )
        # test distortions subdict has (distortion) keys and (energy) values as floats:
        self.assertSetEqual(
            set(map(type, self.organized_V_Cd_distortion_data["distortions"].values())),
            {float},
        )
        self.assertSetEqual(
            set(map(type, self.organized_V_Cd_distortion_data["distortions"].keys())),
            {float},
        )
        # test one entry:
        self.assertEqual(
            self.organized_V_Cd_distortion_data["distortions"][-0.35], -206.47790687
        )

        # test In_Cd_1:
        self.assertDictEqual(
            self.organized_In_Cd_1_distortion_data,
            {"distortions": {"rattled": -214.88259023}, "Unperturbed": -214.87608986},
        )

        # test with no 'Unperturbed':
        self.assertTrue(
            isinstance(self.organized_V_Cd_distortion_data_no_unperturbed, dict)
        )
        self.assertFalse(
            "Unperturbed" in self.organized_V_Cd_distortion_data_no_unperturbed
        )
        self.assertEqual(len(self.organized_V_Cd_distortion_data_no_unperturbed), 1)
        self.assertEqual(
            len(self.organized_V_Cd_distortion_data_no_unperturbed["distortions"]), 25
        )
        # test one entry:
        self.assertEqual(
            self.organized_V_Cd_distortion_data_no_unperturbed["distortions"][-0.35],
            -206.47790687,
        )

    def test_get_gs_distortion(self):
        """Test get_gs_distortion() function."""
        gs_distortion = analyse_defects.get_gs_distortion(
            self.organized_V_Cd_distortion_data
        )
        self.assertEqual(gs_distortion, (-0.7551820700000178, -0.55))

        # test In_Cd_1:
        gs_distortion = analyse_defects.get_gs_distortion(
            self.organized_In_Cd_1_distortion_data
        )
        self.assertEqual(gs_distortion, (-0.006500369999997702, "rattled"))

        # test with 'Unperturbed' not present:
        gs_distortion_no_unperturbed = analyse_defects.get_gs_distortion(
            self.organized_V_Cd_distortion_data_no_unperturbed
        )
        self.assertEqual(gs_distortion_no_unperturbed, (None, -0.55))

    @patch("builtins.print")
    def test_sort_data(self, mock_print):
        """Test sort_data() function."""
        # test V_Cd_distortion_data:
        gs_distortion = analyse_defects.get_gs_distortion(
            self.organized_V_Cd_distortion_data
        )
        sorted_V_Cd_distortion_data = analyse_defects.sort_data(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
        )
        self.assertEqual(
            sorted_V_Cd_distortion_data,
            (self.organized_V_Cd_distortion_data, *gs_distortion),
        )
        mock_print.assert_called_once_with(
            "CdTe_vac_1_Cd_0_stdev_0.25: Energy difference between minimum, "
            + "found with -0.55 bond distortion, and unperturbed: -0.76 eV.\n"
        )

        # test In_Cd_1:
        gs_distortion = analyse_defects.get_gs_distortion(
            self.organized_In_Cd_1_distortion_data
        )
        with patch("builtins.print") as mock_In_Cd_print:
            sorted_In_Cd_1_distortion_data = analyse_defects.sort_data(
                os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt"),
            )
            mock_In_Cd_print.assert_not_called()
        self.assertEqual(
            sorted_In_Cd_1_distortion_data,
            (self.organized_In_Cd_1_distortion_data, *gs_distortion),
        )

        # test with 'Unperturbed' not present:
        gs_distortion_no_unperturbed = analyse_defects.get_gs_distortion(
            self.organized_V_Cd_distortion_data_no_unperturbed
        )
        with patch("builtins.print") as mock_no_unperturbed_print:
            organized_V_Cd_distortion_data_no_unperturbed = analyse_defects.sort_data(
                os.path.join(
                    self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt"
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
                f"{os.path.join(self.DATA_DIR, 'CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt')}. "
                f"Lowest energy structure found with -0.55 bond distortion.\n"
            )

    def test_grab_contcar(self):
        """Test grab_contcar() function."""
        with warnings.catch_warnings(record=True) as w:
            output = analyse_defects.grab_contcar("fake_file")
            warning_message = (
                "fake_file file doesn't exist, storing as 'Not converged'. Check "
                "path & relaxation"
            )
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertEqual(output, "Not converged")

        with warnings.catch_warnings(record=True) as w:
            output = analyse_defects.grab_contcar(
                os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt")
            )
            warning_message = (
                f"Problem obtaining structure from: "
                f"{os.path.join(self.DATA_DIR, 'CdTe_sub_1_In_on_Cd_1.txt')}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertEqual(output, "Not converged")

        with warnings.catch_warnings(record=True) as w:
            output = analyse_defects.grab_contcar(
                os.path.join(self.DATA_DIR, "CdTe_V_Cd_POSCAR")
            )
            V_Cd_struc = Structure.from_file(
                os.path.join(self.DATA_DIR, "CdTe_V_Cd_POSCAR")
            )
            self.assertEqual(len(w), 0)
            self.assertEqual(output, V_Cd_struc)

    def test_analyse_defect_site(self):
        """Test analyse_defect_site() function."""
        # test V_Cd:
        with patch("builtins.print") as mock_print:
            output = analyse_defects.analyse_defect_site(
                self.V_Cd_minus0pt5_struc_rattled, name="Test pop", vac_site=[0, 0, 0]
            )
            mock_print.assert_any_call("==> ", "Test pop structural analysis ", " <==")
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
                        0: 0.21944138045080427,
                        1: 0.8789337233604618,
                        2: 0.11173135591518825,
                        3: 0.3876174560001795,
                        4: 0.3965691083975486,
                    },
                }
            )
            pd.testing.assert_frame_equal(expected_V_Cd_crystalNN_coord_df, output[0])
            expected_V_Cd_crystalNN_bonding_df = pd.DataFrame(
                {
                    "Element": {0: "Te", 1: "Te", 2: "Te", 3: "Te"},
                    "Distance": {0: "1.417", 1: "1.417", 2: "2.620", 3: "3.008"},
                }
            )
            pd.testing.assert_frame_equal(expected_V_Cd_crystalNN_bonding_df, output[1])
            self.assertEqual(len(output), 2)

        # test Int_Cd_2:
        with patch("builtins.print") as mock_print:
            output = analyse_defects.analyse_defect_site(
                self.Int_Cd_2_minus0pt6_NN_10_struc_rattled,
                name="Int_Cd_2",
                site_num=65,
            )
            mock_print.assert_any_call("==> ", "Int_Cd_2 structural analysis ", " <==")
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
                        0: 0.778391113850372,
                        1: 0.014891251252011014,
                        2: 0.058350214482398306,
                    },
                }
            )
            pd.testing.assert_frame_equal(
                expected_Int_Cd_2_NN_10_crystalNN_coord_df, output[0]
            )
            expected_Int_Cd_2_NN_10_crystalNN_bonding_df = pd.DataFrame(
                {
                    "Element": {0: "Cd", 1: "Cd", 2: "Cd", 3: "Te", 4: "Te", 5: "Te"},
                    "Distance": {
                        0: "1.085",
                        1: "1.085",
                        2: "1.085",
                        3: "1.085",
                        4: "1.085",
                        5: "1.085",
                    },
                }
            )
            pd.testing.assert_frame_equal(
                expected_Int_Cd_2_NN_10_crystalNN_bonding_df, output[1]
            )
            self.assertEqual(len(output), 2)

        # test error catching:
        with self.assertRaises(ValueError):
            analyse_defects.analyse_defect_site(
                self.Int_Cd_2_minus0pt6_NN_10_struc_rattled,
                name="Int_Cd_2",
                # no site_num or vac_coords specified
            )

    def test_get_structures(self):
        """Test get_structures() function."""
        # V_Cd_0 with defaults (reading from `distortion_metadata.json`):
        defect_structures_dict = analyse_defects.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        self.assertEqual(len(defect_structures_dict), 26)
        bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
        self.assertEqual(
            list(defect_structures_dict.keys()), bond_distortions + ["Unperturbed"]
        )
        relaxed_0pt5_V_Cd_structure = Structure.from_file(
            os.path.join(
                self.DATA_DIR,
                "vac_1_Cd_0/BDM/vac_1_Cd_0_-50.0%_Bond_Distortion/vasp_gam/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)

        # V_Cd_0 with a defined subset (using `bond_distortions`):
        defect_structures_dict = analyse_defects.get_structures(
            defect_species="vac_1_Cd_0",
            output_path=self.DATA_DIR,
            bond_distortions=[-0.5, -0.25, 0],
        )
        self.assertEqual(
            len(defect_structures_dict), 4
        )  # 3 distortions plus unperturbed
        relaxed_0pt5_V_Cd_structure = Structure.from_file(
            os.path.join(
                self.DATA_DIR,
                "vac_1_Cd_0/BDM/vac_1_Cd_0_-50.0%_Bond_Distortion/vasp_gam/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)

        # V_Cd_0 with a defined subset (using `distortion_increment`):
        with warnings.catch_warnings(record=True) as w:
            defect_structures_dict = analyse_defects.get_structures(
                defect_species="vac_1_Cd_0",
                output_path=self.DATA_DIR,
                distortion_increment=0.2,
            )  # only read 20% increments
            self.assertEqual(
                len(defect_structures_dict), 8
            )  # 7 distortions plus unperturbed
            self.assertEqual(len(w), 3)  # 3 warnings for positive distortions
            self.assertEqual(defect_structures_dict[0.4], "Not converged")

        # test warnings for wrong defect species:
        with warnings.catch_warnings(record=True) as w:
            wrong_defect_structures_dict = analyse_defects.get_structures(
                defect_species="vac_1_Cd_1",  # wrong defect species
                output_path=self.DATA_DIR,
                distortion_increment=0.025,
            )
            self.assertEqual(len(w), 50)
            for warning in w:
                self.assertEqual(warning.category, UserWarning)
            final_warning_message = (
                "vac_1_Cd_1/BDM/vac_1_Cd_1_Unperturbed_Defect/vasp_gam"
                "/CONTCAR file doesn't exist, storing as 'Not converged'. "
                "Check path & relaxation"
            )
            self.assertIn(final_warning_message, str(w[-1].message))
            penultimate_warning_message = (  # assumes range of +/- 60%
                "vac_1_Cd_1/BDM/vac_1_Cd_1_60.0%_Bond_Distortion/"
                "vasp_gam/CONTCAR file doesn't exist, storing as 'Not "
                "converged'. Check path & relaxation"
            )
            self.assertIn(penultimate_warning_message, str(w[-2].message))
            for val in wrong_defect_structures_dict.values():
                self.assertEqual(val, "Not converged")

        # test error catching:
        with self.assertRaises(Exception) as e:
            wrong_path_exception = Exception(
                "No `distortion_metadata.json` file found in wrong_path. Please specify "
                "`distortion_increment` or `bond_distortions`."
            )
            analyse_defects.get_structures(
                defect_species="vac_1_Cd_0", output_path="wrong_path"
            )
            self.assertIn(wrong_path_exception, e.exception)

    def test_get_energies(self):
        """Test get_energies() function."""
        # V_Cd_0 with defaults (reading from `vac_1_Cd_0/BDM/vac_1_Cd_0.txt`):
        defect_energies_dict = analyse_defects.get_energies(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
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

        # V_Cd_0 with meV (reading from `vac_1_Cd_0/BDM/vac_1_Cd_0.txt`):
        defect_energies_meV_dict = analyse_defects.get_energies(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR, units="meV"
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
                self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt"
            ),
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/BDM/vac_1_Cd_0.txt"),
        )
        # Note we copy back to original in self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            warning_message = (
                "Unperturbed defect energy not found in energies file. Energies will be given "
                "relative to the lowest energy defect structure found."
            )
            defect_energies_dict = analyse_defects.get_energies(
                defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
            )
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))

            energies_dict_keys_dict = {"distortions": None}
            self.assertEqual(defect_energies_dict.keys(), energies_dict_keys_dict.keys())
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
        defect_structures_dict = analyse_defects.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        with patch("builtins.print") as mock_print:
            max_dist_dict = analyse_defects.calculate_struct_comparison(
                defect_structures_dict
            )
            mock_print.assert_called_with("Comparing structures to Unperturbed...")
        self.assertEqual(
            len(max_dist_dict), len(defect_structures_dict)
        )  # one for each
        self.assertEqual(
            max_dist_dict.keys(), defect_structures_dict.keys()
        )  # one for each
        np.testing.assert_almost_equal(max_dist_dict[-0.4], 0.24573512684427087)
        np.testing.assert_almost_equal(max_dist_dict[-0.2], 0.007657854604646658)
        np.testing.assert_almost_equal(
            max_dist_dict["Unperturbed"], 5.320996143118748e-16
        )

        # V_Cd_0 with 'rms' (reading from `vac_1_Cd_0` and `distortion_metadata.json`):
        rms_dict = analyse_defects.calculate_struct_comparison(
            defect_structures_dict, "rms"
        )
        self.assertEqual(len(rms_dict), len(defect_structures_dict))  # one for each
        self.assertEqual(rms_dict.keys(), defect_structures_dict.keys())  # one for each
        np.testing.assert_almost_equal(rms_dict[-0.4], 0.0666267898227637)
        np.testing.assert_almost_equal(rms_dict[-0.2], 0.0023931134449495075)
        np.testing.assert_almost_equal(rms_dict["Unperturbed"], 1.4198258237093096e-16)

        # test with specified ref_structure as dict key:
        with patch("builtins.print") as mock_print:
            rms_dict = analyse_defects.calculate_struct_comparison(
                defect_structures_dict, "rms", ref_structure=-0.4
            )
            mock_print.assert_called_with(
                "Comparing structures to -40.0% bond distorted structure..."
            )
        # spot check:
        self.assertEqual(round(rms_dict[-0.2], 3), 0.067)
        self.assertTrue(np.isclose(rms_dict[-0.4], 0))
        self.assertEqual(round(rms_dict["Unperturbed"], 3), 0.067)

        # test with specified ref_structure as Structure object:
        with patch("builtins.print") as mock_print:
            rms_dict = analyse_defects.calculate_struct_comparison(
                defect_structures_dict,
                "rms",
                ref_structure=self.V_Cd_minus0pt5_struc_rattled,
            )
            mock_print.assert_called_with(
                "Comparing structures to specified ref_structure (Cd31 Te32)..."
            )
        # spot check:
        self.assertEqual(round(rms_dict[-0.2], 3), 0.142)
        self.assertEqual(round(rms_dict[-0.4], 3), 0.139)
        self.assertEqual(round(rms_dict["Unperturbed"], 3), 0.142)

        # test kwargs:
        rms_dict = analyse_defects.calculate_struct_comparison(
            defect_structures_dict, "max_dist", stol=0.01
        )
        # spot check:
        self.assertEqual(round(rms_dict[-0.2], 3), 0.008)
        self.assertIsNone(rms_dict[-0.4])
        self.assertTrue(np.isclose(rms_dict["Unperturbed"], 0))

        # test error catching:
        with self.assertRaises(KeyError) as e:
            wrong_key_error = KeyError(
                "Reference structure key 'Test pop' not found in defect_structures_dict."
            )
            analyse_defects.calculate_struct_comparison(
                defect_structures_dict, ref_structure="Test pop"
            )
            self.assertIn(wrong_key_error, e.exception)

        with self.assertRaises(ValueError) as e:
            unconverged_error = ValueError(
                "Specified reference structure (with key 'Not converged') is not converged and "
                "cannot be used for structural comparison."
            )
            unconverged_structures_dict = defect_structures_dict.copy()
            unconverged_structures_dict["Unperturbed"] = "Not converged"
            analyse_defects.calculate_struct_comparison(
                unconverged_structures_dict,
            )
            self.assertIn(unconverged_error, e.exception)

        with self.assertRaises(TypeError) as e:
            wrong_type_error = TypeError(
                "ref_structure must be either a key from defect_structures_dict or a pymatgen "
                "Structure object. Got <class 'int'> instead."
            )
            analyse_defects.calculate_struct_comparison(
                defect_structures_dict, ref_structure=1
            )
            self.assertIn(wrong_type_error, e.exception)

    def test_compare_structures(self):
        """Test compare_structures() function."""
        # V_Cd_0 with defaults (reading from `vac_1_Cd_0` and `distortion_metadata.json`):
        defect_structures_dict = analyse_defects.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        defect_energies_dict = analyse_defects.get_energies(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        with patch("builtins.print") as mock_print:
            struct_comparison_df = analyse_defects.compare_structures(
                defect_structures_dict, defect_energies_dict
            )
            mock_print.assert_called_with("Comparing structures to Unperturbed...")
        self.assertIsInstance(struct_comparison_df, pd.DataFrame)
        self.assertEqual(len(struct_comparison_df), len(defect_structures_dict))
        self.assertEqual(
            struct_comparison_df.columns.to_list(),
            ["Bond Dist.", "RMS", "Max. dist (Å)", "Rel. E (eV)"],
        )
        self.assertEqual(
            struct_comparison_df["Bond Dist."].to_list(),
            list(defect_structures_dict.keys()),
        )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 0.002, 0.008, 0.0]
        )
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 0.067, 0.246, -0.75]
        )
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 0.000, 0.000, 0.00]
        )

        # test with specified ref_structure as dict key:
        with patch("builtins.print") as mock_print:
            struct_comparison_df = analyse_defects.compare_structures(
                defect_structures_dict, defect_energies_dict, ref_structure=-0.4
            )
            mock_print.assert_called_with(
                "Comparing structures to -40.0% bond distorted " "structure..."
            )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 0.067, 0.243, 0.00]
        )
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 0.00, 0.00, -0.75]
        )
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 0.067, 0.246, 0.00]
        )

        # test with specified ref_structure as Structure object:
        with patch("builtins.print") as mock_print:
            struct_comparison_df = analyse_defects.compare_structures(
                defect_structures_dict,
                defect_energies_dict,
                ref_structure=self.V_Cd_minus0pt5_struc_rattled,
            )
            mock_print.assert_called_with(
                "Comparing structures to specified ref_structure (Cd31 Te32)..."
            )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 0.142, 0.392, 0.00]
        )
        self.assertEqual(
            struct_comparison_df.iloc[8].to_list(), [-0.4, 0.139, 0.304, -0.75]
        )
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 0.142, 0.399, 0.00]
        )

        # test kwargs:
        struct_comparison_df = analyse_defects.compare_structures(
            defect_structures_dict, defect_energies_dict, stol=0.01, units="meV"
        )
        # spot check:
        self.assertEqual(
            struct_comparison_df.iloc[16].to_list(), [-0.2, 0.002, 0.008, 0.00]
        )
        self.assertTrue(np.isnan(struct_comparison_df.iloc[8].to_list()[1]))  # no RMS
        self.assertTrue(
            np.isnan(struct_comparison_df.iloc[8].to_list()[2])
        )  # no max dist
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 0.000, 0.000, 0.00]
        )
        self.assertEqual(
            struct_comparison_df.columns.to_list(),
            ["Bond Dist.", "RMS", "Max. dist (Å)", "Rel. E (meV)"],
        )

        # test error catching:
        with self.assertRaises(KeyError) as e:
            wrong_key_error = KeyError(
                "Reference structure key 'Test pop' not found in defect_structures_dict."
            )
            analyse_defects.compare_structures(
                defect_structures_dict, defect_energies_dict, ref_structure="Test pop"
            )
            self.assertIn(wrong_key_error, e.exception)

        with self.assertRaises(ValueError) as e:
            unconverged_error = ValueError(
                "Specified reference structure (with key 'Not converged') is not converged and "
                "cannot be used for structural comparison."
            )
            unconverged_structures_dict = defect_structures_dict.copy()
            unconverged_structures_dict["Unperturbed"] = "Not converged"
            analyse_defects.compare_structures(
                unconverged_structures_dict, defect_energies_dict
            )
            self.assertIn(unconverged_error, e.exception)

        with self.assertRaises(TypeError) as e:
            wrong_type_error = TypeError(
                "ref_structure must be either a key from defect_structures_dict or a pymatgen "
                "Structure object. Got <class 'int'> instead."
            )
            analyse_defects.compare_structures(
                defect_structures_dict, defect_energies_dict, ref_structure=1
            )
            self.assertIn(wrong_type_error, e.exception)


if __name__ == "__main__":
    unittest.main()
