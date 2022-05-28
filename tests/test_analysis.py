import unittest
import os
from unittest.mock import patch
import shutil
import warnings

import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure, Element
from shakenbreak import analysis


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class AnalyseDefectsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        self.V_Cd_distortion_data = analysis._open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
        )
        self.organized_V_Cd_distortion_data = analysis._organize_data(
            self.V_Cd_distortion_data
        )
        self.V_Cd_distortion_data_no_unperturbed = analysis._open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt")
        )
        self.organized_V_Cd_distortion_data_no_unperturbed = analysis._organize_data(
            self.V_Cd_distortion_data_no_unperturbed
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR")
        )
        self.In_Cd_1_distortion_data = analysis._open_file(
            os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt")
        )  # note this was rattled with the old, non-Monte Carlo rattling (ASE's atoms.rattle())
        self.organized_In_Cd_1_distortion_data = analysis._organize_data(
            self.In_Cd_1_distortion_data
        )
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR")
        )

    def tearDown(self):
        # restore the original file (after 'no unperturbed' tests):
        shutil.copy(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt"),
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/vac_1_Cd_0.txt"),
        )

    @patch("builtins.print")
    def test_open_file(self, mock_print):
        """Test _open_file() function."""
        returned_data = analysis._open_file("fake_file")
        self.assertEqual(returned_data, [])
        mock_print.assert_called_once_with("Path fake_file does not exist")

        self.assertEqual(len(self.V_Cd_distortion_data), 52)
        self.assertListEqual(
            [
                "Bond_Distortion_0.0%",
                "-205.72650569",
                "Bond_Distortion_-10.0%",
            ],
            self.V_Cd_distortion_data[:3],
        )
        self.assertEqual("-205.72311458", self.V_Cd_distortion_data[-1])

        # test In_Cd_1 parsing:
        self.assertListEqual(
            self.In_Cd_1_distortion_data,
            [
                "Rattled",
                "-214.88259023",
                "Unperturbed",
                "-214.87608986",
            ],
        )

    def test_organize_data(self):
        """Test _organize_data() function."""
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
            {"distortions": {"Rattled": -214.88259023}, "Unperturbed": -214.87608986},
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
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
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
                os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt"),
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
                os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt"),
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
                f"Lowest energy structure found with -0.55 bond distortion."
            )

        # test error catching:
        with warnings.catch_warnings(record=True) as w:
            output = analysis._sort_data("fake_file")
            warning_message = "No data parsed from fake_file, returning None"
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertEqual(output, (None, None, None))

    def test_grab_contcar(self):
        """Test grab_contcar() function."""
        with warnings.catch_warnings(record=True) as w:
            output = analysis.grab_contcar("fake_file")
            warning_message = (
                "fake_file file doesn't exist, storing as 'Not converged'. Check "
                "path & relaxation"
            )
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertEqual(output, "Not converged")

        with warnings.catch_warnings(record=True) as w:
            output = analysis.grab_contcar(
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
            output = analysis.grab_contcar(
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
            output = analysis.analyse_defect_site(
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
            output = analysis.analyse_defect_site(
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
            analysis.analyse_defect_site(
                self.Int_Cd_2_minus0pt6_NN_10_struc_rattled,
                name="Int_Cd_2",
                # no site_num or vac_coords specified
            )

    def test_get_structures(self):
        """Test get_structures() function."""
        # V_Cd_0 with defaults (reading from subdirectories):
        defect_structures_dict = analysis.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        self.assertEqual(len(defect_structures_dict), 26)
        bond_distortions = list(np.around(np.arange(-0.6, 0.001, 0.025), 3))
        self.assertEqual(
            set(defect_structures_dict.keys()), set(bond_distortions + ["Unperturbed"])
        )
        relaxed_0pt5_V_Cd_structure = Structure.from_file(
            os.path.join(
                self.DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_-50.0%/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)

        # V_Cd_0 with a defined subset (using `bond_distortions`):
        defect_structures_dict = analysis.get_structures(
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
                "vac_1_Cd_0/Bond_Distortion_-50.0%/CONTCAR",
            )
        )
        self.assertEqual(defect_structures_dict[-0.5], relaxed_0pt5_V_Cd_structure)

        # test exception for wrong defect species
        with self.assertRaises(FileNotFoundError) as e:
            wrong_path_error = FileNotFoundError(
                f"Path {self.DATA_DIR}/vac_1_Cd_1 does not exist!"
            )
            analysis.get_structures(
                defect_species="vac_1_Cd_1",
                output_path=self.DATA_DIR,  # wrong defect species
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

    def test_get_energies(self):
        """Test get_energies() function."""
        # V_Cd_0 with defaults (reading from `vac_1_Cd_0/vac_1_Cd_0.txt`):
        defect_energies_dict = analysis.get_energies(
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

        # test verbose = False
        with patch("builtins.print") as mock_not_verbose_print:
            defect_energies_dict = analysis.get_energies(
                defect_species="vac_1_Cd_0", output_path=self.DATA_DIR, verbose=False
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

        # V_Cd_0 with meV (reading from `vac_1_Cd_0/vac_1_Cd_0.txt`):
        defect_energies_meV_dict = analysis.get_energies(
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
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/vac_1_Cd_0.txt"),
        )
        # Note we copy back to original in self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            warning_message = (
                "Unperturbed defect energy not found in energies file. Energies will be given "
                "relative to the lowest energy defect structure found."
            )
            defect_energies_dict = analysis.get_energies(
                defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
            )
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))

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
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        with patch("builtins.print") as mock_print:
            max_dist_dict = analysis.calculate_struct_comparison(defect_structures_dict)
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
                defect_structures_dict, "disp", ref_structure=-0.4
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
        # V_Cd_0 with defaults (reading from `vac_1_Cd_0` and `distortion_metadata.json`):
        defect_structures_dict = analysis.get_structures(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
        )
        defect_energies_dict = analysis.get_energies(
            defect_species="vac_1_Cd_0", output_path=self.DATA_DIR
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
        self.assertTrue(np.isnan(struct_comparison_df.iloc[8].to_list()[1]))  # no RMS
        self.assertTrue(
            np.isnan(struct_comparison_df.iloc[8].to_list()[2])
        )  # no max dist
        self.assertEqual(
            struct_comparison_df.iloc[-1].to_list(), ["Unperturbed", 0.000, 0.000, 0.00]
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
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertEqual(output, None)

    # TODO: Add magnetisation tests


if __name__ == "__main__":
    unittest.main()
