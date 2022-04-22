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
            "CdTe_vac_1_Cd_0_stdev_0.25: E diff. between minimum found with -0.55 RBDM and "
            "unperturbed: -0.76 eV.\n"
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


if __name__ == "__main__":
    unittest.main()
