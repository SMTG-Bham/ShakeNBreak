import unittest
import os
import pickle
import copy
from unittest.mock import patch
import shutil

import numpy as np

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
        self.In_Cd_1_distortion_data = analyse_defects.open_file(
            os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt")
        )  # note this was rattled with the old, non-Monte Carlo rattling (ASE's atoms.rattle())
        self.organized_In_Cd_1_distortion_data = analyse_defects.organize_data(
            self.In_Cd_1_distortion_data
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


if __name__ == "__main__":
    unittest.main()
