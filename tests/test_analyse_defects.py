import unittest
import os
import pickle
import copy
from unittest.mock import patch
import shutil

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from doped import vasp_input
from shakenbreak import analyse_defects


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class AnalyseDefectsTestCase(unittest.TestCase):
    def setUp(self):
        DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        self.V_Cd_distortion_data = analyse_defects.open_file(
            os.path.join(DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
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

    def test_organize_data(self):
        """Test organize_data() function."""
        organized_V_Cd_distortion_data = analyse_defects.organize_data(
            self.V_Cd_distortion_data
        )
        empty_dict = {"distortions": [], "Unperturbed": []}
        self.assertTrue(isinstance(organized_V_Cd_distortion_data, dict))
        self.assertTrue(empty_dict.keys() == organized_V_Cd_distortion_data.keys())
        self.assertEqual(len(organized_V_Cd_distortion_data), 2)
        self.assertEqual(len(organized_V_Cd_distortion_data["distortions"]), 25)
        self.assertEqual(organized_V_Cd_distortion_data["Unperturbed"], -205.72311458)
        # test distortions subdict has (distortion) keys and (energy) values as floats:
        self.assertSetEqual(
            set(map(type, organized_V_Cd_distortion_data["distortions"].values())),
            {float},
        )
        self.assertSetEqual(
            set(map(type, organized_V_Cd_distortion_data["distortions"].keys())),
            {float},
        )
        # test one entry:
        self.assertEqual(organized_V_Cd_distortion_data["distortions"][-0.35], -206.47790687)


if __name__ == "__main__":
    unittest.main()
