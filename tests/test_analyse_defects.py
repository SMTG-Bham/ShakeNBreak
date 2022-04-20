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
    @patch("builtins.print")
    def test_open_file(self, mock_print):
        """Test open_file() function."""
        returned_data = analyse_defects.open_file("fake_file")
        self.assertEqual(returned_data, [])
        mock_print.assert_called_once_with("Path fake_file does not exist")

        V_Cd_distortion_data = analyse_defects.open_file(
            "../data/CdTe_vac_1_Cd_0_stdev_0.25.txt"
        )
        self.assertEqual(len(V_Cd_distortion_data), 52)
        self.assertListEqual(
            [
                "vac_1_Cd_0_0.0%_Bond_Distortion",
                "-205.72650569",
                "vac_1_Cd_0_-10.0%_Bond_Distortion",
            ],
            V_Cd_distortion_data[:3],
        )
        self.assertEqual("-205.72311458", V_Cd_distortion_data[-1])


if __name__ == "__main__":
    unittest.main()
