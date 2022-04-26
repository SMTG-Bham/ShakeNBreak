import unittest
import os
import pickle
import copy
from unittest.mock import patch
import shutil

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from shakenbreak import champion_defects_rerun, input


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class ChampTestCase(unittest.TestCase):
    """Test ShakeNBreak energy-lowering distortion identification functions"""

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        with open(os.path.join(self.DATA_DIR, "CdTe_defects_dict.pickle"), "rb") as fp:
            self.cdte_defect_dict = pickle.load(fp)
        self.V_Cd_dict = self.cdte_defect_dict["vacancies"][0]
        self.Int_Cd_2_dict = self.cdte_defect_dict["interstitials"][1]

        self.defect_folders_list = [
            "Int_Cd_2_0",
            "Int_Cd_2_1",
            "Int_Cd_2_-1",
            "Int_Cd_2_2",
            "as_1_Cd_on_Te_1",
            "as_1_Cd_on_Te_2",
            "sub_1_In_on_Cd_1",
        ]

    def tearDown(self):
        # remove folders generated during tests
        for i in self.defect_folders_list:
            if_present_rm(os.path.join(self.DATA_DIR, i))

    def test_read_defects_directories(self):
        """Test reading defect directories and parsing to dictionaries"""
        defect_charges_dict = champion_defects_rerun.read_defects_directories(
            self.DATA_DIR
        )
        self.assertDictEqual(defect_charges_dict, {"vac_1_Cd": [0]})

        for i in self.defect_folders_list:
            os.mkdir(os.path.join(self.DATA_DIR, i))

        defect_charges_dict = champion_defects_rerun.read_defects_directories(
            self.DATA_DIR
        )
        expected_dict = {
                "Int_Cd_2": [2, -1, 1, 0],
                "as_1_Cd_on_Te": [1, 2],
                "sub_1_In_on_Cd": [1],
                "vac_1_Cd": [0]
            }
        self.assertEqual(
            defect_charges_dict.keys(),
            expected_dict.keys(),
        )
        for i in expected_dict:  # Need to do this way to allow different list orders
            self.assertCountEqual(defect_charges_dict[i], expected_dict[i])


if __name__ == "__main__":
    unittest.main()
