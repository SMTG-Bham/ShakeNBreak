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
        if_present_rm(os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion"))

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
            "vac_1_Cd": [0],
        }
        self.assertEqual(
            defect_charges_dict.keys(),
            expected_dict.keys(),
        )
        for i in expected_dict:  # Need to do this way to allow different list orders
            self.assertCountEqual(defect_charges_dict[i], expected_dict[i])

    def test_compare_champion_to_distortions(self):
        """Test comparing champion defect energies to distorted defect energies"""
        os.mkdir(os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion"))
        champion_txt = f"""vac_1_Cd_2_-7.5%_Bond_Distortion
        -205.700
        vac_1_Cd_2_-10.0%_Bond_Distortion
        -205.750
        vac_1_Cd_2_Unperturbed_Defect
        -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion/vac_1_Cd_0.txt"), "w"
        ) as fp:
            fp.write(champion_txt)

        output = champion_defects_rerun.compare_champion_to_distortions(
            defect_species="vac_1_Cd_0", base_path=self.DATA_DIR
        )
        self.assertFalse(output[0])
        np.testing.assert_almost_equal(output[1], 0.635296650000015)

        champion_txt = f"""vac_1_Cd_2_only_rattled
        -206.700
        vac_1_Cd_2_Unperturbed_Defect
        -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion/vac_1_Cd_0.txt"), "w"
        ) as fp:
            fp.write(champion_txt)

        with patch("builtins.print") as mock_print:
            output = champion_defects_rerun.compare_champion_to_distortions(
                defect_species="vac_1_Cd_0", base_path=self.DATA_DIR
            )
            mock_print.assert_called_with(
                "Lower energy structure found for the 'champion' relaxation with vac_1_Cd_0, "
                "with an energy -0.22 eV lower than the previous lowest energy from distortions, "
                "with an energy -0.98 eV lower than relaxation from the Unperturbed structure."
            )
        self.assertTrue(output[0])
        np.testing.assert_almost_equal(output[1], -0.9768854200000021)


if __name__ == "__main__":
    unittest.main()
