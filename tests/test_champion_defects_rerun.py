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
        if_present_rm(os.path.join(self.DATA_DIR, "vac_1_Cd_2"))
        if_present_rm(os.path.join(self.DATA_DIR, "Int_Cd_2_1"))
        if os.path.exists(os.path.join(self.DATA_DIR,'vac_1_Cd_0/champion_vac_1_Cd_0.txt')):
            os.remove(os.path.join(self.DATA_DIR,'vac_1_Cd_0/champion_vac_1_Cd_0.txt'))

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
        # os.mkdir(os.path.join(self.DATA_DIR, "vac_1_Cd_0"))

        # False test:
        champion_txt = f"""Bond_Distortion_-7.5%
        -205.700
        Bond_Distortion_-10.0%
        -205.750
        Unperturbed
        -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion_vac_1_Cd_0.txt"), "w"
        ) as fp:
            fp.write(champion_txt)

        output = champion_defects_rerun.compare_champion_to_distortions(
            defect_species="vac_1_Cd_0", base_path=self.DATA_DIR
        )
        self.assertFalse(output[0])
        np.testing.assert_almost_equal(output[1], 0.635296650000015)

        # True test:
        champion_txt = f"""rattled
        -206.700
        Unperturbed
        -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion_vac_1_Cd_0.txt"), "w"
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

        # test when lower energy, but within threshold:
        output = champion_defects_rerun.compare_champion_to_distortions(
            defect_species="vac_1_Cd_0", base_path=self.DATA_DIR, min_e_diff=0.25
        )
        self.assertFalse(output[0])
        np.testing.assert_almost_equal(output[1], -0.2217033499999843)

    def test_get_champion_defects(self):
        """Test getting champion defect energies"""
        os.mkdir(os.path.join(self.DATA_DIR, "vac_1_Cd_2"))
        os.mkdir(os.path.join(self.DATA_DIR, "vac_1_Cd_2/rattled"))
        os.mkdir(
            os.path.join(self.DATA_DIR, "vac_1_Cd_2/Unperturbed")
        )
        os.mkdir(os.path.join(self.DATA_DIR, "Int_Cd_2_1"))
        os.mkdir(
            os.path.join(
                self.DATA_DIR, "Int_Cd_2_1/Bond_Distortion_-10.0%"
            )
        )
        os.mkdir(
            os.path.join(self.DATA_DIR, "Int_Cd_2_1/Unperturbed")
        )

        # False test (champion higher energy for vac_1_Cd_0):
        champion_txt = f"""rattled
                -205.700
                Unperturbed
                -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion_vac_1_Cd_0.txt"), "w"
        ) as fp:
            fp.write(champion_txt)

        V_Cd_2_txt = f"""rattled
                        -205.900
                        Unperturbed
                        -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_2/vac_1_Cd_2.txt"), "w"
        ) as fp:
            fp.write(V_Cd_2_txt)

        Int_Cd_2_1_txt = f"""Bond_Distortion_-10.0%
                                -205.400
                                Unperturbed
                                -205.843"""
        with open(
            os.path.join(self.DATA_DIR, "Int_Cd_2_1/Int_Cd_2_1.txt"), "w"
        ) as fp:
            fp.write(Int_Cd_2_1_txt)

        defect_charges_dict = champion_defects_rerun.read_defects_directories(
            self.DATA_DIR
        )
        output = champion_defects_rerun.get_champion_defects(
            defect_charges_dict=defect_charges_dict, base_path=self.DATA_DIR
        )
        V_Cd_relaxed_unperturbed_structure = Structure.from_file(
            os.path.join(
                self.DATA_DIR,
                "vac_1_Cd_0/Unperturbed/CONTCAR",
            )
        )
        V_Cd_relaxed_distorted_structure = Structure.from_file(
            os.path.join(
                self.DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR",
            )
        )
        self.assertEqual(
            len(output), 2
        )  # 3 defect species, only 2 with a non-spontaneous
        # energy lowering distortion
        self.assertEqual(
            output["vac_1_Cd_0"]["Unperturbed"], V_Cd_relaxed_unperturbed_structure
        )
        self.assertEqual(
            output["vac_1_Cd_0"]["Distorted"], V_Cd_relaxed_distorted_structure
        )
        np.testing.assert_almost_equal(
            output["vac_1_Cd_0"]["E_drop"], -0.7551820700000178
        )

        self.assertEqual(output["vac_1_Cd_2"]["Unperturbed"], "Not converged")  # No CONTCARs
        self.assertEqual(output["vac_1_Cd_2"]["Distorted"], "Not converged")  # No CONTCARs
        np.testing.assert_almost_equal(
            output["vac_1_Cd_2"]["E_drop"], -0.05700000000001637
        )

        # True test (champion lower energy for vac_1_Cd_0):
        champion_txt = f"""rattled
                        -206.700
                        Unperturbed
                        -205.823"""  # also changing 'Unperturbed' energy here to confirm we take
        # 'Unperturbed' reference from 'BDM' rather than 'champion' folders
        with open(
            os.path.join(self.DATA_DIR, "vac_1_Cd_0/champion_vac_1_Cd_0.txt"), "w"
        ) as fp:
            fp.write(champion_txt)

        output = champion_defects_rerun.get_champion_defects(
            defect_charges_dict=defect_charges_dict, base_path=self.DATA_DIR
        )
        self.assertEqual(
            len(output), 2
        )  # 3 defect species, only 2 with a non-spontaneous
        # energy lowering distortion
        self.assertEqual(
            output["vac_1_Cd_0"]["Unperturbed"], V_Cd_relaxed_unperturbed_structure
        )
        self.assertEqual(
            output["vac_1_Cd_0"]["Distorted"], "Not converged"  # No CONTCAR
        )
        np.testing.assert_almost_equal(
            output["vac_1_Cd_0"]["E_drop"], -0.9768854200000021
        )

        self.assertEqual(output["vac_1_Cd_2"]["Unperturbed"], "Not converged")
        self.assertEqual(output["vac_1_Cd_2"]["Distorted"], "Not converged")
        np.testing.assert_almost_equal(
            output["vac_1_Cd_2"]["E_drop"], -0.05700000000001637
        )


if __name__ == "__main__":
    unittest.main()
