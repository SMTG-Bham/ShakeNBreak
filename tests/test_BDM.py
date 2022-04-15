import unittest
import os
import pickle
import copy
from unittest.mock import patch

import numpy as np

from pymatgen.core.structure import Structure
from doped import vasp_input
from shakenbreak import BDM, distortions


class BDMTestCase(unittest.TestCase):
    """Test ShakeNBreak structure distortion helper functions"""

    def setUp(self):
        DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        with open(os.path.join(DATA_DIR, "CdTe_defects_dict.pickle"), "rb") as fp:
            self.cdte_defect_dict = pickle.load(fp)

        self.V_Cd_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_0pt1_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Distortion_stdev0pt1_Rattled_POSCAR")
        )
        self.Int_Cd_1_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_1_POSCAR")
        )
        self.Int_Cd_1_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_1_-60%_Distortion_Rattled_POSCAR")
        )

    def test_update_struct_defect_dict(self):
        """Test update_struct_defect_dict function"""
        vasp_defect_inputs = vasp_input.prepare_vasp_defect_inputs(
            copy.deepcopy(self.cdte_defect_dict)
        )
        for key, struc, comment in [
            ("vac_1_Cd_0", self.V_Cd_struc, "V_Cd Undistorted"),
            ("vac_1_Cd_0", self.V_Cd_minus0pt5_struc_rattled, "V_Cd Rattled"),
            ("vac_1_Cd_-2", self.V_Cd_struc, "V_Cd_-2 Undistorted"),
            ("Int_Cd_1_1", self.Int_Cd_1_minus0pt6_struc_rattled, "Int_Cd_1 Rattled"),
        ]:
            charged_defect_dict = vasp_defect_inputs[key]
            output = BDM.update_struct_defect_dict(charged_defect_dict, struc, comment)
            self.assertEqual(output["Defect Structure"], struc)
            self.assertEqual(output["POSCAR Comment"], comment)
            self.assertEqual(
                output["Transformation Dict"],
                charged_defect_dict["Transformation Dict"],
            )

    @patch('builtins.print')
    def test_calc_number_electrons(self, mock_print):
        """Test calc_number_electrons function"""
        oxidation_states = {"Cd": +2, "Te": -2}
        for defect, electron_change in [
            ("vac_1_Cd", -2),
            ("vac_2_Te", 2),
            ("as_1_Cd_on_Te", 4),
            ("as_1_Te_on_Cd", -4),
            ("Int_Cd_1", 2),
            ("Int_Cd_2", 2),
            ("Int_Cd_3", 2),
            ("Int_Te_1", -2),
            ("Int_Te_2", -2),
            ("Int_Te_3", -2),
        ]:
            for defect_type, defect_list in self.cdte_defect_dict.items():
                if defect_type != "bulk":
                    for i in defect_list:
                        if i["name"] == defect:
                            self.assertEqual(
                                BDM.calc_number_electrons(
                                    i, oxidation_states, verbose=False  # test non-verbose
                                ),
                                -electron_change,  # returns negative of electron change
                            )
                            BDM.calc_number_electrons(i, oxidation_states, verbose=True)
                            mock_print.assert_called_with(f"Number of extra/missing electrons of "
                                                          f"defect {defect}: {electron_change} "
                                                          f"-> Δq = {-electron_change}")

    def test_calc_number_neighbours(self):
        """Test calc_number_neighbours function"""
        self.assertEqual(BDM.calc_number_neighbours(0), 0)
        self.assertEqual(BDM.calc_number_neighbours(-2), 2)
        self.assertEqual(BDM.calc_number_neighbours(2), 2)
        self.assertEqual(BDM.calc_number_neighbours(6), 2)
        self.assertEqual(BDM.calc_number_neighbours(-6), 2)
        self.assertEqual(BDM.calc_number_neighbours(8), 0)
        self.assertEqual(BDM.calc_number_neighbours(-8), 0)
        self.assertEqual(BDM.calc_number_neighbours(4), 4)
        self.assertEqual(BDM.calc_number_neighbours(-4), 4)

    def test_apply_rattle_bond_distortions_V_Cd(self):
        """Test apply_rattle_bond_distortions function for V_Cd"""
        V_Cd_dict = self.cdte_defect_dict["vacancies"][0]
        V_Cd_distorted_dict = BDM.apply_rattle_bond_distortions(
            V_Cd_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.5,
            stdev=0.25,
            verbose=True,
        )
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.bdm(
            self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords, verbose=False
        )
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        d_min = 0.85 * sorted_distances[len(self.V_Cd_struc) + 20]
        rattling_atom_indices = np.arange(0, 63)
        idx = np.in1d(rattling_atom_indices, [i - 1 for i in [33, 42]])
        rattling_atom_indices = rattling_atom_indices[
            ~idx
        ]  # removed distorted Te indices
        output[
            "distorted_structure"
        ] = distortions.rattle(  # overwrite with distorted and rattle
            # structure
            output["distorted_structure"],
            d_min=d_min,
            active_atoms=rattling_atom_indices,
        )
        np.testing.assert_equal(V_Cd_distorted_dict, output)
        self.assertEqual(
            V_Cd_distorted_dict["distorted_structure"],
            self.V_Cd_minus0pt5_struc_rattled,
        )

    def test_apply_rattle_bond_distortions_Int_Cd_1(self):
        """Test apply_rattle_bond_distortions function for Int_Cd_1"""
        Int_Cd_1_dict = self.cdte_defect_dict["interstitials"][1]
        Int_Cd_1_distorted_dict = BDM.apply_rattle_bond_distortions(
            Int_Cd_1_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.4,
            stdev=0.25,
            verbose=True,
        )
        output = distortions.bdm(
            self.Int_Cd_1_struc, 2, 0.4, site_index=65, verbose=False
        )
        sorted_distances = np.sort(self.Int_Cd_1_struc.distance_matrix.flatten())
        d_min = 0.85 * sorted_distances[len(self.Int_Cd_1_struc) + 20]
        rattling_atom_indices = np.arange(
            0, 64
        )  # not including index 64 which is Int_Cd_1
        idx = np.in1d(rattling_atom_indices, [i - 1 for i in [10, 22]])
        rattling_atom_indices = rattling_atom_indices[
            ~idx
        ]  # removed distorted Cd indices
        output[
            "distorted_structure"
        ] = distortions.rattle(  # overwrite with distorted and rattle
            output["distorted_structure"],
            d_min=d_min,
            active_atoms=rattling_atom_indices,
        )
        np.testing.assert_equal(Int_Cd_1_distorted_dict, output)
        self.assertEqual(
            Int_Cd_1_distorted_dict["distorted_structure"],
            self.Int_Cd_1_minus0pt6_struc_rattled,
        )
        self.assertDictEqual(output, Int_Cd_1_distorted_dict)


if __name__ == "__main__":
    unittest.main()
