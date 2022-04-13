import unittest
import os
import pickle

import numpy as np

from pymatgen.core.structure import Structure
from defect_finder import BDM, distortions


class BDMTestCase(unittest.TestCase):
    """Test defect-finder structure distortion helper functions"""

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
