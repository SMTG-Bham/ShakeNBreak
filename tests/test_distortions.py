import unittest
import os
import pickle

import numpy as np

from pymatgen.core.structure import Structure
from shakenbreak import distortions


class DistortionTestCase(unittest.TestCase):
    """Test shakenbreak structure distortion functions"""

    def setUp(self):
        DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        with open(os.path.join(DATA_DIR, "CdTe_defects_dict.pickle"), "rb") as fp:
            self.cdte_defect_dict = pickle.load(fp)

        self.V_Cd_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_POSCAR")
        )
        self.V_Cd_minus0pt5_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Unrattled_POSCAR")
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
        self.Int_Cd_1_minus0pt6_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_1_-60%_Distortion_Unrattled_POSCAR")
        )
        self.Int_Cd_1_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_1_-60%_Distortion_Rattled_POSCAR")
        )
        # Confirm correct structures and pickle dict:
        self.assertEqual(
            self.V_Cd_struc,
            self.cdte_defect_dict["vacancies"][0]["supercell"]["structure"],
        )
        self.assertEqual(
            self.Int_Cd_1_struc,
            self.cdte_defect_dict["interstitials"][1]["supercell"]["structure"],
        )

    def test_bdm_V_Cd(self):
        """Test bond distortion function for V_Cd"""
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.bdm(
            self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords, verbose=False
        )
        self.assertEqual(output["distorted_structure"], self.V_Cd_minus0pt5_struc)
        self.assertEqual(output["undistorted_structure"], self.V_Cd_struc)
        self.assertEqual(output["num_distorted_neighbours"], 2)
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        self.assertEqual(output.get("defect_site_index"), None)
        self.assertCountEqual(output["distorted_atoms"], [(33, "Te"), (42, "Te")])

    def test_bdm_Int_Cd_1(self):
        """Test bond distortion function for Int_Cd_1"""
        site_index = 65  # Cd interstitial site index (VASP indexing)
        output = distortions.bdm(
            self.Int_Cd_1_struc, 2, 0.4, site_index=site_index, verbose=False
        )
        self.assertEqual(output["distorted_structure"], self.Int_Cd_1_minus0pt6_struc)
        self.assertEqual(output["undistorted_structure"], self.Int_Cd_1_struc)
        self.assertEqual(output["num_distorted_neighbours"], 2)
        self.assertEqual(output["defect_site_index"], 65)
        self.assertEqual(output.get("defect_frac_coords"), None)
        self.assertCountEqual(output["distorted_atoms"], [(10, "Cd"), (22, "Cd")])

    def test_rattle_V_Cd(self):
        """Test structure rattle function for V_Cd"""
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        d_min = 0.85 * sorted_distances[len(self.V_Cd_struc) + 20]

        rattling_atom_indices = np.arange(0, 63)
        idx = np.in1d(rattling_atom_indices, [i - 1 for i in [33, 42]])
        rattling_atom_indices = rattling_atom_indices[
            ~idx
        ]  # removed distorted Te indices

        self.assertEqual(
            distortions.rattle(
                self.V_Cd_minus0pt5_struc,
                d_min=d_min,
                active_atoms=rattling_atom_indices,
            ),
            self.V_Cd_minus0pt5_struc_rattled,
        )
        self.assertEqual(
            distortions.rattle(
                self.V_Cd_minus0pt5_struc,
                stdev=0.1,
                d_min=d_min,
                active_atoms=rattling_atom_indices,
            ),
            self.V_Cd_minus0pt5_struc_0pt1_rattled,
        )

    def test_rattle_Int_Cd_1(self):
        """Test structure rattle function for Int_Cd_1"""
        sorted_distances = np.sort(self.Int_Cd_1_struc.distance_matrix.flatten())
        d_min = 0.85 * sorted_distances[len(self.Int_Cd_1_struc) + 20]

        rattling_atom_indices = np.arange(
            0, 64
        )  # not including index 64 which is Int_Cd_1
        idx = np.in1d(rattling_atom_indices, [i - 1 for i in [10, 22]])
        rattling_atom_indices = rattling_atom_indices[
            ~idx
        ]  # removed distorted Cd indices

        self.assertEqual(
            distortions.rattle(
                self.Int_Cd_1_minus0pt6_struc,
                d_min=d_min,
                active_atoms=rattling_atom_indices,
            ),
            self.Int_Cd_1_minus0pt6_struc_rattled,
        )


if __name__ == "__main__":
    unittest.main()
