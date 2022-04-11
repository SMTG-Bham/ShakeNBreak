import unittest
import os

import numpy as np

from pymatgen.core.structure import Structure
from defect_finder import distortions


class DistortionTestCase(unittest.TestCase):
    """Test defect-finder structure distortion functions"""

    def setUp(self):
        DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
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
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Distortion_0.1_Rattled_POSCAR")
        )

    def test_bdm(self):
        """Test bond distortion function"""
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.bdm(
            self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords, verbose=False
        )
        self.assertEqual(output["distorted_structure"], self.V_Cd_minus0pt5_struc)
        self.assertEqual(output["undistorted_structure"], self.V_Cd_struc)
        self.assertEqual(output["num_distorted_neighbours"], 2)
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        self.assertCountEqual(output["distorted_atoms"], [(33, "Te"), (42, "Te")])

    def test_rattle(self):
        """Test structure rattle function"""
        self.assertEqual(distortions.rattle(self.V_Cd_minus0pt5_struc),
                         self.V_Cd_minus0pt5_struc_rattled)
        self.assertEqual(distortions.rattle(self.V_Cd_minus0pt5_struc, stdev=0.1),
                         self.V_Cd_minus0pt5_struc_0pt1_rattled)


if __name__ == "__main__":
    unittest.main()
