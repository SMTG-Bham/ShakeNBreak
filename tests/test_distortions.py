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

    def test_bdm(self):
        """Test bond distortion function"""
        output = distortions.bdm(
            self.V_Cd_struc, 2, 0.5, frac_coords=np.array([0, 0, 0]), verbose=False
        )
        self.assertEqual(output["distorted_structure"], self.V_Cd_minus0pt5_struc)


if __name__ == "__main__":
    unittest.main()
