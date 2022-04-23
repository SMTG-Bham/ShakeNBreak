import unittest
import os
import pickle
import warnings
from unittest.mock import patch

import numpy as np

from pymatgen.core.structure import Structure
from shakenbreak import distortions


class DistortionTestCase(unittest.TestCase):
    """Test shakenbreak structure distortion functions"""

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        with open(os.path.join(self.DATA_DIR, "CdTe_defects_dict.pickle"), "rb") as fp:
            self.cdte_defect_dict = pickle.load(fp)

        self.V_Cd_struc = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_V_Cd_POSCAR")
        )
        self.V_Cd_minus0pt5_struc = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Unrattled_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_0pt1_rattled = Structure.from_file(
            os.path.join(
                self.DATA_DIR, "CdTe_V_Cd_-50%_Distortion_stdev0pt1_Rattled_POSCAR"
            )
        )
        self.V_Cd_minus0pt5_struc_kwarged = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_V_Cd_-50%_Kwarged_POSCAR")
        )
        self.Int_Cd_2_struc = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_Int_Cd_2_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_struc = Structure.from_file(
            os.path.join(
                self.DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_Unrattled_POSCAR"
            )
        )
        self.Int_Cd_2_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_Rattled_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
            os.path.join(self.DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR")
        )
        # Confirm correct structures and pickle dict:
        self.assertEqual(
            self.V_Cd_struc,
            self.cdte_defect_dict["vacancies"][0]["supercell"]["structure"],
        )
        self.assertEqual(
            self.Int_Cd_2_struc,
            self.cdte_defect_dict["interstitials"][1]["supercell"]["structure"],
        )

        warnings.simplefilter("always")  # Cause all warnings to always be triggered.

    def tearDown(self):
        warnings.resetwarnings()  # Reset warnings to default state.

    @patch("builtins.print")
    def test_distort_V_Cd(self, mock_print):
        """Test bond distortion function for V_Cd"""
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.distort(self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords)
        self.assertEqual(output["distorted_structure"], self.V_Cd_minus0pt5_struc)
        self.assertEqual(output["undistorted_structure"], self.V_Cd_struc)
        self.assertEqual(output["num_distorted_neighbours"], 2)
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        self.assertEqual(output.get("defect_site_index"), None)
        self.assertCountEqual(output["distorted_atoms"], [(33, "Te"), (42, "Te")])
        distortions.distort(
            self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords, verbose=True
        )
        mock_print.assert_called_with(
            f"""\tDefect Site Index / Frac Coords: {vac_coords}
        Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]
        Distorted Neighbour Distances:\n\t[(1.42, 33, 'Te'), (1.42, 42, 'Te')]"""
        )

        # Test if num_nearest_neighbours = 0 that nothing happens:
        output = distortions.distort(self.V_Cd_struc, 0, 0.5, frac_coords=vac_coords)
        self.assertEqual(output["distorted_structure"], self.V_Cd_struc)
        self.assertEqual(output["undistorted_structure"], self.V_Cd_struc)
        self.assertEqual(output["num_distorted_neighbours"], 0)
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        self.assertEqual(output.get("defect_site_index"), None)
        self.assertCountEqual(output["distorted_atoms"], [])
        distortions.distort(
            self.V_Cd_struc, 0, 0.5, frac_coords=vac_coords, verbose=True
        )
        mock_print.assert_called_with(
            f"""\tDefect Site Index / Frac Coords: {vac_coords}
        Original Neighbour Distances: []
        Distorted Neighbour Distances:\n\t[]"""
        )

    @patch("builtins.print")
    def test_distort_Int_Cd_2(self, mock_print):
        """Test bond distortion function for Int_Cd_2"""
        site_index = 65  # Cd interstitial site index (VASP indexing)
        output = distortions.distort(
            self.Int_Cd_2_struc, 2, 0.4, site_index=site_index, verbose=False
        )
        self.assertEqual(output["distorted_structure"], self.Int_Cd_2_minus0pt6_struc)
        self.assertEqual(output["undistorted_structure"], self.Int_Cd_2_struc)
        self.assertEqual(output["num_distorted_neighbours"], 2)
        self.assertEqual(output["defect_site_index"], 65)
        self.assertEqual(output.get("defect_frac_coords"), None)
        self.assertCountEqual(output["distorted_atoms"], [(10, "Cd"), (22, "Cd")])
        distortions.distort(
            self.Int_Cd_2_struc, 2, 0.4, site_index=site_index, verbose=True
        )
        mock_print.assert_called_with(
            f"""\tDefect Site Index / Frac Coords: {site_index}
        Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd')]
        Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd')]"""
        )

        # test correct behaviour with `num_nearest_neighbours` is greater than number of
        # `distorted_element` atoms withing 4.5 Å of the defect site
        output = distortions.distort(
            self.Int_Cd_2_struc,
            num_nearest_neighbours=10,
            distortion_factor=0.4,
            distorted_element="Cd",
            site_index=site_index,
        )
        self.assertEqual(
            output["distorted_structure"], self.Int_Cd_2_minus0pt6_NN_10_struc_rattled
        )
        self.assertEqual(output["undistorted_structure"], self.Int_Cd_2_struc)
        self.assertEqual(output["num_distorted_neighbours"], 10)
        self.assertEqual(output["defect_site_index"], 65)
        self.assertEqual(output.get("defect_frac_coords"), None)
        self.assertCountEqual(
            output["distorted_atoms"],
            [
                (10, "Cd"),
                (22, "Cd"),
                (29, "Cd"),
                (1, "Cd"),
                (14, "Cd"),
                (24, "Cd"),
                (30, "Cd"),
                (38, "Te"),
                (54, "Te"),
                (62, "Te"),
            ],
        )
        distortions.distort(
            self.Int_Cd_2_struc,
            num_nearest_neighbours=10,
            distortion_factor=0.4,
            site_index=site_index,
            distorted_element="Cd",
            verbose=True,
        )
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: {site_index}\n"
            + "        Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd'), "
            + "(2.71, 29, 'Cd'), (4.25, 1, 'Cd'), (4.25, 14, 'Cd'), (4.25, 24, 'Cd'), (4.25, 30, "
            + "'Cd'), (2.71, 38, 'Te'), (2.71, 54, 'Te'), (2.71, 62, 'Te')]\n"
            + "        Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd'), "
            + "(1.09, 29, 'Cd'), (1.7, 1, 'Cd'), (1.7, 14, 'Cd'), (1.7, 24, 'Cd'), "
            + "(1.7, 30, 'Cd'), (1.09, 38, 'Te'), (1.09, 54, 'Te'), (1.09, 62, 'Te')]"
        )

    def test_distort_warnings(self):
        """Test warning messages for bond distortion function"""
        site_index = 65  # Cd interstitial site index (VASP indexing)
        for missing_element in ["C", "O", "H", "N", "S", "P", "X"]:
            for num_neighbours in range(8):
                for distortion_factor in np.arange(-0.6, 0.61, 0.1):
                    with warnings.catch_warnings(record=True) as w:
                        distortions.distort(
                            self.Int_Cd_2_struc,  # cause warning for no `missing_element`
                            # neighbours
                            num_nearest_neighbours=num_neighbours,
                            distortion_factor=distortion_factor,
                            site_index=site_index,
                            distorted_element=missing_element,
                        )
                        warning_message = (
                            f"{missing_element} was specified as the nearest neighbour element to "
                            f"distort, with `distortion_factor` {distortion_factor} but did not "
                            f"find `num_nearest_neighbours` ({num_neighbours}) of these elements "
                            f"within 4.5 Å of the defect site. For the remaining neighbours to "
                            f"distort, we ignore the elemental identity. The final distortion "
                            f"information is:"
                        )
                        if num_neighbours > 0:
                            self.assertEqual(len(w), 1)
                            self.assertEqual(w[0].category, UserWarning)
                            self.assertIn(warning_message, str(w[0].message))
                        else:
                            self.assertEqual(len(w), 0)  # No warning if we distort none

        # test the case where we do have some of the `distorted_element` neighbours, but less than
        # `num_nearest_neighbours` of them with 4.5 Å of the defect site
        for num_neighbours in range(12):
            for distortion_factor in np.arange(-0.6, 0.61, 0.1):
                with warnings.catch_warnings(record=True) as w:
                    distortions.distort(
                        self.Int_Cd_2_struc,  # we have 3 Cd at 2.71 Å, 4 Cd at 4.25 Å from the
                        # defect site
                        num_nearest_neighbours=num_neighbours,
                        distortion_factor=distortion_factor,
                        site_index=site_index,
                        distorted_element="Cd",
                    )
                    warning_message = (
                        f"Cd was specified as the nearest neighbour element to "
                        f"distort, with `distortion_factor` {distortion_factor} but did not "
                        f"find `num_nearest_neighbours` ({num_neighbours}) of these elements "
                        f"within 4.5 Å of the defect site. For the remaining neighbours to "
                        f"distort, we ignore the elemental identity. The final distortion "
                        f"information is:"
                    )
                    if (
                        num_neighbours > 7
                    ):  # should only give warning when more than 7 distorted
                        # neighbours requested
                        self.assertEqual(len(w), 1)
                        self.assertEqual(w[0].category, UserWarning)
                        self.assertIn(warning_message, str(w[0].message))
                    else:
                        self.assertEqual(len(w), 0)  # No warning if we distort none

    def test_rattle_V_Cd(self):
        """Test structure rattle function for V_Cd"""
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        d_min = 0.8 * sorted_distances[len(self.V_Cd_struc) + 20]

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

    def test_rattle_Int_Cd_2(self):
        """Test structure rattle function for Int_Cd_2"""
        sorted_distances = np.sort(self.Int_Cd_2_struc.distance_matrix.flatten())
        d_min = 0.8 * sorted_distances[len(self.Int_Cd_2_struc) + 20]

        rattling_atom_indices = np.arange(
            0, 64
        )  # not including index 64 which is Int_Cd_2
        idx = np.in1d(rattling_atom_indices, [i - 1 for i in [10, 22]])
        rattling_atom_indices = rattling_atom_indices[
            ~idx
        ]  # removed distorted Cd indices

        self.assertEqual(
            distortions.rattle(
                self.Int_Cd_2_minus0pt6_struc,
                d_min=d_min,
                active_atoms=rattling_atom_indices,
            ),
            self.Int_Cd_2_minus0pt6_struc_rattled,
        )

    def test_rattle_kwargs(
        self,
    ):  # test all possible kwargs and explicitly check output
        """Test structure rattle function with all possible kwargs"""
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        bulk_bond_length = sorted_distances[len(self.V_Cd_struc) + 20]
        self.assertEqual(bulk_bond_length, 2.8333683853583165)

        V_Cd_kwarg_rattled = distortions.rattle(
            self.V_Cd_minus0pt5_struc,
            stdev=0.15,
            d_min=0.75 * bulk_bond_length,
            nbr_cutoff=3.4,
            n_iter=3,
            active_atoms=rattling_atom_indices,
            width=0.3,
            max_attempts=10000,
            max_disp=1.0,
            seed=20,
        )

        self.assertEqual(V_Cd_kwarg_rattled, self.V_Cd_minus0pt5_struc_kwarged)


if __name__ == "__main__":
    unittest.main()
