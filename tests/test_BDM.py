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
        self.V_Cd_dict = self.cdte_defect_dict["vacancies"][0]
        self.Int_Cd_2_dict = self.cdte_defect_dict["interstitials"][1]

        self.V_Cd_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_0pt1_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Distortion_stdev0pt1_Rattled_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_kwarged = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_V_Cd_-50%_Kwarged_POSCAR")
        )
        self.V_Cd_distortion_parameters = {
            "unique_site": np.array([0.0, 0.0, 0.0]),
            "num_distorted_neighbours": 2,
            "distorted_atoms": [(33, "Te"), (42, "Te")],
        }
        self.Int_Cd_2_struc = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_2_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_Rattled_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
            os.path.join(DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR")
        )
        self.Int_Cd_2_normal_distortion_parameters = {
            "unique_site": self.Int_Cd_2_dict["unique_site"].frac_coords,
            "num_distorted_neighbours": 2,
            "distorted_atoms": [(10, "Cd"), (22, "Cd")],
            "defect_site_index": 65,
        }
        self.Int_Cd_2_NN_10_distortion_parameters = {
            "unique_site": self.Int_Cd_2_dict["unique_site"].frac_coords,
            "num_distorted_neighbours": 10,
            "distorted_atoms":   [
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
            "defect_site_index": 65,
        }

        # Note that Int_Cd_2 has been chosen as a test case, because the first few nonzero bond
        # distances are the interstitial bonds, rather than the bulk bond length, so here we are
        # also testing that the package correctly ignores these and uses the bulk bond length of
        # 2.8333... for d_min in the structure rattling functions.

    def test_update_struct_defect_dict(self):
        """Test update_struct_defect_dict function"""
        vasp_defect_inputs = vasp_input.prepare_vasp_defect_inputs(
            copy.deepcopy(self.cdte_defect_dict)
        )
        for key, struc, comment in [
            ("vac_1_Cd_0", self.V_Cd_struc, "V_Cd Undistorted"),
            ("vac_1_Cd_0", self.V_Cd_minus0pt5_struc_rattled, "V_Cd Rattled"),
            ("vac_1_Cd_-2", self.V_Cd_struc, "V_Cd_-2 Undistorted"),
            ("Int_Cd_2_1", self.Int_Cd_2_minus0pt6_struc_rattled, "Int_Cd_2 Rattled"),
        ]:
            charged_defect_dict = vasp_defect_inputs[key]
            output = BDM.update_struct_defect_dict(charged_defect_dict, struc, comment)
            self.assertEqual(output["Defect Structure"], struc)
            self.assertEqual(output["POSCAR Comment"], comment)
            self.assertDictEqual(
                output["Transformation Dict"],
                charged_defect_dict["Transformation Dict"],
            )

    @patch("builtins.print")
    def test_calc_number_electrons(self, mock_print):
        """Test calc_number_electrons function"""
        oxidation_states = {"Cd": +2, "Te": -2}
        for defect, electron_change in [
            ("vac_1_Cd", -2),
            ("vac_2_Te", 2),
            ("as_1_Cd_on_Te", 4),
            ("as_1_Te_on_Cd", -4),
            ("Int_Cd_2", 2),
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
                                    i,
                                    oxidation_states,
                                    verbose=False,  # test non-verbose
                                ),
                                -electron_change,  # returns negative of electron change
                            )
                            BDM.calc_number_electrons(i, oxidation_states, verbose=True)
                            mock_print.assert_called_with(
                                f"Number of extra/missing electrons of "
                                f"defect {defect}: {electron_change} "
                                f"-> Δq = {-electron_change}"
                            )

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
        V_Cd_distorted_dict = BDM.apply_rattle_bond_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.5,
        )
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.bdm(self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, V_Cd_distorted_dict, output
        )  # Shouldn't match because rattling not done yet
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

    def test_apply_rattle_bond_distortions_Int_Cd_2(self):
        """Test apply_rattle_bond_distortions function for Int_Cd_2"""
        Int_Cd_2_distorted_dict = BDM.apply_rattle_bond_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.4,
        )
        output = distortions.bdm(self.Int_Cd_2_struc, 2, 0.4, site_index=65)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            Int_Cd_2_distorted_dict,
            output,
        )  # Shouldn't match because
        # rattling not done yet
        sorted_distances = np.sort(self.Int_Cd_2_struc.distance_matrix.flatten())
        d_min = 0.85 * sorted_distances[len(self.Int_Cd_2_struc) + 20]
        rattling_atom_indices = np.arange(
            0, 64
        )  # not including index 64 which is Int_Cd_2
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
        np.testing.assert_equal(Int_Cd_2_distorted_dict, output)
        self.assertEqual(
            Int_Cd_2_distorted_dict["distorted_structure"],
            self.Int_Cd_2_minus0pt6_struc_rattled,
        )
        self.assertDictEqual(output, Int_Cd_2_distorted_dict)

    @patch("builtins.print")
    def test_apply_rattle_bond_distortions_kwargs(self, mock_print):
        """Test apply_rattle_bond_distortions function with all possible kwargs"""
        # test distortion kwargs with Int_Cd_2
        Int_Cd_2_distorted_dict = BDM.apply_rattle_bond_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=10,
            distortion_factor=0.4,
            distorted_element="Cd",
            stdev=0,  # no rattling here
            verbose=True,
        )
        self.assertEqual(
            Int_Cd_2_distorted_dict["distorted_structure"],
            self.Int_Cd_2_minus0pt6_NN_10_struc_rattled,
        )
        self.assertEqual(
            Int_Cd_2_distorted_dict["undistorted_structure"], self.Int_Cd_2_struc
        )
        self.assertEqual(Int_Cd_2_distorted_dict["num_distorted_neighbours"], 10)
        self.assertEqual(Int_Cd_2_distorted_dict["defect_site_index"], 65)
        self.assertEqual(Int_Cd_2_distorted_dict.get("defect_frac_coords"), None)
        self.assertCountEqual(
            Int_Cd_2_distorted_dict["distorted_atoms"],
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
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: 65\n"
            + "        Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd'), "
            + "(2.71, 29, 'Cd'), (4.25, 1, 'Cd'), (4.25, 14, 'Cd'), (4.25, 24, 'Cd'), (4.25, 30, "
            + "'Cd'), (2.71, 38, 'Te'), (2.71, 54, 'Te'), (2.71, 62, 'Te')]\n"
            + "        Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd'), "
            + "(1.09, 29, 'Cd'), (1.7, 1, 'Cd'), (1.7, 14, 'Cd'), (1.7, 24, 'Cd'), "
            + "(1.7, 30, 'Cd'), (1.09, 38, 'Te'), (1.09, 54, 'Te'), (1.09, 62, 'Te')]"
        )

        # test all possible rattling kwargs with V_Cd
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates

        V_Cd_kwarg_distorted_dict = BDM.apply_rattle_bond_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.5,
            stdev=0.15,
            d_min=0.75 * 2.8333683853583165,
            nbr_cutoff=3.4,
            n_iter=3,
            active_atoms=rattling_atom_indices,
            width=0.3,
            max_attempts=10000,
            max_disp=1.0,
            seed=20,
        )

        self.assertEqual(
            V_Cd_kwarg_distorted_dict["distorted_structure"],
            self.V_Cd_minus0pt5_struc_kwarged,
        )
        self.assertEqual(
            V_Cd_kwarg_distorted_dict["undistorted_structure"], self.V_Cd_struc
        )
        self.assertEqual(V_Cd_kwarg_distorted_dict["num_distorted_neighbours"], 2)
        self.assertEqual(V_Cd_kwarg_distorted_dict.get("defect_site_index"), None)
        np.testing.assert_array_equal(
            V_Cd_kwarg_distorted_dict.get("defect_frac_coords"), vac_coords
        )

    def test_apply_distortions_V_Cd(self):
        """Test apply_distortions function for V_Cd"""
        V_Cd_distorted_dict = BDM.apply_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.5],
            stdev=0.25,
            verbose=True,
        )
        self.assertDictEqual(self.V_Cd_dict, V_Cd_distorted_dict["Unperturbed_Defect"])

        distorted_V_Cd_struc = V_Cd_distorted_dict["Distortions"][
            "-50.0%_Bond_Distortion"
        ]
        self.assertNotEqual(self.V_Cd_struc, distorted_V_Cd_struc)
        self.assertEqual(self.V_Cd_minus0pt5_struc_rattled, distorted_V_Cd_struc)

        V_Cd_0pt1_distorted_dict = BDM.apply_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.5],
            stdev=0.1,
            verbose=True,
        )
        distorted_V_Cd_struc = V_Cd_0pt1_distorted_dict["Distortions"][
            "-50.0%_Bond_Distortion"
        ]
        self.assertNotEqual(self.V_Cd_struc, distorted_V_Cd_struc)
        self.assertEqual(self.V_Cd_minus0pt5_struc_0pt1_rattled, distorted_V_Cd_struc)

        np.testing.assert_equal(
            V_Cd_distorted_dict["distortion_parameters"], self.V_Cd_distortion_parameters
        )

        V_Cd_3_neighbours_distorted_dict = BDM.apply_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=3,
            bond_distortions=[-0.5],
            stdev=0.25,
            verbose=True,
        )
        V_Cd_3_neighbours_distortion_parameters = self.V_Cd_distortion_parameters.copy()
        V_Cd_3_neighbours_distortion_parameters["num_distorted_neighbours"] = 3
        V_Cd_3_neighbours_distortion_parameters["distorted_atoms"] += [(52, "Te")]
        np.testing.assert_equal(
            V_Cd_3_neighbours_distorted_dict["distortion_parameters"],
            V_Cd_3_neighbours_distortion_parameters,
        )

        with patch("builtins.print") as mock_print:
            distortion_range = np.arange(-0.6, 0.61, 0.1)
            V_Cd_distorted_dict = BDM.apply_distortions(
                self.V_Cd_dict,
                num_nearest_neighbours=2,
                bond_distortions=distortion_range,
                verbose=True,
            )
            prev_struc = V_Cd_distorted_dict["Unperturbed_Defect"]["supercell"][
                "structure"
            ]
            for distortion in distortion_range:
                key = f"{round(distortion,3)+0:.1%}_Bond_Distortion"
                self.assertIn(key, V_Cd_distorted_dict["Distortions"])
                self.assertNotEqual(prev_struc, V_Cd_distorted_dict["Distortions"][key])
                prev_struc = V_Cd_distorted_dict["Distortions"][
                    key
                ]  # different structure for each
                # distortion
                mock_print.assert_any_call(f"--Distortion {round(distortion,3)+0:.1%}")

        # plus some hard-coded checks
        self.assertIn("-60.0%_Bond_Distortion", V_Cd_distorted_dict["Distortions"])
        self.assertIn("60.0%_Bond_Distortion", V_Cd_distorted_dict["Distortions"])
        # test zero distortion is written as positive zero (not "-0.0%")
        self.assertIn("0.0%_Bond_Distortion", V_Cd_distorted_dict["Distortions"])

    def test_apply_distortions_Int_Cd_2(self):
        """Test apply_distortions function for Int_Cd_2"""
        Int_Cd_2_distorted_dict = BDM.apply_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.6],
            stdev=0.25,
            verbose=True,
        )
        self.assertDictEqual(
            self.Int_Cd_2_dict, Int_Cd_2_distorted_dict["Unperturbed_Defect"]
        )
        distorted_Int_Cd_2_struc = Int_Cd_2_distorted_dict["Distortions"][
            "-60.0%_Bond_Distortion"
        ]
        self.assertNotEqual(self.Int_Cd_2_struc, distorted_Int_Cd_2_struc)
        self.assertEqual(
            self.Int_Cd_2_minus0pt6_struc_rattled, distorted_Int_Cd_2_struc
        )

        np.testing.assert_equal(
            Int_Cd_2_distorted_dict["distortion_parameters"],
            self.Int_Cd_2_normal_distortion_parameters,
        )

    @patch("builtins.print")
    def test_apply_distortions_kwargs(self, mock_print):
        """Test apply_rattle_bond_distortions function with all possible kwargs"""
        # test distortion kwargs with Int_Cd_2
        Int_Cd_2_distorted_dict = BDM.apply_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=10,
            bond_distortions=[-0.6],
            distorted_element="Cd",
            stdev=0,  # no rattling here
            verbose=True,
        )
        self.assertDictEqual(
            self.Int_Cd_2_dict, Int_Cd_2_distorted_dict["Unperturbed_Defect"]
        )
        distorted_Int_Cd_2_struc = Int_Cd_2_distorted_dict["Distortions"][
            "-60.0%_Bond_Distortion"
        ]
        self.assertNotEqual(self.Int_Cd_2_struc, distorted_Int_Cd_2_struc)
        self.assertEqual(
            self.Int_Cd_2_minus0pt6_NN_10_struc_rattled, distorted_Int_Cd_2_struc
        )
        np.testing.assert_equal(
            Int_Cd_2_distorted_dict["distortion_parameters"],
            self.Int_Cd_2_NN_10_distortion_parameters,
        )
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: 65\n"
            + "        Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd'), "
            + "(2.71, 29, 'Cd'), (4.25, 1, 'Cd'), (4.25, 14, 'Cd'), (4.25, 24, 'Cd'), (4.25, 30, "
            + "'Cd'), (2.71, 38, 'Te'), (2.71, 54, 'Te'), (2.71, 62, 'Te')]\n"
            + "        Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd'), "
            + "(1.09, 29, 'Cd'), (1.7, 1, 'Cd'), (1.7, 14, 'Cd'), (1.7, 24, 'Cd'), "
            + "(1.7, 30, 'Cd'), (1.09, 38, 'Te'), (1.09, 54, 'Te'), (1.09, 62, 'Te')]"
        )

        # test all possible rattling kwargs with V_Cd
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates

        V_Cd_kwarg_distorted_dict = BDM.apply_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.5],
            stdev=0.15,
            d_min=0.75 * 2.8333683853583165,
            nbr_cutoff=3.4,
            n_iter=3,
            active_atoms=rattling_atom_indices,
            width=0.3,
            max_attempts=10000,
            max_disp=1.0,
            seed=20,
            verbose=True,
        )
        self.assertDictEqual(self.V_Cd_dict, V_Cd_kwarg_distorted_dict["Unperturbed_Defect"])
        distorted_V_Cd_struc = V_Cd_kwarg_distorted_dict["Distortions"][
            "-50.0%_Bond_Distortion"
        ]
        self.assertNotEqual(self.V_Cd_struc, distorted_V_Cd_struc)
        self.assertEqual(self.V_Cd_minus0pt5_struc_kwarged, distorted_V_Cd_struc)
        np.testing.assert_equal(
            V_Cd_kwarg_distorted_dict["distortion_parameters"], self.V_Cd_distortion_parameters
        )


if __name__ == "__main__":
    unittest.main()
