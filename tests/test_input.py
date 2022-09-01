import unittest
import os
import pickle
import copy
from unittest.mock import patch
import shutil
import numpy as np
import json

from pymatgen.core.structure import Structure, Composition
from pymatgen.io.vasp.inputs import Poscar

from ase.calculators.aims import Aims

from shakenbreak import input, io, distortions, vasp


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def _update_struct_defect_dict(
    defect_dict: dict, structure: Structure, poscar_comment: str
) -> dict:
    """
    Given a Structure object and POSCAR comment, update the folders dictionary (generated with
    `doped.vasp_input.prepare_vasp_defect_inputs()`) with the given values.
    Args:
        defect_dict (:obj:`dict`):
            Dictionary with defect information, as generated with doped prepare_vasp_defect_inputs()
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a pymatgen object
        poscar_comment (:obj:`str`):
            Comment to include in the top line of the POSCAR file
    Returns:
        single defect dict in the `doped` format.
    """
    defect_dict_copy = copy.deepcopy(defect_dict)
    defect_dict_copy["Defect Structure"] = structure
    defect_dict_copy["POSCAR Comment"] = poscar_comment
    return defect_dict_copy


class InputTestCase(unittest.TestCase):
    """Test ShakeNBreak structure distortion helper functions"""

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.CASTEP_DATA_DIR = os.path.join(self.DATA_DIR, "castep")
        self.CP2K_DATA_DIR = os.path.join(self.DATA_DIR, "cp2k")
        self.FHI_AIMS_DATA_DIR = os.path.join(self.DATA_DIR, "fhi_aims")
        self.ESPRESSO_DATA_DIR = os.path.join(self.DATA_DIR, "quantum_espresso")

        with open(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.pickle"), "rb"
        ) as fp:
            self.cdte_defect_dict = pickle.load(fp)
        with open(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_extrinsic_defects_dict.pickle"),
            "rb",
        ) as fp:
            self.cdte_extrinsic_defects_dict = pickle.load(fp)
        self.V_Cd_dict = self.cdte_defect_dict["vacancies"][0]
        self.Int_Cd_2_dict = self.cdte_defect_dict["interstitials"][1]

        self.V_Cd_struc = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR"
            )
        )
        self.V_Cd_minus0pt5_struc_0pt1_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_V_Cd_-50%_Distortion_stdev0pt1_Rattled_POSCAR",
            )
        )
        self.V_Cd_minus0pt5_struc_kwarged = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Kwarged_POSCAR")
        )
        self.V_Cd_distortion_parameters = {
            "unique_site": np.array([0.0, 0.0, 0.0]),
            "num_distorted_neighbours": 2,
            "distorted_atoms": [(33, "Te"), (42, "Te")],
        }
        self.Int_Cd_2_struc = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_Rattled_POSCAR"
            )
        )
        self.Int_Cd_2_minus0pt6_NN_10_struc_unrattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR"
            )
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
            "distorted_atoms": [
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

        self.cdte_defect_folders = [
            "as_1_Cd_on_Te_-1",
            "as_1_Cd_on_Te_-2",
            "as_1_Cd_on_Te_0",
            "as_1_Cd_on_Te_1",
            "as_1_Cd_on_Te_2",
            "as_1_Cd_on_Te_3",
            "as_1_Cd_on_Te_4",
            "as_1_Te_on_Cd_-1",
            "as_1_Te_on_Cd_-2",
            "as_1_Te_on_Cd_0",
            "as_1_Te_on_Cd_1",
            "as_1_Te_on_Cd_2",
            "as_1_Te_on_Cd_3",
            "as_1_Te_on_Cd_4",
            "Int_Cd_1_0",
            "Int_Cd_1_1",
            "Int_Cd_1_2",
            "Int_Cd_2_0",
            "Int_Cd_2_1",
            "Int_Cd_2_2",
            "Int_Cd_3_0",
            "Int_Cd_3_1",
            "Int_Cd_3_2",
            "Int_Te_1_-1",
            "Int_Te_1_-2",
            "Int_Te_1_0",
            "Int_Te_1_1",
            "Int_Te_1_2",
            "Int_Te_1_3",
            "Int_Te_1_4",
            "Int_Te_1_5",
            "Int_Te_1_6",
            "Int_Te_2_-1",
            "Int_Te_2_-2",
            "Int_Te_2_0",
            "Int_Te_2_1",
            "Int_Te_2_2",
            "Int_Te_2_3",
            "Int_Te_2_4",
            "Int_Te_2_5",
            "Int_Te_2_6",
            "Int_Te_3_-1",
            "Int_Te_3_-2",
            "Int_Te_3_0",
            "Int_Te_3_1",
            "Int_Te_3_2",
            "Int_Te_3_3",
            "Int_Te_3_4",
            "Int_Te_3_5",
            "Int_Te_3_6",
            "vac_1_Cd_-1",
            "vac_1_Cd_-2",
            "vac_1_Cd_0",
            "vac_1_Cd_1",
            "vac_1_Cd_2",
            "vac_2_Te_-1",
            "vac_2_Te_-2",
            "vac_2_Te_0",
            "vac_2_Te_1",
            "vac_2_Te_2",
        ]

    def tearDown(self) -> None:
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders if present
        for fname in os.listdir("./"):
            if fname.startswith("distortion_metadata"):
                os.remove(f"./{fname}")
        if_present_rm("test_path")  # remove test_path if present

    def test_get_bulk_comp(self):
        V_Cd_comp = input._get_bulk_comp(self.V_Cd_dict)
        self.assertEqual(V_Cd_comp, Composition("Cd32Te32"))

        Int_Cd_comp = input._get_bulk_comp(self.Int_Cd_2_dict)
        self.assertEqual(Int_Cd_comp, Composition("Cd32Te32"))

    def test_most_common_oxi(self):
        self.assertEqual(input._most_common_oxi("Cd"), +2)
        self.assertEqual(input._most_common_oxi("Te"), -2)
        self.assertEqual(input._most_common_oxi("Cl"), -1)
        self.assertEqual(input._most_common_oxi("Al"), +3)
        self.assertEqual(input._most_common_oxi("Mg"), +2)
        self.assertEqual(input._most_common_oxi("Si"), +4)
        self.assertEqual(input._most_common_oxi("Ca"), +2)
        self.assertEqual(input._most_common_oxi("Fe"), +3)
        self.assertEqual(input._most_common_oxi("Ni"), 0)
        self.assertEqual(input._most_common_oxi("Cu"), +1)
        self.assertEqual(input._most_common_oxi("Ag"), +1)
        self.assertEqual(input._most_common_oxi("Zn"), +2)
        self.assertEqual(input._most_common_oxi("Pb"), +2)
        self.assertEqual(input._most_common_oxi("Hg"), +2)
        self.assertEqual(input._most_common_oxi("O"), -2)
        self.assertEqual(input._most_common_oxi("S"), -2)
        self.assertEqual(input._most_common_oxi("Se"), -2)
        self.assertEqual(input._most_common_oxi("N"), -3)

    @patch("builtins.print")
    def test_calc_number_electrons(self, mock_print):
        """Test _calc_number_electrons function"""
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
                                input._calc_number_electrons(
                                    i,
                                    oxidation_states,
                                    verbose=False,  # test non-verbose
                                ),
                                -electron_change,  # returns negative of electron change
                            )
                            input._calc_number_electrons(
                                i, oxidation_states, verbose=True
                            )
                            mock_print.assert_called_with(
                                f"Number of extra/missing electrons of "
                                f"defect {defect}: {electron_change} "
                                f"-> Δq = {-electron_change}"
                            )

    def test_calc_number_neighbours(self):
        """Test _calc_number_neighbours function"""
        self.assertEqual(input._calc_number_neighbours(0), 0)
        self.assertEqual(input._calc_number_neighbours(-2), 2)
        self.assertEqual(input._calc_number_neighbours(2), 2)
        self.assertEqual(input._calc_number_neighbours(6), 2)
        self.assertEqual(input._calc_number_neighbours(-6), 2)
        self.assertEqual(input._calc_number_neighbours(8), 0)
        self.assertEqual(input._calc_number_neighbours(-8), 0)
        self.assertEqual(input._calc_number_neighbours(4), 4)
        self.assertEqual(input._calc_number_neighbours(-4), 4)

    def test_apply_rattle_bond_distortions_V_Cd(self):
        """Test _apply_rattle_bond_distortions function for V_Cd"""
        V_Cd_distorted_dict = input._apply_rattle_bond_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.5,
        )
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.distort(self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, V_Cd_distorted_dict, output
        )  # Shouldn't match because rattling not done yet
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        d_min = 0.8 * sorted_distances[len(self.V_Cd_struc) + 20]
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
        """Test _apply_rattle_bond_distortions function for Int_Cd_2"""
        Int_Cd_2_distorted_dict = input._apply_rattle_bond_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=2,
            distortion_factor=0.4,
        )
        output = distortions.distort(self.Int_Cd_2_struc, 2, 0.4, site_index=65)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            Int_Cd_2_distorted_dict,
            output,
        )  # Shouldn't match because
        # rattling not done yet
        sorted_distances = np.sort(self.Int_Cd_2_struc.distance_matrix.flatten())
        d_min = 0.8 * sorted_distances[len(self.Int_Cd_2_struc) + 20]
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
        """Test _apply_rattle_bond_distortions function with all possible kwargs"""
        # test distortion kwargs with Int_Cd_2
        Int_Cd_2_distorted_dict = input._apply_rattle_bond_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=10,
            distortion_factor=0.4,
            distorted_element="Cd",
            stdev=0,  # no rattling here
            verbose=True,
        )
        self.assertEqual(
            Int_Cd_2_distorted_dict["distorted_structure"],
            self.Int_Cd_2_minus0pt6_NN_10_struc_unrattled,
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
            + "            Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd'), "
            + "(2.71, 29, 'Cd'), (4.25, 1, 'Cd'), (4.25, 14, 'Cd'), (4.25, 24, 'Cd'), (4.25, 30, "
            + "'Cd'), (2.71, 38, 'Te'), (2.71, 54, 'Te'), (2.71, 62, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd'), "
            + "(1.09, 29, 'Cd'), (1.7, 1, 'Cd'), (1.7, 14, 'Cd'), (1.7, 24, 'Cd'), "
            + "(1.7, 30, 'Cd'), (1.09, 38, 'Te'), (1.09, 54, 'Te'), (1.09, 62, 'Te')]"
        )

        # test all possible rattling kwargs with V_Cd
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates

        V_Cd_kwarg_distorted_dict = input._apply_rattle_bond_distortions(
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

    def test_apply_snb_distortions_V_Cd(self):
        """Test apply_distortions function for V_Cd"""
        V_Cd_distorted_dict = input.apply_snb_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.5],
            stdev=0.25,
            verbose=True,
        )
        self.assertDictEqual(self.V_Cd_dict, V_Cd_distorted_dict["Unperturbed"])

        distorted_V_Cd_struc = V_Cd_distorted_dict["distortions"][
            "Bond_Distortion_-50.0%"
        ]
        self.assertNotEqual(self.V_Cd_struc, distorted_V_Cd_struc)
        self.assertEqual(self.V_Cd_minus0pt5_struc_rattled, distorted_V_Cd_struc)

        V_Cd_0pt1_distorted_dict = input.apply_snb_distortions(
            self.V_Cd_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.5],
            stdev=0.1,
            verbose=True,
        )
        distorted_V_Cd_struc = V_Cd_0pt1_distorted_dict["distortions"][
            "Bond_Distortion_-50.0%"
        ]
        self.assertNotEqual(self.V_Cd_struc, distorted_V_Cd_struc)
        self.assertEqual(self.V_Cd_minus0pt5_struc_0pt1_rattled, distorted_V_Cd_struc)

        np.testing.assert_equal(
            V_Cd_distorted_dict["distortion_parameters"],
            self.V_Cd_distortion_parameters,
        )

        V_Cd_3_neighbours_distorted_dict = input.apply_snb_distortions(
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
            V_Cd_distorted_dict = input.apply_snb_distortions(
                self.V_Cd_dict,
                num_nearest_neighbours=2,
                bond_distortions=distortion_range,
                verbose=True,
            )
            prev_struc = V_Cd_distorted_dict["Unperturbed"]["supercell"]["structure"]
            for distortion in distortion_range:
                key = f"Bond_Distortion_{round(distortion,3)+0:.1%}"
                self.assertIn(key, V_Cd_distorted_dict["distortions"])
                self.assertNotEqual(prev_struc, V_Cd_distorted_dict["distortions"][key])
                prev_struc = V_Cd_distorted_dict["distortions"][
                    key
                ]  # different structure for each
                # distortion
                mock_print.assert_any_call(f"--Distortion {round(distortion,3)+0:.1%}")

        # plus some hard-coded checks
        self.assertIn("Bond_Distortion_-60.0%", V_Cd_distorted_dict["distortions"])
        self.assertIn("Bond_Distortion_60.0%", V_Cd_distorted_dict["distortions"])
        # test zero distortion is written as positive zero (not "-0.0%")
        self.assertIn("Bond_Distortion_0.0%", V_Cd_distorted_dict["distortions"])

    def test_apply_snb_distortions_Int_Cd_2(self):

        """Test apply_distortions function for Int_Cd_2"""
        Int_Cd_2_distorted_dict = input.apply_snb_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=2,
            bond_distortions=[-0.6],
            stdev=0.25,
            verbose=True,
        )
        self.assertDictEqual(self.Int_Cd_2_dict, Int_Cd_2_distorted_dict["Unperturbed"])
        distorted_Int_Cd_2_struc = Int_Cd_2_distorted_dict["distortions"][
            "Bond_Distortion_-60.0%"
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
    def test_apply_snb_distortions_kwargs(self, mock_print):
        """Test _apply_rattle_bond_distortions function with all possible kwargs"""
        # test distortion kwargs with Int_Cd_2
        Int_Cd_2_distorted_dict = input.apply_snb_distortions(
            self.Int_Cd_2_dict,
            num_nearest_neighbours=10,
            bond_distortions=[-0.6],
            distorted_element="Cd",
            stdev=0,  # no rattling here
            verbose=True,
        )
        self.assertDictEqual(self.Int_Cd_2_dict, Int_Cd_2_distorted_dict["Unperturbed"])
        distorted_Int_Cd_2_struc = Int_Cd_2_distorted_dict["distortions"][
            "Bond_Distortion_-60.0%"
        ]
        self.assertNotEqual(self.Int_Cd_2_struc, distorted_Int_Cd_2_struc)
        self.assertEqual(
            self.Int_Cd_2_minus0pt6_NN_10_struc_unrattled, distorted_Int_Cd_2_struc
        )
        np.testing.assert_equal(
            Int_Cd_2_distorted_dict["distortion_parameters"],
            self.Int_Cd_2_NN_10_distortion_parameters,
        )
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: 65\n"
            + "            Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd'), "
            + "(2.71, 29, 'Cd'), (4.25, 1, 'Cd'), (4.25, 14, 'Cd'), (4.25, 24, 'Cd'), (4.25, 30, "
            + "'Cd'), (2.71, 38, 'Te'), (2.71, 54, 'Te'), (2.71, 62, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd'), "
            + "(1.09, 29, 'Cd'), (1.7, 1, 'Cd'), (1.7, 14, 'Cd'), (1.7, 24, 'Cd'), "
            + "(1.7, 30, 'Cd'), (1.09, 38, 'Te'), (1.09, 54, 'Te'), (1.09, 62, 'Te')]"
        )

        # test all possible rattling kwargs with V_Cd
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates

        V_Cd_kwarg_distorted_dict = input.apply_snb_distortions(
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
        self.assertDictEqual(self.V_Cd_dict, V_Cd_kwarg_distorted_dict["Unperturbed"])
        distorted_V_Cd_struc = V_Cd_kwarg_distorted_dict["distortions"][
            "Bond_Distortion_-50.0%"
        ]
        self.assertNotEqual(self.V_Cd_struc, distorted_V_Cd_struc)
        self.assertEqual(self.V_Cd_minus0pt5_struc_kwarged, distorted_V_Cd_struc)
        np.testing.assert_equal(
            V_Cd_kwarg_distorted_dict["distortion_parameters"],
            self.V_Cd_distortion_parameters,
        )

    # test create_folder and create_vasp_input simultaneously:
    def test_create_vasp_input(self):
        """Test create_vasp_input function"""
        # Create doped/PyCDT-style defect dict:
        supercell = self.V_Cd_dict["supercell"]
        V_Cd_defect_relax_set = vasp.DefectRelaxSet(supercell["structure"], charge=0)
        poscar = V_Cd_defect_relax_set.poscar
        struct = V_Cd_defect_relax_set.structure
        dict_transf = {
            "defect_type": self.V_Cd_dict["name"],
            "defect_site": self.V_Cd_dict["unique_site"],
            "defect_supercell_site": self.V_Cd_dict["bulk_supercell_site"],
            "defect_multiplicity": self.V_Cd_dict["site_multiplicity"],
            "charge": 0,
            "supercell": supercell["size"],
        }
        poscar.comment = (
            self.V_Cd_dict["name"]
            + str(dict_transf["defect_supercell_site"].frac_coords)
            + "_-dNELECT="  # change in NELECT from bulk supercell
            + str(0)
        )
        vasp_defect_inputs = {
            "vac_1_Cd_0": {
                "Defect Structure": struct,
                "POSCAR Comment": poscar.comment,
                "Transformation Dict": dict_transf,
            }
        }
        V_Cd_updated_charged_defect_dict = _update_struct_defect_dict(
            vasp_defect_inputs["vac_1_Cd_0"],
            self.V_Cd_minus0pt5_struc_rattled,
            "V_Cd Rattled",
        )
        V_Cd_charged_defect_dict = {
            "Bond_Distortion_-50.0%": V_Cd_updated_charged_defect_dict
        }
        self.assertFalse(os.path.exists("vac_1_Cd_0"))
        input._create_vasp_input(
            "vac_1_Cd_0",
            distorted_defect_dict=V_Cd_charged_defect_dict,
            incar_settings=vasp.default_incar_settings,
        )
        V_Cd_Bond_Distortion_folder = "vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_Bond_Distortion_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_Bond_Distortion_folder + "/POSCAR")
        self.assertEqual(V_Cd_POSCAR.comment, "V_Cd Rattled")
        self.assertEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)
        # only test POSCAR as INCAR, KPOINTS and POTCAR not written on GitHub actions,
        # but tested locally

        # test with kwargs: (except POTCAR settings because we can't have this on the GitHub test
        # server)
        kwarg_incar_settings = {
            "NELECT": 3,
            "IBRION": 42,
            "LVHAR": True,
            "LWAVE": True,
            "LCHARG": True,
        }
        kwarged_incar_settings = vasp.default_incar_settings.copy()
        kwarged_incar_settings.update(kwarg_incar_settings)
        input._create_vasp_input(
            "vac_1_Cd_0",
            distorted_defect_dict=V_Cd_charged_defect_dict,
            incar_settings=kwarged_incar_settings,
        )
        V_Cd_kwarg_folder = "vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_kwarg_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_kwarg_folder + "/POSCAR")
        self.assertEqual(V_Cd_POSCAR.comment, "V_Cd Rattled")
        self.assertEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)
        # only test POSCAR as INCAR, KPOINTS and POTCAR not written on GitHub actions,
        # but tested locally

        # test output_path option
        input._create_vasp_input(
            "vac_1_Cd_0",
            distorted_defect_dict=V_Cd_charged_defect_dict,
            incar_settings=kwarged_incar_settings,
            output_path="test_path",
        )
        V_Cd_kwarg_folder = "test_path/vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_kwarg_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_kwarg_folder + "/POSCAR")
        self.assertEqual(V_Cd_POSCAR.comment, "V_Cd Rattled")
        self.assertEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)

    def test_Distortions_initialisation(self):
        # test auto oxidation state determination:
        for defect_dict in [
            self.cdte_defect_dict,
            {"vacancies": [self.V_Cd_dict]},
            {"interstitials": [self.Int_Cd_2_dict]},
        ]:
            dist = input.Distortions(defect_dict)
            self.assertEqual(dist.oxidation_states, {"Cd": +2, "Te": -2})

        extrinsic_dist = input.Distortions(
            self.cdte_extrinsic_defects_dict,
        )
        self.assertEqual(
            extrinsic_dist.oxidation_states,
            {
                "Cd": 2.0,
                "Te": -2.0,
                "Zn": 2.0,
                "Mn": 2.0,
                "Al": 3.0,
                "Sb": 0.0,
                "Cl": -1,
            },
        )

    @patch("builtins.print")
    def test_write_vasp_files(self, mock_print):
        """Test `write_vasp_files` methods"""
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = list(np.arange(-0.6, 0.601, 0.05))

        dist = input.Distortions(
            self.cdte_defect_dict,
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,
        )
        # Test `write_vasp_files` method
        _, distortion_metadata = dist.write_vasp_files(
            incar_settings={"ENCUT": 212, "IBRION": 0, "EDIFF": 1e-4},
            verbose=False,
        )

        # check if expected folders were created:
        self.assertTrue(set(self.cdte_defect_folders).issubset(set(os.listdir())))
        # check expected info printing:
        mock_print.assert_any_call(
            "Applying ShakeNBreak...",
            "Will apply the following bond distortions:",
            "['-0.6', '-0.55', '-0.5', '-0.45', '-0.4', '-0.35', '-0.3', "
            "'-0.25', '-0.2', '-0.15', '-0.1', '-0.05', '0.0', '0.05', "
            "'0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', "
            "'0.5', '0.55', '0.6'].",
            "Then, will rattle with a std dev of 0.25 Å \n",
        )
        mock_print.assert_any_call(
            "\033[1m" + "\nDefect: vac_1_Cd" + "\033[0m"
        )  # bold print
        mock_print.assert_any_call(
            "\033[1m" + "Number of missing electrons in neutral state: 2" + "\033[0m"
        )
        mock_print.assert_any_call(
            "\nDefect vac_1_Cd in charge state: -2. Number of distorted "
            "neighbours: 0"
        )
        mock_print.assert_any_call(
            "\nDefect vac_1_Cd in charge state: -1. Number of distorted "
            "neighbours: 1"
        )
        mock_print.assert_any_call(
            "\nDefect vac_1_Cd in charge state: 0. Number of distorted " "neighbours: 2"
        )
        # test correct distorted neighbours based on oxidation states:
        mock_print.assert_any_call(
            "\nDefect vac_2_Te in charge state: -2. Number of distorted "
            "neighbours: 4"
        )
        mock_print.assert_any_call(
            "\nDefect as_1_Cd_on_Te in charge state: -2. Number of "
            "distorted neighbours: 2"
        )
        mock_print.assert_any_call(
            "\nDefect as_1_Te_on_Cd in charge state: -2. Number of "
            "distorted neighbours: 2"
        )
        mock_print.assert_any_call(
            "\nDefect Int_Cd_1 in charge state: 0. Number of distorted " "neighbours: 2"
        )
        mock_print.assert_any_call(
            "\nDefect Int_Te_1 in charge state: -2. Number of distorted "
            "neighbours: 0"
        )

        # check if correct files were created:
        V_Cd_Bond_Distortion_folder = "vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_Bond_Distortion_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_Bond_Distortion_folder + "/POSCAR")
        self.assertEqual(
            V_Cd_POSCAR.comment,
            "-50.0%__num_neighbours=2_vac_1_Cd",
        )  # default
        self.assertEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)
        # only test POSCAR as INCAR, KPOINTS and POTCAR not written on GitHub actions,
        # but tested locally

        Int_Cd_2_Bond_Distortion_folder = "Int_Cd_2_0/Bond_Distortion_-60.0%"
        self.assertTrue(os.path.exists(Int_Cd_2_Bond_Distortion_folder))
        Int_Cd_2_POSCAR = Poscar.from_file(Int_Cd_2_Bond_Distortion_folder + "/POSCAR")
        self.assertEqual(
            Int_Cd_2_POSCAR.comment,
            "-60.0%__num_neighbours=2_Int_Cd_2",
        )
        self.assertEqual(
            Int_Cd_2_POSCAR.structure, self.Int_Cd_2_minus0pt6_struc_rattled
        )
        # only test POSCAR as INCAR, KPOINTS and POTCAR not written on GitHub actions,
        # but tested locally

        # test rattle kwargs:
        reduced_V_Cd_dict = self.V_Cd_dict.copy()
        reduced_V_Cd_dict["charges"] = [0]
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        dist = input.Distortions(
            {"vacancies": [reduced_V_Cd_dict]},
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            stdev=0.15,
            d_min=0.75 * 2.8333683853583165,
            nbr_cutoff=3.4,
            n_iter=3,
            active_atoms=rattling_atom_indices,
            width=0.3,
            max_attempts=10000,
            max_disp=1.0,
            seed=20,
            local_rattle=False,
        )
        _, distortion_metadata = dist.write_vasp_files(
            verbose=False,
        )
        V_Cd_kwarged_POSCAR = Poscar.from_file(
            "vac_1_Cd_0/Bond_Distortion_-50.0%/POSCAR"
        )
        self.assertEqual(
            V_Cd_kwarged_POSCAR.structure, self.V_Cd_minus0pt5_struc_kwarged
        )
        rounded_bond_distortions = np.around(bond_distortions, 3)
        np.testing.assert_equal(
            distortion_metadata["distortion_parameters"]["bond_distortions"],
            rounded_bond_distortions,
        )

        # test other kwargs:
        reduced_Int_Cd_2_dict = self.Int_Cd_2_dict.copy()
        reduced_Int_Cd_2_dict["charges"] = [1]

        with patch("builtins.print") as mock_Int_Cd_2_print:
            dist = input.Distortions(
                {"interstitials": [reduced_Int_Cd_2_dict]},
                oxidation_states=oxidation_states,
                distortion_increment=0.25,
                distorted_elements={"Int_Cd_2": ["Cd"]},
                dict_number_electrons_user={"Int_Cd_2": 3},
                local_rattle=False,
            )
            _, distortion_metadata = dist.write_vasp_files(
                verbose=True,
            )

            kwarged_Int_Cd_2_dict = {
                "distortion_parameters": {
                    "distortion_increment": 0.25,
                    "bond_distortions": [-0.5, -0.25, 0.0, 0.25, 0.5],
                    "rattle_stdev": 0.25,
                    "local_rattle": False,
                },
                "defects": {
                    "Int_Cd_2": {
                        "unique_site": reduced_Int_Cd_2_dict[
                            "bulk_supercell_site"
                        ].frac_coords,
                        "charges": {
                            1: {
                                "num_nearest_neighbours": 4,
                                "distorted_atoms": [
                                    (10, "Cd"),
                                    (22, "Cd"),
                                    (29, "Cd"),
                                    (38, "Te"),
                                ],
                                "distortion_parameters": {
                                    "bond_distortions": [
                                        -0.5,
                                        -0.25,
                                        0.0,
                                        0.25,
                                        0.5,
                                    ],
                                    "rattle_stdev": 0.25,
                                },
                            },
                        },
                        "defect_site_index": 65,
                    },
                    "vac_1_Cd": {
                        "unique_site": [0.0, 0.0, 0.0],
                        "charges": {
                            0: {
                                "num_nearest_neighbours": 2,
                                "distorted_atoms": [[33, "Te"], [42, "Te"]],
                                "distortion_parameters": {
                                    "bond_distortions": [
                                        -0.6,
                                        -0.55,
                                        -0.5,
                                        -0.45,
                                        -0.4,
                                        -0.35,
                                        -0.3,
                                        -0.25,
                                        -0.2,
                                        -0.15,
                                        -0.1,
                                        -0.05,
                                        0.0,
                                        0.05,
                                        0.1,
                                        0.15,
                                        0.2,
                                        0.25,
                                        0.3,
                                        0.35,
                                        0.4,
                                        0.45,
                                        0.5,
                                        0.55,
                                        0.6,
                                    ],
                                    "rattle_stdev": 0.15,
                                },
                            },
                        },
                    },
                },
            }
            np.testing.assert_equal(
                distortion_metadata["defects"]["Int_Cd_2"]["charges"][
                    1
                ],  # check defect in distortion_defect_dict
                kwarged_Int_Cd_2_dict["defects"]["Int_Cd_2"]["charges"][1],
            )
            self.assertTrue(os.path.exists("distortion_metadata.json"))
            # check defects from old metadata file are in new metadata file
            with open(f"distortion_metadata.json", "r") as metadata_file:
                metadata = json.load(metadata_file)
            for defect in metadata["defects"].values():
                defect["charges"] = {int(k): v for k, v in defect["charges"].items()}
                # json converts integer keys to strings
            metadata["defects"]["Int_Cd_2"]["charges"][1]["distorted_atoms"] = [
                tuple(x)
                for x in metadata["defects"]["Int_Cd_2"]["charges"][1][
                    "distorted_atoms"
                ]
            ]
            np.testing.assert_equal(
                metadata,  # check defect in distortion_defect_dict
                kwarged_Int_Cd_2_dict,
            )

            # check expected info printing:
            mock_Int_Cd_2_print.assert_any_call(
                "Applying ShakeNBreak...",
                "Will apply the following bond distortions:",
                "['-0.5', '-0.25', '0.0', '0.25', '0.5'].",
                "Then, will rattle with a std dev of 0.25 Å \n",
            )
            mock_Int_Cd_2_print.assert_any_call(
                "\033[1m" + "\nDefect: Int_Cd_2" + "\033[0m"
            )
            mock_Int_Cd_2_print.assert_any_call(
                "\033[1m"
                + "Number of missing electrons in neutral state: 3"
                + "\033[0m"
            )
            mock_Int_Cd_2_print.assert_any_call(
                "\nDefect Int_Cd_2 in charge state: +1. Number of distorted neighbours: 4"
            )
            mock_Int_Cd_2_print.assert_any_call("--Distortion -50.0%")
            mock_Int_Cd_2_print.assert_any_call(
                f"\tDefect Site Index / Frac Coords: 65\n"
                + "            Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd'), "
                + "(2.71, 29, 'Cd'), (2.71, 38, 'Te')]\n"
                + "            Distorted Neighbour Distances:\n\t[(1.36, 10, 'Cd'), (1.36, 22, 'Cd'), "
                + "(1.36, 29, 'Cd'), (1.36, 38, 'Te')]"
            )
            # check correct folder was created:
            self.assertTrue(os.path.exists("Int_Cd_2_1/Unperturbed"))

        # check correct output for "extra" electrons and positive charge state:
        with patch("builtins.print") as mock_Int_Cd_2_print:
            dist = input.Distortions(
                {"interstitials": [reduced_Int_Cd_2_dict]},
                oxidation_states=oxidation_states,
                local_rattle=False,
            )
            _, distortion_metadata = dist.write_vasp_files(
                verbose=True,
            )
            self.assertTrue(os.path.exists("distortion_metadata.json"))
            # check expected info printing:
            mock_Int_Cd_2_print.assert_any_call(
                "\033[1m" + "\nDefect: Int_Cd_2" + "\033[0m"
            )
            mock_Int_Cd_2_print.assert_any_call(
                "\033[1m" + "Number of extra electrons in neutral state: 2" + "\033[0m"
            )
            mock_Int_Cd_2_print.assert_any_call(
                "\nDefect Int_Cd_2 in charge state: +1. Number of distorted neighbours: 1"
            )

        # test output_path parameter:
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        dist = input.Distortions(
            {"vacancies": [reduced_V_Cd_dict]},
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            stdev=0.15,
            d_min=0.75 * 2.8333683853583165,
            nbr_cutoff=3.4,
            n_iter=3,
            active_atoms=rattling_atom_indices,
            width=0.3,
            max_attempts=10000,
            max_disp=1.0,
            seed=20,
            local_rattle=False,
        )
        _, distortion_metadata = dist.write_vasp_files(
            output_path="test_path",
            verbose=False,
        )
        self.assertTrue(os.path.exists("test_path/vac_1_Cd_0/Bond_Distortion_-50.0%"))
        self.assertTrue(os.path.exists("test_path/distortion_metadata.json"))
        V_Cd_kwarged_POSCAR = Poscar.from_file(
            "test_path/vac_1_Cd_0/Bond_Distortion_-50.0%/POSCAR"
        )
        self.assertEqual(
            V_Cd_kwarged_POSCAR.structure, self.V_Cd_minus0pt5_struc_kwarged
        )

    @patch("builtins.print")
    def test_write_espresso_files(self, mock_print):
        """Test method write_espresso_files"""
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = [
            0.3,
        ]

        Dist = input.Distortions(
            {
                "vacancies": [
                    self.V_Cd_dict,
                ]
            },
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,
        )

        # Test `write_espresso_files` method
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        pseudopotentials = {  # Your chosen pseudopotentials
            "Cd": "Cd_pbe_v1.uspp.F.UPF",
            "Te": "Te.pbe-n-rrkjus_psl.1.0.0.UPF",
        }
        _, _ = Dist.write_espresso_files(
            pseudopotentials=pseudopotentials,
        )
        self.assertTrue(os.path.exists("vac_1_Cd_0/Unperturbed"))
        with open(
            os.path.join(
                self.ESPRESSO_DATA_DIR, "vac_1_Cd_0/Bond_Distortion_30.0%/espresso.pwi"
            )
        ) as f:
            test_input = f.read()
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/espresso.pwi") as f:
            generated_input = f.read()
        self.assertEqual(test_input, generated_input)

        # Test parameter file is not written if write_structures_only = True
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_espresso_files(write_structures_only=True)
        with open(
            os.path.join(
                self.ESPRESSO_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/espresso_structure.pwi",
            )
        ) as f:
            test_input = f.read()
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/espresso.pwi") as f:
            generated_input = f.read()
        self.assertEqual(test_input, generated_input)

        # Test user defined parameters
        _, _ = Dist.write_espresso_files(
            pseudopotentials=pseudopotentials,
            input_parameters={
                "SYSTEM": {
                    "ecutwfc": 40,
                    "exx_fraction": 0.30,
                    "degauss": 0.02,
                    "input_dft": "PBE",
                    "nspin": 1,
                }
            },
        )
        with open(
            os.path.join(
                self.ESPRESSO_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/espresso_user_parameters.pwi",
            )
        ) as f:
            test_input = f.read()
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/espresso.pwi") as f:
            generated_input = f.read()
        self.assertEqual(test_input, generated_input)
        # The input_file option is tested through the test for `generate_all()`
        # (in `test_cli.py`)

    @patch("builtins.print")
    def test_write_cp2k_files(self, mock_print):
        """Test method write_cp2k_files"""
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = [
            0.3,
        ]

        Dist = input.Distortions(
            {
                "vacancies": [
                    self.V_Cd_dict,
                ]
            },
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,
        )
        # Test `write_cp2k_files` method
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_cp2k_files()
        self.assertTrue(os.path.exists("vac_1_Cd_0/Unperturbed"))
        # Test input parameter file
        with open(
            os.path.join(
                self.CP2K_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/cp2k_input.inp",
            )
        ) as f:
            test_input = f.read()
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/cp2k_input.inp") as f:
            generated_input = f.read()
        self.assertEqual(test_input, generated_input)
        # Test input structure file
        with open(
            os.path.join(
                self.CP2K_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/structure.cif",
            )
        ) as f:
            test_input_struct = f.read()
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/structure.cif") as f:
            generated_input_struct = f.read()
        self.assertEqual(test_input_struct, generated_input_struct)

        # Test parameter file not written if write_structures_only = True
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_cp2k_files(write_structures_only=True)
        self.assertFalse(
            os.path.exists("vac_1_Cd_0/Bond_Distortion_30.0%/cp2k_input.inp")
        )
        self.assertTrue(
            os.path.exists("vac_1_Cd_0/Bond_Distortion_30.0%/structure.cif")
        )

        # Test user defined parameters
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_cp2k_files(
            input_file=os.path.join(self.CP2K_DATA_DIR, "cp2k_input_mod.inp"),
        )
        with open(
            os.path.join(
                self.CP2K_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/cp2k_input_user_parameters.inp",
            )
        ) as f:
            test_input = f.read()
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/cp2k_input.inp") as f:
            generated_input = f.read()
        self.assertEqual(test_input, generated_input)
        # The input_file option is tested through the test for `generate_all()`
        # (in `test_cli.py`)

    @patch("builtins.print")
    def test_write_castep_files(self, mock_print):
        """Test method write_castep_files"""
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = [
            0.3,
        ]

        Dist = input.Distortions(
            {
                "vacancies": [
                    self.V_Cd_dict,
                ]
            },
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,
        )
        # Test `write_castep_files` method, without specifing input file
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_castep_files()
        self.assertTrue(os.path.exists("vac_1_Cd_0/Unperturbed"))
        # Test input parameter file
        with open(
            os.path.join(
                self.CASTEP_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/castep.param",
            )
        ) as f:
            test_input = f.readlines()[
                28:
            ]  # only last line contains parameter (charge)
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/castep.param") as f:
            generated_input = f.readlines()[28:]
        self.assertEqual(test_input, generated_input)
        # Test input structure file
        with open(
            os.path.join(
                self.CASTEP_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/castep.cell",
            )
        ) as f:
            test_input_struct = f.readlines()[6:-3]  # avoid comment with file path etc
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/castep.cell") as f:
            generated_input_struct = f.readlines()[6:-3]
        self.assertEqual(test_input_struct, generated_input_struct)

        # Test only structure files are written if write_structures_only = True
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_castep_files(write_structures_only=True)
        self.assertFalse(
            os.path.exists("vac_1_Cd_0/Bond_Distortion_30.0%/castep.param")
        )
        self.assertTrue(os.path.exists("vac_1_Cd_0/Bond_Distortion_30.0%/castep.cell"))

        # Test user defined parameters
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_castep_files(
            input_file=os.path.join(self.CASTEP_DATA_DIR, "castep_mod.param"),
        )
        with open(
            os.path.join(
                self.CASTEP_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/castep_user_parameters.param",
            )
        ) as f:
            test_input = f.readlines()[28:]  # avoid comment with file path etc
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/castep.param") as f:
            generated_input = f.readlines()[28:]  # avoid comment with file path etc
        self.assertEqual(test_input, generated_input)
        # The input_file option is tested through the test for `generate_all()`
        # (in `test_cli.py`)

    @patch("builtins.print")
    def test_write_fhi_aims_files(self, mock_print):
        """Test method write_fhi_aims_files"""
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = [
            0.3,
        ]

        Dist = input.Distortions(
            {
                "vacancies": [
                    self.V_Cd_dict,
                ]
            },
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,
        )
        # Test `write_fhi_aims_files` method
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_fhi_aims_files()
        self.assertTrue(os.path.exists("vac_1_Cd_0/Unperturbed"))
        # Test input parameter file
        with open(
            os.path.join(
                self.FHI_AIMS_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/control.in",
            )
        ) as f:
            test_input = f.readlines()[6:]  # First 5 lines contain irrelevant info
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/control.in") as f:
            generated_input = f.readlines()[6:]
        self.assertEqual(test_input, generated_input)
        # Test input structure file
        with open(
            os.path.join(
                self.FHI_AIMS_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/geometry.in",
            )
        ) as f:
            test_input_struct = f.readlines()[6:]
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/geometry.in") as f:
            generated_input_struct = f.readlines()[6:]
        self.assertEqual(test_input_struct, generated_input_struct)

        # Test parameter file not written if write_structures_only = True
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        _, _ = Dist.write_fhi_aims_files(write_structures_only=True)
        self.assertFalse(os.path.exists("vac_1_Cd_0/Bond_Distortion_30.0%/control.in"))
        self.assertTrue(os.path.exists("vac_1_Cd_0/Bond_Distortion_30.0%/geometry.in"))

        # User defined parameters
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        ase_calculator = Aims(
            k_grid=(1, 1, 1),
            relax_geometry=("bfgs", 5e-4),
            xc=("hse06", 0.11),
            hse_unit="A",  # Angstrom
            spin="collinear",  # Spin polarized
            default_initial_moment=0,  # Needs to be set
            hybrid_xc_coeff=0.15,
            # By default symmetry is not preserved
        )
        _, _ = Dist.write_fhi_aims_files(ase_calculator=ase_calculator)
        with open(
            os.path.join(
                self.FHI_AIMS_DATA_DIR,
                "vac_1_Cd_0/Bond_Distortion_30.0%/control_user_parameters.in",
            )
        ) as f:
            test_input = f.readlines()[6:]  # First 5 lines contain irrelevant info
        with open("vac_1_Cd_0/Bond_Distortion_30.0%/control.in") as f:
            generated_input = f.readlines()[6:]
        self.assertEqual(test_input, generated_input)
        # The input_file option is tested through the test for `generate_all()`
        # (in `test_cli.py`)

    @patch("builtins.print")
    def test_apply_distortions(self, mock_print):
        """Test method apply_distortions"""
        # check files are not written if `apply_distortions()` method is used
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated defect folders
        reduced_V_Cd_dict = self.V_Cd_dict.copy()
        reduced_V_Cd_dict["charges"] = [0]
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = list(np.arange(-0.6, 0.601, 0.05))
        dist = input.Distortions(
            {"vacancies": [reduced_V_Cd_dict]},
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,  # default is True
        )
        distortion_defect_dict, distortion_metadata = dist.apply_distortions(
            verbose=False,
        )
        self.assertFalse(os.path.exists("vac_1_Cd_0"))

    def test_local_rattle(
        self,
    ):
        """ "Test option local_rattle of Distortions class"""
        reduced_V_Cd_dict = self.V_Cd_dict.copy()
        reduced_V_Cd_dict["charges"] = [0]
        oxidation_states = {"Cd": +2, "Te": -2}
        dist = input.Distortions(
            {"vacancies": [reduced_V_Cd_dict]},
            oxidation_states=oxidation_states,
            bond_distortions=[-0.3],
            local_rattle=True,  # default
        )
        self.assertTrue(dist.local_rattle)
        defects_dict, metadata_dict = dist.apply_distortions()
        # Check structure
        self.assertEqual(
            Structure.from_file(
                f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_0_-30.0%_Distortion_tailed_off_rattle_POSCAR"
            ),
            defects_dict["vac_1_Cd"]["charges"][0]["structures"]["distortions"][
                "Bond_Distortion_-30.0%"
            ],
        )
        # Check if option written to metadata file
        self.assertTrue(metadata_dict["distortion_parameters"]["local_rattle"])

        # Check interstitial (internally uses defect_index rather fractional coords)
        int_Cd_2 = self.Int_Cd_2_dict.copy()
        int_Cd_2["charges"] = [+2]
        oxidation_states = {"Cd": +2, "Te": -2}
        dist = input.Distortions(
            {"interstitials": [int_Cd_2]},
            oxidation_states=oxidation_states,
            bond_distortions=[
                -0.3,
            ],  # zero electron change
            local_rattle=True,  # default
        )
        defects_dict, metadata_dict = dist.apply_distortions()
        self.assertEqual(
            Structure.from_file(
                f"{self.VASP_CDTE_DATA_DIR}/Int_Cd_2_2_tailed_off_rattle_POSCAR"
            ),
            defects_dict["Int_Cd_2"]["charges"][2]["structures"]["distortions"][
                "Rattled"
            ],
        )


if __name__ == "__main__":
    unittest.main()
