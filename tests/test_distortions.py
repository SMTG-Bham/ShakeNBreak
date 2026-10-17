"""
Note that most of the tests in ``test_input`` implicitly test these functions.
"""

import os
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.core.periodic_table import Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Lattice, Structure

from shakenbreak import analysis, distortions


class DistortionTestCase(unittest.TestCase):
    """Test shakenbreak structure distortion functions."""

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.cdte_defect_dict = loadfn(os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.json"))

        self.V_Cd_struc = Structure.from_file(os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_POSCAR"))
        self.V_Cd_minus0pt5_struc = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Unrattled_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR")
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
        self.Int_Cd_2_struc = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_struc = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_Int_Cd_2_-60%_Distortion_Unrattled_POSCAR",
            )
        )
        self.Int_Cd_2_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_Rattled_POSCAR")
        )
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_NN_10_POSCAR")
        )
        # Confirm correct structures and json dict:
        assert self.V_Cd_struc == self.cdte_defect_dict["vacancies"][0]["supercell"]["structure"]
        assert self.Int_Cd_2_struc == self.cdte_defect_dict["interstitials"][1]["supercell"]["structure"]

        self.K4MnFe2F12_struc = Structure.from_file(os.path.join(self.DATA_DIR, "vasp/K4MnFe2F12_POSCAR"))

        self.V_Cd_dimer_distortion = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_Dimer_Distortion_Unrattled_POSCAR")
        )
        self.V_Si_HfSiO_interface_entry = loadfn(
            os.path.join(self.DATA_DIR, "v_Si_C1_O1.70_HfSiO_interface_+1.json")
        )
        warnings.simplefilter("always")  # Cause all warnings to always be triggered.

    def tearDown(self):
        warnings.resetwarnings()  # Reset warnings to default state.

    @patch("builtins.print")
    def test_distort_V_Cd(self, mock_print):
        """Test bond distortion function for V_Cd."""
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.distort(self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords)
        assert output["distorted_structure"] == self.V_Cd_minus0pt5_struc
        assert output["undistorted_structure"] == self.V_Cd_struc
        assert output["num_distorted_neighbours"] == 2
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        assert output.get("defect_site_index") is None
        assert output["distorted_atoms"] == [[32, "Te"], [41, "Te"]]
        distortions.distort(self.V_Cd_struc, 2, 0.5, frac_coords=vac_coords, verbose=True)
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: {vac_coords}\n"
            "            Original Neighbour Distances: [(2.83, 32, 'Te'), (2.83, 41, 'Te')]\n"
            "            Distorted Neighbour Distances:\n\t[(1.42, 32, 'Te'), (1.42, 41, 'Te')]"
        )

        # Test if num_nearest_neighbours = 0 that nothing happens:
        output = distortions.distort(self.V_Cd_struc, 0, 0.5, frac_coords=vac_coords)
        assert output["distorted_structure"] == self.V_Cd_struc
        assert output["undistorted_structure"] == self.V_Cd_struc
        assert output["num_distorted_neighbours"] == 0
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        assert output.get("defect_site_index") is None
        assert output["distorted_atoms"] == []
        distortions.distort(self.V_Cd_struc, 0, 0.5, frac_coords=vac_coords, verbose=True)
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: {vac_coords}\n"
            "            Original Neighbour Distances: []\n"
            "            Distorted Neighbour Distances:\n\t[]"
        )

    def test_distort_negative_factor(self):
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        with pytest.raises(ValueError) as e:
            distortions.distort(self.V_Cd_struc, 2, -0.5, frac_coords=vac_coords)
        assert (
            str(e.value)
            == "Bond distortion factor (i.e. factor to multiply inter-atomic distances to get bond"
            " distortions) cannot be less than zero! Note that input `bond_distortions` are converted"
            " to distortion factors as: `distortion_factor = 1 + bond_distortion`."
        )

    @patch("builtins.print")
    def test_distort_degenerate_case(self, mock_print):
        """
        Test bond distortion function for case of degenerate distances (but non-degenerate
        NN distortion combinations).
        """
        # create a pymatgen structure with octahedral coordination of vacant site
        # (this is perovskite-like)
        lattice = Lattice.cubic(3.0)
        sites = [
            PeriodicSite(Element("H"), [0, 0, 0], lattice),
            PeriodicSite(Element("H"), [0, 0.5, 0.5], lattice),
            PeriodicSite(Element("H"), [0.5, 0, 0.5], lattice),
            PeriodicSite(Element("H"), [0.5, 0.5, 0], lattice),
            # PeriodicSite(Element("V"), [0.5, 0.5, 0.5], lattice),  # vacancy site
        ]
        structure = Structure.from_sites(sites)
        vac_coords = np.array([0.5, 0.5, 0.5])  # centre of octahedral coordination
        output = distortions.distort(structure, 2, 0.5, frac_coords=vac_coords, verbose=True)
        assert output["undistorted_structure"] == structure
        assert output["num_distorted_neighbours"] == 2
        np.testing.assert_array_equal(output["defect_frac_coords"], vac_coords)
        assert output.get("defect_site_index") is None
        assert output["distorted_atoms"] == [[1, "H"], [2, "H"]]

        # get min distance in distorted structure:
        # (would be 1.5 Å if trans NN combo distorted, but shorter for cis distortion combo)
        dist_matrix = output["distorted_structure"].distance_matrix.flatten()
        min_dist = np.min(dist_matrix[dist_matrix > 0])
        assert np.isclose(min_dist, 1.06, rtol=0, atol=5e-3)

        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: {vac_coords}\n"
            "            Original Neighbour Distances: [(1.5, 1, 'H'), (1.5, 2, 'H')]\n"
            "            Distorted Neighbour Distances:\n\t[(0.75, 1, 'H'), (0.75, 2, 'H')]"
        )

    @patch("builtins.print")
    def test_distort_Int_Cd_2(self, mock_print):
        """Test bond distortion function for Int_Cd_2."""
        site_index = 64  # Cd interstitial site index (python indexing)
        output = distortions.distort(self.Int_Cd_2_struc, 2, 0.4, site_index=site_index, verbose=False)
        assert output["distorted_structure"] == self.Int_Cd_2_minus0pt6_struc
        assert output["undistorted_structure"] == self.Int_Cd_2_struc
        assert output["num_distorted_neighbours"] == 2
        assert output["defect_site_index"] == site_index
        assert output.get("defect_frac_coords") is None
        assert output["distorted_atoms"] == [[9, "Cd"], [21, "Cd"]]
        distortions.distort(self.Int_Cd_2_struc, 2, 0.4, site_index=site_index, verbose=True)
        mock_print.assert_called_with(
            f"\tDefect Site Index / Frac Coords: {site_index}\n"
            "            Original Neighbour Distances: [(2.71, 9, 'Cd'), (2.71, 21, 'Cd')]\n"
            "            Distorted Neighbour Distances:\n\t[(1.09, 9, 'Cd'), (1.09, 21, 'Cd')]"
        )

        # test correct behaviour with odd parameters input:
        output = distortions.distort(
            self.Int_Cd_2_struc,
            num_nearest_neighbours=10,
            distortion_factor=0.4,
            distorted_element="Cd",
            site_index=site_index,
            verbose=True,
        )
        assert output["distorted_structure"] == self.Int_Cd_2_minus0pt6_NN_10_struc_rattled
        assert output["undistorted_structure"] == self.Int_Cd_2_struc
        assert output["num_distorted_neighbours"] == 10
        assert output["defect_site_index"] == site_index
        assert output.get("defect_frac_coords") is None
        assert sorted(output["distorted_atoms"]) == sorted(
            [
                [9, "Cd"],
                [21, "Cd"],
                [28, "Cd"],
                [0, "Cd"],
                [13, "Cd"],
                [23, "Cd"],
                [29, "Cd"],
                [1, "Cd"],
                [2, "Cd"],
                [4, "Cd"],
            ]
        )
        mock_print.assert_called_with(
            "\tDefect Site Index / Frac Coords: 64\n"
            "            Original Neighbour Distances: [(2.71, 9, 'Cd'), (2.71, 21, 'Cd'), "
            "(2.71, 28, 'Cd'), (4.25, 0, 'Cd'), (4.25, 13, 'Cd'), (4.25, 23, 'Cd'), (4.25, 29, 'Cd'), "
            "(5.36, 1, 'Cd'), (5.36, 2, 'Cd'), (5.36, 4, 'Cd')]\n"
            "            Distorted Neighbour Distances:\n\t[(1.09, 9, 'Cd'), (1.09, 21, 'Cd'), "
            "(1.09, 28, 'Cd'), (1.7, 0, 'Cd'), (1.7, 13, 'Cd'), (1.7, 23, 'Cd'), (1.7, 29, 'Cd'), "
            "(2.15, 1, 'Cd'), (2.15, 2, 'Cd'), (2.15, 4, 'Cd')]"
        )

    def test_distorted_element_error(self):
        """Test error message when non-existent element symbol provided."""
        site_index = 64  # Cd interstitial site index (python indexing)
        for missing_element in ["C", "O", "H", "N", "S", "P", "X"]:
            for num_neighbours in range(8):
                for distortion_factor in np.arange(-0.6, 0.61, 0.1):
                    with pytest.raises(ValueError) as e:
                        distortions.distort(
                            self.Int_Cd_2_struc,  # cause error for no `missing_element` neighbours
                            num_nearest_neighbours=num_neighbours,
                            distortion_factor=distortion_factor,
                            site_index=site_index,
                            distorted_element=missing_element,
                        )

                    assert (
                        f"No atoms of `distorted_element` = ['{missing_element}'] found in the defect"
                        " structure, cannot apply bond distortions." == str(e.value)
                    )

    def test_rattle_V_Cd(self):
        """Test structure rattle function for V_Cd."""
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        d_min = 0.8 * sorted_distances[len(self.V_Cd_struc) + 20]

        rattling_atom_indices = np.arange(0, 63)
        idx = np.isin(rattling_atom_indices, [32, 41])
        rattling_atom_indices = rattling_atom_indices[~idx]  # removed distorted Te indices

        assert (
            distortions.rattle(
                self.V_Cd_minus0pt5_struc, stdev=0.25, d_min=d_min, active_atoms=rattling_atom_indices
            )
            == self.V_Cd_minus0pt5_struc_rattled
        )
        assert (
            distortions.rattle(
                self.V_Cd_minus0pt5_struc, stdev=0.1, d_min=d_min, active_atoms=rattling_atom_indices
            )
            == self.V_Cd_minus0pt5_struc_0pt1_rattled
        )

        # test default d_min setting:
        assert (
            distortions.rattle(
                self.V_Cd_minus0pt5_struc,
                stdev=0.1,
                # d_min=d_min,
                active_atoms=rattling_atom_indices,
            )
            == self.V_Cd_minus0pt5_struc_0pt1_rattled
        )

    def test_rattle_Int_Cd_2(self):
        """Test structure rattle function for Int_Cd_2."""
        sorted_distances = np.sort(self.Int_Cd_2_struc.distance_matrix.flatten())
        d_min = 0.8 * sorted_distances[len(self.Int_Cd_2_struc) + 20]

        rattling_atom_indices = np.arange(0, 64)  # not including index 64 which is Int_Cd_2
        idx = np.isin(rattling_atom_indices, [9, 21])
        rattling_atom_indices = rattling_atom_indices[~idx]  # removed distorted Cd indices

        assert (
            distortions.rattle(
                self.Int_Cd_2_minus0pt6_struc,
                d_min=d_min,
                active_atoms=rattling_atom_indices,
                stdev=0.28333683853583164,  # 10% of CdTe bond length, default
                seed=40,  # distortion_factor * 100, default
            )
            == self.Int_Cd_2_minus0pt6_struc_rattled
        )

        # test defaults:
        assert (
            distortions.rattle(
                self.Int_Cd_2_minus0pt6_struc,
                # d_min=d_min,  # default
                active_atoms=rattling_atom_indices,
                # stdev=0.28333683853583164,  # 10% of CdTe bond length, default
                seed=40,  # distortion_factor * 100, default
            )
            == self.Int_Cd_2_minus0pt6_struc_rattled
        )

    def test_rattle_kwargs(
        self,
    ):  # test all possible kwargs and explicitly check output
        """Test structure rattle function with all possible kwargs."""
        rattling_atom_indices = np.arange(0, 31)  # Only rattle Cd
        sorted_distances = np.sort(self.V_Cd_struc.distance_matrix.flatten())
        bulk_bond_length = sorted_distances[len(self.V_Cd_struc) + 20]
        assert bulk_bond_length == 2.8333683853583165

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

        assert V_Cd_kwarg_rattled == self.V_Cd_minus0pt5_struc_kwarged

    def test_rattle_bulk_K4MnFe2F12(self):
        # failed with previous rattle code
        # (which ignored first 20 non-zero distances, because assumed a supercell structure as input)
        rattled_struc = distortions.rattle(self.K4MnFe2F12_struc)
        rattled_distances = rattled_struc.distance_matrix[rattled_struc.distance_matrix > 0]

        assert np.isclose(min(rattled_distances), 1.6166697059123376, rtol=0, atol=5e-8)
        assert np.isclose(np.mean(rattled_distances), 3.4581277362842666, rtol=0, atol=5e-8)
        assert np.isclose(np.sort(rattled_distances.flatten())[3], 1.7410986565232485, rtol=0, atol=5e-8)

    def test_apply_dimer_distortion(self):
        """Test apply_dimer_distortion function."""
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.apply_dimer_distortion(self.V_Cd_struc, frac_coords=vac_coords)
        # to update dimer structure (last change was switch to ``get_dimer_bond_length`` default):
        # output["distorted_structure"].to(fmt="POSCAR", filename=os.path.join(
        #         self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_Dimer_Distortion_Unrattled_POSCAR"
        #     )
        # )
        assert output["distorted_structure"] == self.V_Cd_dimer_distortion
        assert output["undistorted_structure"] == self.V_Cd_struc
        assert output["num_distorted_neighbours"] == 2
        assert output["distorted_atoms"] == [[32, "Te"], [41, "Te"]]
        # Check that 2 Te are separated by 2 Å
        homo_bonds = analysis.get_homoionic_bonds(
            output["distorted_structure"],
            elements=[
                "Te",
            ],
        )
        assert homo_bonds == {"Te(32)": {"Te(41)": "2.76 A"}}

    def test_apply_dimer_distortion_V_Te(self):
        """Test apply_dimer_distortion function."""
        output = distortions.apply_dimer_distortion(
            self.V_Cd_struc,
            site_index=len(self.V_Cd_struc) - 1,  # Te
        )
        assert output["undistorted_structure"] == self.V_Cd_struc
        assert output["num_distorted_neighbours"] == 2
        assert output["distorted_atoms"] == [[10, "Cd"], [20, "Cd"]]
        # Check that 2 Te are separated by 2 Å
        homo_bonds = analysis.get_homoionic_bonds(
            output["distorted_structure"],
            elements=[
                "Cd",
            ],
        )
        assert homo_bonds == {"Cd(10)": {"Cd(20)": "2.88 A"}}

    def test_get_dimer_bond_length(self):
        assert np.isclose(distortions.get_dimer_bond_length("O", "O"), 1.48, rtol=0, atol=5e-8)
        assert np.isclose(distortions.get_dimer_bond_length("H", "H"), 1.07, rtol=0, atol=5e-8)
        assert np.isclose(distortions.get_dimer_bond_length("S", "S"), 2.06, rtol=0, atol=5e-8)
        assert np.isclose(distortions.get_dimer_bond_length("O", "N"), 1.51, rtol=0, atol=5e-8)
        assert np.isclose(distortions.get_dimer_bond_length("O", "S"), 1.57, rtol=0, atol=5e-8)

    def test_apply_dimer_distortion_custom_bond_length(self):
        """Test apply_dimer_distortion function."""
        vac_coords = np.array([0, 0, 0])  # Cd vacancy fractional coordinates
        output = distortions.apply_dimer_distortion(
            self.V_Cd_struc, frac_coords=vac_coords, dimer_bond_length=1
        )
        assert output["distorted_structure"] != self.V_Cd_dimer_distortion
        assert output["undistorted_structure"] == self.V_Cd_struc
        assert output["num_distorted_neighbours"] == 2
        assert output["distorted_atoms"] == [[32, "Te"], [41, "Te"]]
        # Check that 2 Te are separated by 1 Å
        homo_bonds = analysis.get_homoionic_bonds(
            output["distorted_structure"],
            elements=[
                "Te",
            ],
        )
        assert homo_bonds == {"Te(32)": {"Te(41)": "1.0 A"}}

    def test_apply_dimer_distortion_less_than_2_CNN_NN(self):
        """
        Test the ``apply_dimer_distortion`` function, for a case where
        ``CrystalNN`` returns less than 2 neighbours, and so we default to
        using ``_get_nns_to_distort``.

        Previously failed, causing the bug reported in #85.
        """
        output = distortions.apply_dimer_distortion(
            self.V_Si_HfSiO_interface_entry.defect_supercell,
            frac_coords=self.V_Si_HfSiO_interface_entry.defect_supercell_site.frac_coords,
        )
        assert output["undistorted_structure"] == self.V_Si_HfSiO_interface_entry.defect_supercell
        assert output["num_distorted_neighbours"] == 2
        assert output["distorted_atoms"] == [[82, "Si"], [177, "O"]]

        dist_matrix = output["distorted_structure"].distance_matrix.flatten()
        min_dist = np.min(dist_matrix[dist_matrix > 0])
        assert min_dist > 1.5  # Si-O dimer bond length is larger than min_dist in orig structure

    def test_rattle_removes_net_translation(self):
        """
        Test that rattling leaves zero net atomic displacement (i.e. no rigid
        translation of the supercell), and that removing it is an exact
        isometry (all interatomic distances unchanged).
        """

        def _mean_disp(rattled, ref):  # unwrapped mean displacement, in Cartesian Å
            frac_disp = rattled.frac_coords - ref.frac_coords
            frac_disp -= np.round(frac_disp)  # undo any periodic wrapping
            return np.linalg.norm(frac_disp.mean(axis=0) @ ref.lattice.matrix)

        struc = self.V_Cd_minus0pt5_struc  # 63-atom CdTe vacancy supercell
        vac_coords = np.array([0, 0, 0])  # V_Cd frac coords
        active_atoms = list(range(0, len(struc), 2))  # subset, to test the ``active_atoms`` path
        rattled_strucs = {
            "rattle": distortions.rattle(struc, stdev=0.25, d_min=2.25, seed=42),
            "rattle, active_atoms": distortions.rattle(
                struc, stdev=0.25, d_min=2.25, seed=42, active_atoms=active_atoms
            ),
            "local_mc_rattle, site_index": distortions.local_mc_rattle(
                struc, site_index=0, stdev=0.25, d_min=2.25, seed=42
            ),
            "local_mc_rattle, frac_coords": distortions.local_mc_rattle(
                struc, frac_coords=vac_coords, stdev=0.25, d_min=2.25, seed=42
            ),
        }

        for name, rattled in rattled_strucs.items():
            with self.subTest(rattle=name):
                assert len(rattled) == len(struc)
                assert np.isclose(_mean_disp(rattled, struc), 0, rtol=0, atol=5e-11)
                # rattling actually did something (i.e. we haven't just returned the input):
                assert np.abs(rattled.cart_coords - struc.cart_coords).max() > 0.01

        # re-centring is a rigid translation, so it must leave all interatomic distances (and hence
        # ``d_min`` and the applied bond distortions) exactly unchanged. Test the un-recentred case
        # directly, by feeding ``_remove_net_translation`` a structure with a known net offset:
        rattled = rattled_strucs["rattle"]
        offset = rattled.copy()
        offset.translate_sites(
            range(len(offset)), [0.3, -0.15, 0.07], frac_coords=False, to_unit_cell=False
        )
        assert np.isclose(
            _mean_disp(offset, struc), np.linalg.norm([0.3, -0.15, 0.07]), rtol=0, atol=5e-11
        )

        recentred = distortions._remove_net_translation(offset, struc)
        assert np.isclose(_mean_disp(recentred, struc), 0, rtol=0, atol=5e-11)
        assert np.abs(recentred.distance_matrix - rattled.distance_matrix).max() < 1e-10

    def test_local_mc_rattle_d_min_fallback_with_explicit_stdev(self):
        """
        Test the reduced-``d_min`` fallback in ``local_mc_rattle`` when both
        ``stdev`` and ``d_min`` are explicitly provided.

        Previously failed due to a hidden bug.
        """
        # heavily distorted structure, so the initial ``d_min`` cannot be satisfied and the fallback
        # (``sorted_distances[0] + stdev``) is triggered:
        distorted = distortions.distort(self.V_Cd_struc, 2, 0.2, frac_coords=np.array([0, 0, 0]))[
            "distorted_structure"
        ]
        assert np.sort(distorted.distance_matrix[distorted.distance_matrix > 0.5])[0] < 1.0

        with warnings.catch_warnings(record=True) as w:
            rattled = distortions.local_mc_rattle(
                distorted, frac_coords=np.array([0, 0, 0]), stdev=0.25, d_min=2.25, verbose=True, seed=42
            )

        assert len(rattled) == len(distorted)
        assert any("Initial rattle with d_min 2.25" in str(i.message) for i in w)
