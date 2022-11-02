import datetime
import json
import os
import shutil
import subprocess
import unittest
import warnings
import copy

import numpy as np
import yaml

# Click
from click.testing import CliRunner
from monty.serialization import loadfn

# Pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Poscar

from shakenbreak.cli import snb
from shakenbreak.distortions import rattle

file_path = os.path.dirname(__file__)


def if_present_rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class CLITestCase(unittest.TestCase):
    """Test ShakeNBreak structure distortion helper functions"""

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.EXAMPLE_RESULTS = os.path.join(self.DATA_DIR, "example_results")
        self.VASP_DIR = os.path.join(self.DATA_DIR, "vasp")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.VASP_TIO2_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/vac_1_Ti_0")
        self.CdTe_bulk_struc = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_Bulk_Supercell_POSCAR")
        )
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR",
            )
        )  # Local rattled
        self.V_Cd_minus0pt5_struc_local_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_V_Cd_-50%_Distortion_local_rattle_POSCAR",
            )
        )  # Local rattled
        self.CdTe_distortion_config = os.path.join(
            self.VASP_CDTE_DATA_DIR, "distortion_config.yml"
        )
        self.V_Cd_minus0pt5_struc_kwarged = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Kwarged_POSCAR")
        )
        self.V_Cd_0pt3_local_rattled = Structure.from_file(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_0_30%_local_rattle_POSCAR")
        )
        self.V_Cd_minus0pt55_CONTCAR_struc = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR"
        )
        self.cdte_defect_dict = loadfn(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.json")
        )
        self.Int_Cd_2_dict = self.cdte_defect_dict["interstitials"][1]
        self.Int_Cd_2_minus0pt6_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_Int_Cd_2_-60%_Distortion_Rattled_POSCAR"
            )
        )
        previous_default_rattle_settings = {"stdev": 0.25, "seed": 42}
        with open("previous_default_rattle_settings.yaml", "w") as fp:
            yaml.dump(previous_default_rattle_settings, fp)
        self.previous_default_rattle_settings_config = os.path.join(
            os.path.dirname(__file__), "previous_default_rattle_settings.yaml"
        )

    def tearDown(self):
        os.chdir(os.path.dirname(__file__))
        for i in [
            "parsed_defects_dict.json",
            "distortion_metadata.json",
            "test_config.yml",
            "job_file",
            "previous_default_rattle_settings.yaml",
        ]:
            if_present_rm(i)

        for i in os.listdir("."):
            if "distortion_metadata" in i:
                os.remove(i)
            elif (
                "Vac_Cd" in i
                or "Int_Cd" in i
                or "Wally_McDoodle" in i
                or "pesky_defects" in i
                or "vac_1_Cd_0" in i
            ):
                shutil.rmtree(i)

        for defect in os.listdir(self.EXAMPLE_RESULTS):
            if os.path.isdir(f"{self.EXAMPLE_RESULTS}/{defect}"):
                [
                    shutil.rmtree(f"{self.EXAMPLE_RESULTS}/{defect}/{dir}")
                    for dir in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
                    if "_from_" in dir
                ]
                [
                    os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
                    for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
                    if file.endswith(".png") or file.endswith(".svg")
                ]
            elif os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}"):
                os.remove(f"{self.EXAMPLE_RESULTS}/{defect}")

        if_present_rm(f"{self.EXAMPLE_RESULTS}/pesky_defects/")
        if_present_rm(f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0_defect_folder")

        # Remove re-generated files
        folder = "Bond_Distortion_-60.0%_from_0"
        for charge in [-1, -2]:
            if os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, f"vac_1_Cd_{charge}", folder)
            ):
                shutil.rmtree(
                    os.path.join(self.EXAMPLE_RESULTS, f"vac_1_Cd_{charge}", folder)
                )
        folder = "Bond_Distortion_20.0%_from_-1"
        for charge in [0, -2]:
            if os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, f"vac_1_Cd_{charge}", folder)
            ):
                shutil.rmtree(
                    os.path.join(self.EXAMPLE_RESULTS, f"vac_1_Cd_{charge}", folder)
                )
        if_present_rm(
            os.path.join(
                self.EXAMPLE_RESULTS, "vac_1_Cd_0/Bond_Distortion_-48.0%_High_Energy"
            )
        )
        if_present_rm("Rattled_Bulk_CdTe_POSCAR")

        # Remove parsed vac_1_Ti_0 energies file
        if_present_rm(f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0/vac_1_Ti_0.yaml")

    def test_snb_generate(self):
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c 0",
                "--config",
                f"{self.previous_default_rattle_settings_config}",  # previous default
                "-v",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            f"Auto site-matching identified {self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR "
            "to be type Vacancy with site Cd at [0.000, 0.000, 0.000]",
            result.output,
        )
        self.assertIn(
            "Oxidation states were not explicitly set, thus have been guessed as {"
            "'Cd': 2.0, 'Te': -2.0}. If this is unreasonable you should manually set "
            "oxidation_states",
            result.output,
        )
        self.assertIn(
            "Applying ShakeNBreak...",
            "Applying ShakeNBreak... Will apply the following bond distortions: ["
            "'-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', "
            "'0.3', '0.4', '0.5', '0.6']. Then, will rattle with a std dev of 0.25 Å",
            result.output,
        )
        self.assertIn("Defect: Vac_Cd_mult32", result.output)
        self.assertIn("Number of missing electrons in neutral state: 2", result.output)
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        self.assertIn("--Distortion -60.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
            result.output,
        )
        self.assertIn("--Distortion -40.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
            result.output,
        )

        # check if correct files were created:
        V_Cd_Bond_Distortion_folder = "Vac_Cd_mult32_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_Bond_Distortion_folder))
        V_Cd_minus0pt5_rattled_POSCAR = Poscar.from_file(
            V_Cd_Bond_Distortion_folder + "/POSCAR"
        )
        self.assertEqual(
            V_Cd_minus0pt5_rattled_POSCAR.comment,
            "-50.0%__num_neighbours=2__Vac_Cd_mult32",
        )  # default
        self.assertEqual(
            V_Cd_minus0pt5_rattled_POSCAR.structure,
            self.V_Cd_minus0pt5_struc_rattled,
        )

        # Test recognises distortion_metadata.json:
        if_present_rm("Vac_Cd_mult32_0")  # but distortion_metadata.json still present
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c",
                "0",
            ],
            catch_exceptions=False,
        )  # non-verbose this time
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(
            "Auto site-matching identified"
            f" {self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR "
            "to be type Vacancy with site Cd at [0.000, 0.000, 0.000]",
            result.output,
        )
        self.assertIn(
            "Oxidation states were not explicitly set, thus have been guessed as {"
            "'Cd': 2.0, 'Te': -2.0}. If this is unreasonable you should manually set "
            "oxidation_states",
            result.output,
        )
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ["
            "'-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', "
            "'0.3', '0.4', '0.5', '0.6']. Then, will rattle with a std dev of 0.28 Å",
            result.output,
        )
        self.assertIn("Defect: Vac_Cd_mult32", result.output)
        self.assertIn("Number of missing electrons in neutral state: 2", result.output)
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        self.assertNotIn("--Distortion -60.0%", result.output)
        self.assertNotIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
            result.output,
        )
        current_datetime_wo_minutes = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H"
        )  # keep copy of old metadata file
        self.assertIn(
            "There is a previous version of distortion_metadata.json. Will rename old "
            f"metadata to distortion_metadata_{current_datetime_wo_minutes}",
            # skipping minutes comparison in case this changes between test and check
            result.output,
        )
        self.assertIn(
            "Combining old and new metadata in distortion_metadata.json", result.output
        )

        # test defect_index option:
        self.tearDown()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c",
                "0",
                "--defect-index",
                "4",
                "-v",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Auto site-matching", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        self.assertIn("Applying ShakeNBreak...", result.output)
        self.assertIn("Defect: Vac_Cd_mult32", result.output)
        self.assertIn("Number of missing electrons in neutral state: 2", result.output)
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )

        wrong_site_V_Cd_dict = {
            "distortion_parameters": {
                "distortion_increment": 0.1,
                "bond_distortions": [
                    -0.6,
                    -0.5,
                    -0.4,
                    -0.3,
                    -0.2,
                    -0.1,
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                ],
                "rattle_stdev": 0.28333683853583164,
                "local_rattle": False,
            },
            "defects": {
                "Vac_Cd_mult32": {
                    "unique_site": [0.5, 0.0, 0.0],
                    "charges": {
                        "0": {  # json converts integer strings to keys
                            "num_nearest_neighbours": 2,
                            "distorted_atoms": [
                                [37, "Te"],
                                [46, "Te"],
                            ],
                            "distortion_parameters": {
                                "bond_distortions": [
                                    -0.6,
                                    -0.5,
                                    -0.4,
                                    -0.3,
                                    -0.2,
                                    -0.1,
                                    0.0,
                                    0.1,
                                    0.2,
                                    0.3,
                                    0.4,
                                    0.5,
                                    0.6,
                                ],
                                "rattle_stdev": 0.28333683853583164,
                            },
                        },
                    },
                }
            },
        }
        # check defects from old metadata file are in new metadata file
        with open("distortion_metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
        np.testing.assert_equal(metadata, wrong_site_V_Cd_dict)

        # test warning with defect_coords option but wrong site: (matches Cd site in bulk)
        # using Int_Cd because V_Cd is at (0,0,0) so fractional and Cartesian coordinates the same
        self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "generate",
                    "-d",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Int_Cd_2_POSCAR",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "-c",
                    "0",
                    "--defect-coords",
                    0.0,  # 0.8125,  # actual Int_Cd_2 site
                    0.0,  # 0.1875,
                    0.0,  # 0.8125,
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertEqual(result.exit_code, 0)
        warning_message = (
            "Coordinates (0.0, 0.0, 0.0) were specified for (auto-determined) interstitial "
            "defect, but there are no extra/missing/different species within a 2.5 Å radius of "
            "this site when comparing bulk and defect structures. If you are trying to "
            "generate non-defect polaronic distortions, please use the distort() and rattle() "
            "functions in shakenbreak.distortions via the Python API. Reverting to auto-site "
            "matching instead."
        )
        self.assertTrue(any(warning.category == UserWarning for warning in w))
        self.assertTrue(any(str(warning.message) == warning_message for warning in w))
        self.assertIn("--Distortion -60.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: 65\n"
            + "            Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd')]",
            result.output,
        )
        self.assertEqual(
            Structure.from_file("Int_Cd_mult128_0/Bond_Distortion_-60.0%/POSCAR"),
            self.Int_Cd_2_minus0pt6_struc_rattled,
        )

        # test defect_coords working even when slightly off correct site
        self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "generate",
                    "-d",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Int_Cd_2_POSCAR",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "-c",
                    "0",
                    "--defect-coords",
                    0.8,  # 0.8125,  # actual Int_Cd_2 site
                    0.15,  # 0.1875,
                    0.85,  # 0.8125,
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertEqual(result.exit_code, 0)
        if w:
            # Check no problems in identifying the defect site
            self.assertFalse(
                any(str(warning.message) == warning_message for warning in w)
            )
            self.assertFalse(
                any("Coordinates" in str(warning.message) for warning in w)
            )

        self.assertNotIn("Auto site-matching", result.output)
        self.assertIn("--Distortion -60.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: 65\n"
            + "            Original Neighbour Distances: [(2.71, 10, 'Cd'), (2.71, 22, 'Cd')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.09, 10, 'Cd'), (1.09, 22, 'Cd')]",
            result.output,
        )
        self.assertEqual(
            Structure.from_file("Int_Cd_mult128_0/Bond_Distortion_-60.0%/POSCAR"),
            self.Int_Cd_2_minus0pt6_struc_rattled,
        )

        # test defect_coords working even when significantly off (~2.2 Å) correct site,
        # with rattled bulk
        self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            rattled_bulk = rattle(self.CdTe_bulk_struc)
            rattled_bulk.to(filename="./Rattled_Bulk_CdTe_POSCAR", fmt="POSCAR")
            result = runner.invoke(
                snb,
                [
                    "generate",
                    "-d",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Int_Cd_2_POSCAR",
                    "-b",
                    "Rattled_Bulk_CdTe_POSCAR",
                    "-c",
                    "0",
                    "--defect-coords",
                    0.9,  # 0.8125,  # actual Int_Cd_2 site
                    0.3,  # 0.1875,
                    0.9,  # 0.8125,
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertEqual(result.exit_code, 0)
        if w:
            # Check no problems in identifying the defect site
            # Note this also gives the following UserWarning:
            # Bond_Distortion_-60.0% for defect Int_Cd_mult1 gives an interatomic
            # distance less than 1.0 Å (1.0 Å), which is likely to give explosive forces.
            # Omitting this distortion.
            self.assertFalse(
                any("Coordinates" in str(warning.message) for warning in w)
            )

        self.assertNotIn("Auto site-matching", result.output)
        self.assertIn("--Distortion -60.0%", result.output)

        self.assertIn(
            "\tDefect Site Index / Frac Coords: 65\n"
            + "            Original Neighbour Distances: [(2.49, 10, 'Cd'), (2.59, 22, 'Cd')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.0, 10, 'Cd'), (1.04, 22, 'Cd')]",
            result.output,
        )

        # test defect_coords working even when slightly off correct site with V_Cd and rattled bulk
        self.tearDown()
        with warnings.catch_warnings(record=True) as w:
            rattled_bulk = rattle(self.CdTe_bulk_struc)
            rattled_bulk.to(filename="./Rattled_Bulk_CdTe_POSCAR", fmt="POSCAR")
            result = runner.invoke(
                snb,
                [
                    "generate",
                    "-d",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                    "-b",
                    "Rattled_Bulk_CdTe_POSCAR",
                    "-c",
                    "0",
                    "--defect-coords",
                    0.025,
                    0.025,
                    0.025,  # close just not quite 0,0,0
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertEqual(result.exit_code, 0)
        if w:
            # Check no problems in identifying the defect site
            self.assertNotIn("Coordinates", str(w[0].message))
        self.assertIn("--Distortion -60.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0.015687 0.01685  0.001366]\n"  # rattled position
            + "            Original Neighbour Distances: [(2.33, 42, 'Te'), (2.73, 33, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(0.93, 42, 'Te'), (1.09, 33, 'Te')]",
            result.output,
        )
        self.assertNotIn("Auto site-matching", result.output)

        # test distortion dict info with defect_coords slightly off correct site with V_Cd
        if_present_rm("distortion_metadata.json")
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "generate",
                    "-d",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "-c",
                    "0",
                    "--defect-coords",
                    0.025,
                    0.025,
                    0.025,  # close just not quite 0,0,0
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertEqual(result.exit_code, 0)
        if w:
            # Check no problems in identifying the defect site
            self.assertNotIn("Coordinates", str(w[0].message))
        self.assertIn("--Distortion -60.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
            result.output,
        )
        self.assertNotIn("Auto site-matching", result.output)
        spec_coords_V_Cd_dict = {
            "distortion_parameters": {
                "distortion_increment": 0.1,
                "bond_distortions": [
                    -0.6,
                    -0.5,
                    -0.4,
                    -0.3,
                    -0.2,
                    -0.1,
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                ],
                "rattle_stdev": 0.28333683853583164,
                "local_rattle": False,
            },
            "defects": {
                "Vac_Cd_mult32": {
                    "unique_site": [
                        0.0,
                        0.0,
                        0.0,
                    ],  # matching final site not slightly-off
                    # user input
                    "charges": {
                        "0": {  # json converts integer strings to keys
                            "num_nearest_neighbours": 2,
                            "distorted_atoms": [
                                [33, "Te"],
                                [42, "Te"],
                            ],
                            "distortion_parameters": {
                                "bond_distortions": [
                                    -0.6,
                                    -0.5,
                                    -0.4,
                                    -0.3,
                                    -0.2,
                                    -0.1,
                                    0.0,
                                    0.1,
                                    0.2,
                                    0.3,
                                    0.4,
                                    0.5,
                                    0.6,
                                ],
                                "rattle_stdev": 0.28333683853583164,
                            },
                        },
                    },
                }
            },
        }
        # check defects from old metadata file are in new metadata file
        with open("distortion_metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
        np.testing.assert_equal(metadata, spec_coords_V_Cd_dict)

    def test_snb_generate_config(self):
        # test config file:
        test_yml = """
distortion_increment: 0.05
stdev: 0.15
d_min: 2.1250262890187375  # 0.75 * 2.8333683853583165
nbr_cutoff: 3.4
n_iter: 3
active_atoms: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # np.arange(0,31)
width: 0.3
max_attempts: 10000
max_disp: 1.0
seed: 20
local_rattle: False"""
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c 0",
                "-v",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        V_Cd_kwarged_POSCAR = Poscar.from_file(
            "Vac_Cd_mult32_0/Bond_Distortion_-50.0%/POSCAR"
        )
        self.assertEqual(
            V_Cd_kwarged_POSCAR.structure, self.V_Cd_minus0pt5_struc_kwarged
        )

        test_yml = """
oxidation_states:
  Cd: 3
  Te: -3
        """
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c 0",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Auto site-matching identified", result.output)
        self.assertNotIn("Oxidation states were not explicitly set", result.output)
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ["
            "'-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', "
            "'0.3', '0.4', '0.5', '0.6']. Then, will rattle with a std dev of 0.28 Å",
            result.output,
        )
        self.assertIn("Defect: Vac_Cd_mult32", result.output)
        self.assertIn("Number of missing electrons in neutral state: 3", result.output)
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: 0. Number of distorted neighbours: 3",
            result.output,
        )
        V_Cd_ox3_POSCAR = Poscar.from_file(
            "Vac_Cd_mult32_0/Bond_Distortion_-50.0%/POSCAR"
        )
        self.assertNotEqual(
            V_Cd_ox3_POSCAR.structure, self.V_Cd_minus0pt5_struc_local_rattled
        )

        self.tearDown()
        test_yml = """
distortion_increment: 0.25
name: Int_Cd_2
dict_number_electrons_user:
  Int_Cd_2:
    3
distorted_elements:
  Int_Cd_2:
    Cd
local_rattle: False
"""
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)

        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Int_Cd_2_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c",
                "1",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Auto site-matching identified", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ['-0.5', "
            "'-0.25', '0.0', '0.25', '0.5']. Then, will rattle with a std dev of 0.28 Å",
            result.output,
        )
        self.assertIn("Defect: Int_Cd_2", result.output)
        self.assertIn("Number of missing electrons in neutral state: 3", result.output)
        self.assertIn(
            "Defect Int_Cd_2 in charge state: +1. Number of distorted neighbours: 4",
            result.output,
        )

        reduced_Int_Cd_2_dict = self.Int_Cd_2_dict.copy()
        reduced_Int_Cd_2_dict["charges"] = [1]
        kwarged_Int_Cd_2_dict = {
            "distortion_parameters": {
                "distortion_increment": 0.25,
                "bond_distortions": [-0.5, -0.25, 0.0, 0.25, 0.5],
                "rattle_stdev": 0.28333683853583164,
                "local_rattle": False,
            },
            "defects": {
                "Int_Cd_2": {
                    "unique_site": reduced_Int_Cd_2_dict["bulk_supercell_site"]
                    .frac_coords.round(4)
                    .tolist(),
                    "charges": {
                        "1": {  # json converts integer strings to keys
                            "num_nearest_neighbours": 4,
                            "distorted_atoms": [
                                [10, "Cd"],
                                [22, "Cd"],
                                [29, "Cd"],
                                [1, "Cd"],
                            ],
                            "distortion_parameters": {
                                "bond_distortions": [
                                    -0.5,
                                    -0.25,
                                    0.0,
                                    0.25,
                                    0.5,
                                ],
                                "rattle_stdev": 0.28333683853583164,
                            },
                        },
                    },
                    "defect_site_index": 65,
                }
            },
        }
        # check defects from old metadata file are in new metadata file
        with open("distortion_metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
        np.testing.assert_equal(metadata, kwarged_Int_Cd_2_dict)

        self.tearDown()
        test_yml = """
        bond_distortions: [-0.5, -0.25, 0.0, 0.25, 0.5]
        name: Wally_McDoodle
        local_rattle: True
        stdev: 0.25
        seed: 42
        """  # previous default
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)

        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c",
                "0",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("Auto site-matching identified", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ['-0.5', '-0.25', "
            "'0.0', '0.25', '0.5']. Then, will rattle with a std dev of 0.25 Å",
            result.output,
        )
        self.assertIn("Defect: Wally_McDoodle", result.output)
        self.assertIn("Number of missing electrons in neutral state: 2", result.output)
        self.assertIn(
            "Defect Wally_McDoodle in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        # check if correct files were created:
        V_Cd_Bond_Distortion_folder = "Wally_McDoodle_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_Bond_Distortion_folder))
        self.assertTrue(os.path.exists("Wally_McDoodle_0/Unperturbed"))
        V_Cd_minus0pt5_rattled_POSCAR = Poscar.from_file(
            V_Cd_Bond_Distortion_folder + "/POSCAR"
        )
        self.assertEqual(
            V_Cd_minus0pt5_rattled_POSCAR.comment,
            "-50.0%__num_neighbours=2__Wally_McDoodle",
        )  # default
        self.assertEqual(
            V_Cd_minus0pt5_rattled_POSCAR.structure,
            self.V_Cd_minus0pt5_struc_local_rattled,
        )

        # test min, max charge setting:
        self.tearDown()
        test_yml = """
                min_charge: -7
                max_charge: -5
                """
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)

        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: -7. Number of distorted neighbours: 3",
            result.output,
        )
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: -6. Number of distorted neighbours: 4",
            result.output,
        )
        self.assertIn(
            "Defect Vac_Cd_mult32 in charge state: -5. Number of distorted neighbours: 3",
            result.output,
        )
        self.assertNotIn("Defect Vac_Cd_mult32 in charge state: -4", result.output)
        self.assertNotIn("Defect Vac_Cd_mult32 in charge state: 0", result.output)
        self.assertTrue(os.path.exists("Vac_Cd_mult32_-7"))
        self.assertTrue(os.path.exists("Vac_Cd_mult32_-6"))
        self.assertTrue(os.path.exists("Vac_Cd_mult32_-5"))
        self.assertTrue(os.path.exists("Vac_Cd_mult32_-5/Unperturbed"))
        self.assertTrue(os.path.exists("Vac_Cd_mult32_-5/Bond_Distortion_40.0%"))
        self.assertFalse(os.path.exists("Vac_Cd_mult32_-4"))

        # test priority (CLI > config)
        self.tearDown()
        test_yml = """
charge: 1
stdev: 0.25
seed: 42
        """  # previous default
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)

        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c",
                "0",
                "--name",
                "vac_1_Cd",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Defect vac_1_Cd in charge state: 0", result.output)
        self.assertNotIn("Defect vac_1_Cd in charge state: +1", result.output)
        # test parsed defects json
        parsed_defects_dict = loadfn("parsed_defects_dict.json")
        for key in [
            "name",
            "defect_type",
            "site_multiplicity",
            "site_specie",
            "unique_site",
        ]:
            self.assertEqual(
                parsed_defects_dict["vacancies"][0][key],
                self.cdte_defect_dict["vacancies"][0][key],
            )

        # Test non-sense key in config - should be ignored
        # and not feed into Distortions()
        test_yml = """charges: [0,]
defect_coords: [0,0,0]
bond_distortions: [0.3,]
name: vac_1_Cd
local_rattle: False
nonsense_key: nonsense_value"""
        with open("test_config.yml", "w") as fp:
            fp.write(test_yml)
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c 0",
                "--config",
                "test_config.yml",
                "--name",
                "vac_1_Cd",  # to match saved json
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Defect vac_1_Cd in charge state: 0", result.output)
        self.tearDown()

    def test_snb_generate_all(self):
        """Test generate_all() function."""
        # Test parsing defects from folders with non-standard names and default charge states
        # Also test local rattle parameter
        # Create a folder for defect files / directories
        defects_dir = "pesky_defects"
        defect_name = "vac_1_Cd"
        os.mkdir(defects_dir)
        os.mkdir(f"{defects_dir}/{defect_name}")  # non-standard defect name
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}/POSCAR",
        )
        # CONFIG file
        test_yml = """bond_distortions: [0.3,]
local_rattle: True
stdev: 0.25
seed: 42"""  # previous default
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)
        with warnings.catch_warnings(record=True) as w:
            runner = CliRunner()
            result = runner.invoke(
                snb,
                [
                    "generate_all",
                    "-d",
                    f"{defects_dir}/",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "-v",
                    "--config",
                    "test_config.yml",
                ],
                catch_exceptions=False,
            )
        # Test outputs
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto site-matching identified", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            f"No charge (range) set for defect {defect_name} in config file,"
            " assuming default range of +/-2",
            str(w[0].message),
        )
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ['0.3']."
            " Then, will rattle with a std dev of 0.25 Å",
            result.output,
        )
        self.assertIn(f"Defect: {defect_name}", result.output)
        self.assertIn("Number of missing electrons in neutral state: 2", result.output)
        # Charge states
        self.assertIn(
            f"Defect {defect_name} in charge state: -2. Number of distorted neighbours: 0",
            result.output,
        )
        self.assertIn(
            f"Defect {defect_name} in charge state: -1. Number of distorted neighbours: 1",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(3.68, 33, 'Te')]",
            result.output,
        )
        self.assertIn(
            f"Defect {defect_name} in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(3.68, 33, 'Te'), (3.68, 42, 'Te')]",
            result.output,
        )
        self.assertIn(
            f"Defect {defect_name} in charge state: +1. Number of distorted neighbours: 3",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te'), (2.83, 52, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(3.68, 33, 'Te'), (3.68, 42, 'Te'), (3.68, 52, 'Te')]",
            result.output,
        )
        self.assertIn(
            f"Defect {defect_name} in charge state: +2. Number of distorted neighbours: 4",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te'), (2.83, 52, 'Te'), (2.83, 63, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(3.68, 33, 'Te'), (3.68, 42, 'Te'), (3.68, 52, 'Te'), (3.68, 63, 'Te')]",
            result.output,
        )
        for charge in range(-1, 3):
            for dist in ["Unperturbed", "Bond_Distortion_30.0%"]:
                self.assertTrue(os.path.exists(f"{defect_name}_{charge}/{dist}/POSCAR"))
        for dist in ["Unperturbed", "Rattled"]:
            # -2 has 0 electron change -> only Unperturbed & rattled folders
            self.assertTrue(os.path.exists(f"{defect_name}_-2/{dist}/POSCAR"))
        # check POSCAR
        self.assertEqual(
            Structure.from_file(f"{defect_name}_0/Bond_Distortion_30.0%/POSCAR"),
            self.V_Cd_0pt3_local_rattled,
        )
        if_present_rm(defects_dir)
        for charge in range(-2, 3):
            if_present_rm(f"{defect_name}_{charge}")
        self.tearDown()

        # Test defects not organised in folders
        # Test defect_settings (charges, defect index/coords)
        defect_name = "Vac_Cd"
        os.mkdir(defects_dir)
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}_POSCAR",
        )
        # CONFIG file
        test_yml = f"""
        defects:
            {defect_name}:
                charges: [0,]
                defect_coords: [0.0, 0.0, 0.0]
        bond_distortions: [0.3,]
        local_rattle: True
        stdev: 0.25
        seed: 42
        """  # previous default
        with open("test_config.yml", "w") as fp:
            fp.write(test_yml)
        with warnings.catch_warnings(record=True) as w:
            runner = CliRunner()
            result = runner.invoke(
                snb,
                [
                    "generate_all",
                    "-d",
                    f"{defects_dir}/",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "-v",
                    "--config",
                    "test_config.yml",
                ],
                catch_exceptions=False,
            )
        # Test outputs
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto site-matching identified", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        # self.assertFalse(w) # no warnings (charges set in config file)
        # Only neutral charge state
        self.assertNotIn(
            f"Defect {defect_name} in charge state: -1. Number of distorted neighbours: 1",
            result.output,
        )
        self.assertIn(
            f"Defect {defect_name} in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(3.68, 33, 'Te'), (3.68, 42, 'Te')]",
            result.output,
        )
        for dist in ["Unperturbed", "Bond_Distortion_30.0%"]:
            self.assertTrue(os.path.exists(f"{defect_name}_0/{dist}/POSCAR"))
        self.assertFalse(os.path.exists(f"{defect_name}_-1/Unperturbed/POSCAR"))
        # check POSCAR
        self.assertEqual(
            Structure.from_file(f"{defect_name}_0/Bond_Distortion_30.0%/POSCAR"),
            self.V_Cd_0pt3_local_rattled,
        )
        if_present_rm(f"{defect_name}_0")
        self.tearDown()

        # Test wrong names in config file (different from defect folders)
        # Names of defect folders/files should be used as defect names
        # CONFIG file
        # Create a folder for defect files / directories
        defects_dir = "pesky_defects"
        defect_name = "Vac_Cd"
        os.mkdir(defects_dir)
        os.mkdir(f"{defects_dir}/{defect_name}")  # non-standard defect name
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}/POSCAR",
        )
        wrong_defect_name = "Wally_McDoodle"
        test_yml = f"""
        defects:
            {wrong_defect_name}:
                charges: [0,]
                defect_coords: [0.0, 0.0, 0.0]
        bond_distortions: [0.3,]
        """
        with open("test_config.yml", "w") as fp:
            fp.write(test_yml)
        with warnings.catch_warnings(record=True) as w:
            runner = CliRunner()
            result = runner.invoke(
                snb,
                [
                    "generate_all",
                    "-d",
                    f"{defects_dir}/",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "-v",
                    "--config",
                    "test_config.yml",
                ],
                catch_exceptions=False,
            )
        # Test outputs
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto site-matching identified", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            f"Defect {defect_name} not found in config file test_config.yml. "
            f"Will parse defect name from folders/files.",
            str(w[0].message),
        )  # Defect name not parsed from config
        self.assertEqual(w[1].category, UserWarning)
        self.assertEqual(
            f"No charge (range) set for defect {defect_name} in config file,"
            " assuming default range of +/-2",
            str(w[1].message),
        )
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ['0.3']. Then, "
            "will rattle with a std dev of 0.28 Å",
            result.output,  # test auto-determined stdev and bond length
        )
        # Only neutral charge state
        self.assertIn(
            f"Defect {defect_name} in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            "\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(3.68, 33, 'Te'), (3.68, 42, 'Te')]",
            result.output,
        )
        for dist in ["Unperturbed", "Bond_Distortion_30.0%"]:
            self.assertTrue(os.path.exists(f"{defect_name}_0/{dist}/POSCAR"))
        self.tearDown()

        # Test wrong folder defect name
        defects_dir = "pesky_defects"
        defect_name = "Wally_McDoodle"
        os.mkdir(defects_dir)
        os.mkdir(f"{defects_dir}/{defect_name}")  # non-standard defect name
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}/POSCAR",
        )
        right_defect_name = "Vac_Cd"
        test_yml = f"""
        defects:
            {right_defect_name}:
                charges: [0,]
                defect_coords: [0.0, 0.0, 0.0]
        bond_distortions: [0.3,]
        """
        with open("test_config.yml", "w") as fp:
            fp.write(test_yml)
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate_all",
                "-d",
                f"{defects_dir}/",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-v",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=True,
        )
        # Test outputs
        self.assertIsInstance(result.exception, ValueError)
        self.assertIn(
            "Error in defect name parsing; could not parse defect name",
            str(result.exception),
        )
        # The input_file option is tested in local test, as INCAR
        # not written in Github Actions

    def test_run(self):
        """Test snb-run function"""
        os.chdir(self.VASP_TIO2_DATA_DIR)
        proc = subprocess.Popen(
            ["snb-run", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = str(proc.communicate()[0])
        self.assertIn(
            "Job file 'job' not found, so will only submit jobs in folders with "
            "'job' present",
            out,
        )
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)  # also present
        # but no OUTCAR so shouldn't print message

        # test job submit command
        with open("job_file", "w") as fp:
            fp.write("Test pop")
        proc = subprocess.Popen(
            ["snb-run", "-v", "-s echo", "-n this", "-j job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertNotIn(
            "Job file 'job_file' not found, so will only submit jobs in folders with "
            "'job_file' present",
            out,
        )
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn(
            "Bond_Distortion_10.0% fully relaxed", out
        )  # also present but no OUTCAR
        self.assertIn("Running job for Bond_Distortion_10.0%", out)
        self.assertIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%/job_file"))

        if_present_rm("job_file")
        if_present_rm("Bond_Distortion_10.0%/job_file")

        # test job file in different directory
        with open("../job_file", "w") as fp:
            fp.write("Test pop")
        proc = subprocess.Popen(
            ["snb-run", "-v", "-s echo", "-n this", "-j ../job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertNotIn(
            "Job file '../job_file' not found, so will only submit jobs in folders with "
            "'job_file' present",
            out,
        )
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn(
            "Bond_Distortion_10.0% fully relaxed", out
        )  # also present but no OUTCAR
        self.assertIn("Running job for Bond_Distortion_10.0%", out)
        self.assertIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%/job_file"))

        if_present_rm("../job_file")
        if_present_rm("Bond_Distortion_10.0%/job_file")

        # test save_vasp_files:
        with open("Bond_Distortion_10.0%/OUTCAR", "w") as fp:
            fp.write("Test pop")
        proc = subprocess.Popen(
            ["snb-run", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = str(proc.communicate()[0])
        self.assertIn(
            "Bond_Distortion_10.0% not (fully) relaxed, saving files and rerunning", out
        )
        files = os.listdir("Bond_Distortion_10.0%")
        saved_files = [file for file in files if "on" in file and "CAR_" in file]
        self.assertEqual(len(saved_files), 2)
        self.assertEqual(len([i for i in saved_files if "CONTCAR" in i]), 1)
        self.assertEqual(len([i for i in saved_files if "OUTCAR" in i]), 1)
        for i in saved_files:
            os.remove(f"Bond_Distortion_10.0%/{i}")
        os.remove("Bond_Distortion_10.0%/OUTCAR")
        os.remove("Bond_Distortion_10.0%/POSCAR")

        # test "--all" option
        os.chdir("..")
        shutil.copytree("vac_1_Ti_0", "vac_1_Ti_1")
        proc = subprocess.Popen(
            ["snb-run", "-v", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = str(proc.communicate()[0])
        self.assertIn("Looping through distortion folders for vac_1_Ti_0", out)
        self.assertIn("Looping through distortion folders for vac_1_Ti_1", out)
        shutil.rmtree("vac_1_Ti_1")

        os.chdir("..")
        proc = subprocess.Popen(
            ["snb-run", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = str(proc.communicate()[0])
        self.assertIn("No distortion folders found in current directory", out)

        proc = subprocess.Popen(
            ["snb-run", "-v", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = str(proc.communicate()[0])
        self.assertIn(
            "No defect folders (with names ending in a number (charge state)) found in "
            "current directory",
            out,
        )

        # test ignoring and renaming when positive energies present in OUTCAR
        os.chdir(self.VASP_TIO2_DATA_DIR)
        with open("job_file", "w") as fp:
            fp.write("Test pop")
        positive_energies_outcar_string = """
energy  without entropy=     1156.08478433  energy(sigma->0) =     1156.08478433
energy  without entropy=     2923.36313118  energy(sigma->0) =     2923.36252910
energy  without entropy=     3785.53283598  energy(sigma->0) =     3785.53033686
energy  without entropy=     2944.54877982  energy(sigma->0) =     2944.54877982
energy  without entropy=     5882.47593917  energy(sigma->0) =     5882.47494166
energy  without entropy=      762.73605542  energy(sigma->0) =      762.73605542
energy  without entropy=      675.21988502  energy(sigma->0) =      675.21988502
energy  without entropy=      661.30956300  energy(sigma->0) =      661.30956300
energy  without entropy=        9.90115363  energy(sigma->0) =        9.90115363
energy  without entropy=        9.11186084  energy(sigma->0) =        9.11186084
energy  without entropy=        7.99185422  energy(sigma->0) =        7.99185422
"""
        with open("Bond_Distortion_10.0%/OUTCAR", "w") as fp:
            fp.write(positive_energies_outcar_string)
        proc = subprocess.Popen(
            ["snb-run", "-v", "-s echo", "-n this", "-j job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)  # also present
        self.assertNotIn("Running job for Bond_Distortion_10.0%", out)
        self.assertNotIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertIn(
            "Positive energies or forces error encountered for Bond_Distortion_10.0%, ignoring and "
            "renaming to Bond_Distortion_10.0%_High_Energy",
            out,
        )
        self.assertFalse(os.path.exists("Bond_Distortion_10.0%"))
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%_High_Energy"))
        if_present_rm("Bond_Distortion_10.0%_High_Energy/job_file")
        if_present_rm("Bond_Distortion_10.0%_High_Energy/OUTCAR")

        # run again to test ignoring *High_Energy* folder(s)
        proc = subprocess.Popen(
            ["snb-run", "-v", "-s echo", "-n this", "-j job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)  # also present
        self.assertNotIn("Running job for Bond_Distortion_10.0%", out)
        self.assertNotIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertNotIn(
            "Positive energies or forces encountered for Bond_Distortion_10.0%, ignoring and "
            "renaming to Bond_Distortion_10.0%_High_Energy",
            out,
        )
        self.assertFalse("High_Energy" in out)
        self.assertFalse(os.path.exists("Bond_Distortion_10.0%"))
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%_High_Energy"))
        # don't need to remove any files as Bond_Distortion_10.0%_High_Energy has been ignored
        os.rename("Bond_Distortion_10.0%_High_Energy", "Bond_Distortion_10.0%")

        # test runs fine when only positive energies for first 3 ionic steps
        positive_energies_outcar_string = """
        energy  without entropy=     1156.08478433  energy(sigma->0) =     1156.08478433
        energy  without entropy=     2923.36313118  energy(sigma->0) =     2923.36252910
        energy  without entropy=     3785.53283598  energy(sigma->0) =     3785.53033686
        energy  without entropy=     2944.54877982  energy(sigma->0) =     -2944.54877982
        energy  without entropy=     5882.47593917  energy(sigma->0) =     -5882.47494166
        energy  without entropy=      762.73605542  energy(sigma->0) =      -762.73605542
        energy  without entropy=      675.21988502  energy(sigma->0) =      -675.21988502
        """
        with open("Bond_Distortion_10.0%/OUTCAR", "w") as fp:
            fp.write(positive_energies_outcar_string)
        proc = subprocess.Popen(
            ["snb-run", "-v", "-s echo", "-n this", "-j job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)  # also present
        self.assertIn("Running job for Bond_Distortion_10.0%", out)
        self.assertIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertNotIn(
            "Positive energies or forces error encountered for Bond_Distortion_10.0%, ignoring and "
            "renaming to Bond_Distortion_10.0%_High_Energy",
            out,
        )
        self.assertFalse(os.path.exists("Bond_Distortion_10.0%_High_Energy"))
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%"))
        self.assertIn(
            "Bond_Distortion_10.0% not (fully) relaxed, saving files and rerunning", out
        )
        files = os.listdir("Bond_Distortion_10.0%")
        saved_files = [file for file in files if "on" in file and "CAR_" in file]
        self.assertEqual(len(saved_files), 2)
        self.assertEqual(len([i for i in saved_files if "CONTCAR" in i]), 1)
        self.assertEqual(len([i for i in saved_files if "OUTCAR" in i]), 1)
        for i in saved_files:
            os.remove(f"Bond_Distortion_10.0%/{i}")
        os.remove("Bond_Distortion_10.0%/OUTCAR")
        os.remove("Bond_Distortion_10.0%/POSCAR")
        if_present_rm("Bond_Distortion_10.0%/job_file")

        # test ignoring and renaming when forces error in OUTCAR
        def _test_OUTCAR_error(error_string):
            negative_energies_w_error_outcar_string = f"""
energy  without entropy=     -1156.08478433  energy(sigma->0) =     -1156.08478433
energy  without entropy=     -2923.36313118  energy(sigma->0) =     -2923.36252910
energy  without entropy=     -3785.53283598  energy(sigma->0) =     -3785.53033686
energy  without entropy=     -2944.54877982  energy(sigma->0) =     -2944.54877982
energy  without entropy=     -5882.47593917  energy(sigma->0) =     -5882.47494166
energy  without entropy=      -762.73605542  energy(sigma->0) =     -762.73605542
Chosen VASP error message: {error_string}
"""
            with open("Bond_Distortion_10.0%/OUTCAR", "w") as fp:
                fp.write(negative_energies_w_error_outcar_string)
            proc = subprocess.Popen(
                ["snb-run", "-v", "-s echo", "-n this", "-j job_file"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )  # setting 'job command' to 'echo' to
            out = str(proc.communicate()[0])
            self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
            self.assertIn("Unperturbed fully relaxed", out)
            self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)  # also present
            self.assertNotIn("Running job for Bond_Distortion_10.0%", out)
            self.assertNotIn(
                "this vac_1_Ti_0_10.0% job_file", out
            )  # job submit command
            self.assertIn(
                "Positive energies or forces error encountered for Bond_Distortion_10.0%, "
                "ignoring and renaming to Bond_Distortion_10.0%_High_Energy",
                out,
            )
            self.assertFalse(os.path.exists("Bond_Distortion_10.0%"))
            self.assertTrue(os.path.exists("Bond_Distortion_10.0%_High_Energy"))
            if_present_rm("job_file")
            if_present_rm("Bond_Distortion_10.0%_High_Energy/job_file")
            if_present_rm("Bond_Distortion_10.0%_High_Energy/OUTCAR")
            os.rename("Bond_Distortion_10.0%_High_Energy", "Bond_Distortion_10.0%")

        for error in [
            "EDDDAV",
            "ZHEGV",
            "CNORMN",
            "ZPOTRF",
            "ZTRTRI",
        ]:
            _test_OUTCAR_error(error)

        # test treating-as-converged when energy change less than 2 meV and >50 ionic steps
        residual_forces_outcar_string = (
            """
            energy  without entropy=      675.21988502  energy(sigma->0) =      -675.21988502
            """
            * 70
        )
        with open("Bond_Distortion_10.0%/OUTCAR", "w") as fp:
            fp.write(residual_forces_outcar_string)
        proc = subprocess.Popen(
            ["snb-run", "-v", "-s echo", "-n this", "-j job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)
        self.assertNotIn(
            "Running job for Bond_Distortion_10.0%", out
        )  # not running job though!
        self.assertNotIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertNotIn(
            "Positive energies or forces error encountered for Bond_Distortion_10.0%, "
            "ignoring and renaming to Bond_Distortion_10.0%_High_Energy",
            out,
        )
        self.assertIn(
            "Bond_Distortion_10.0% has some (small) residual forces but energy converged "
            "to < 2 meV, considering this converged.",
            out,
        )
        self.assertFalse(os.path.exists("Bond_Distortion_10.0%_High_Energy"))
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%"))
        self.assertNotIn(
            "Bond_Distortion_10.0% not (fully) relaxed, saving files and rerunning", out
        )
        files = os.listdir("Bond_Distortion_10.0%")
        saved_files = [file for file in files if "on" in file and "CAR_" in file]
        self.assertEqual(len(saved_files), 0)

        # Test no warning when not verbose
        proc = subprocess.Popen(
            ["snb-run", "-s echo", "-n this", "-j job_file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # setting 'job command' to 'echo' to
        out = str(proc.communicate()[0])
        self.assertNotIn(
            "Bond_Distortion_10.0% has some (small) residual forces but energy "
            "converged to < 2 meV, considering this converged.",
            out,
        )
        os.remove("Bond_Distortion_10.0%/OUTCAR")
        if_present_rm("Bond_Distortion_10.0%/job_file")

    def test_parse(self):
        """Test parse() function.
        Implicitly, this also tests the io.parse_energies() function"""
        # Specifying defect to parse
        # All OUTCAR's present in distortion directories
        # Energies file already present
        defect = "vac_1_Ti_0"
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml", "w") as f:
            f.write("")
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                self.EXAMPLE_RESULTS,
            ],
            catch_exceptions=False,
        )
        self.assertIn(
            f"Moving old {self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml to ",
            result.output,
        )
        energies = loadfn(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml")
        test_energies = {
            "distortions": {
                -0.4: -1176.28458753,
            },
            "Unperturbed": -1173.02056574,
        }  # Using dictionary here (rather than file/string), because parsing order
        # is difference on github actions
        self.assertEqual(test_energies, energies)
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]

        # Test when OUTCAR not present in one of the distortion directories
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "parse",
                    "-d",
                    defect,
                    "-p",
                    self.VASP_DIR,
                ],
                catch_exceptions=False,
            )
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            "No output file in Bond_Distortion_10.0% directory",
            str(w[0].message),
        )
        self.assertTrue(os.path.exists(f"{self.VASP_DIR}/{defect}/{defect}.yaml"))
        energies = loadfn(f"{self.VASP_DIR}/{defect}/{defect}.yaml")
        test_energies = {
            "distortions": {-0.4: -1176.28458753},
            "Unperturbed": -1173.02056574,
        }
        self.assertEqual(test_energies, energies)
        os.remove(f"{self.VASP_DIR}/{defect}/{defect}.yaml")

        # Test --all option
        self.tearDown()
        os.mkdir(f"{self.EXAMPLE_RESULTS}/pesky_defects")
        defect_name = "vac_1_Ti_-1"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0",
            f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}",
        )
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0",
            f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0",
        )
        result = runner.invoke(
            snb,
            [
                "parse",
                "--all",
                "-p",
                f"{self.EXAMPLE_RESULTS}/pesky_defects",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.yaml"
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0/vac_1_Ti_0.yaml"
            )
        )

        # Test parsing from inside the defect folder
        defect_name = "vac_1_Ti_-1"
        os.remove(
            f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.yaml"
        )
        os.chdir(f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}")
        result = runner.invoke(
            snb,
            [
                "parse",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.yaml"
            )
        )
        os.chdir(file_path)

        # Test warning when setting path and parsing from inside the defect folder
        defect_name = "vac_1_Ti_-1"
        os.remove(
            f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.yaml"
        )
        os.chdir(f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}")
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "parse",
                    "-p",
                    self.EXAMPLE_RESULTS,
                ],
                catch_exceptions=False,
            )
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == "`--path` option ignored when running from within defect folder (i.e. "
                    "when `--defect` is not specified."
                    for warning in w
                ]
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.yaml"
            )
        )
        os.chdir(file_path)
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/pesky_defects/")

        # Test when `defect` is present higher up in `path`
        defect_name = "vac_1_Ti_0"
        os.mkdir(f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder")
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/{defect_name}",
            f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder/{defect_name}",
        )
        result = runner.invoke(
            snb,
            [
                "parse",
                "-p",
                f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder/{defect_name}",
                "-d",
                defect_name,
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder/{defect_name}"
                f"/{defect_name}.yaml"
            )
        )

        # test warning when nothing parsed because defect folder not recognised
        os.chdir(self.EXAMPLE_RESULTS)
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb, ["parse", "-d", "defect"], catch_exceptions=True
            )
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == "Energies could not be parsed for defect 'defect' in '.'. If these "
                    "directories are correct, check calculations have converged, and that "
                    "distortion subfolders match ShakeNBreak naming (e.g. "
                    "Bond_Distortion_xxx, Rattled, Unperturbed)"
                    for warning in w
                ]
            )
        )
        self.assertFalse(
            any(os.path.exists(i) for i in os.listdir() if i.endswith(".yaml"))
        )
        os.chdir(file_path)

        # Test warning when run with no arguments in top-level folder
        os.chdir(self.EXAMPLE_RESULTS)
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(snb, ["parse"], catch_exceptions=True)
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == "Energies could not be parsed for defect 'example_results' in"
                    f" '{self.DATA_DIR}'. If these directories are correct, "
                    "check calculations have converged, and that distortion subfolders match "
                    "ShakeNBreak naming (e.g. Bond_Distortion_xxx, Rattled, Unperturbed)"
                    for warning in w
                ]
            )
        )
        self.assertFalse(
            any(os.path.exists(i) for i in os.listdir() if i.endswith(".yaml"))
        )
        os.chdir(file_path)

        # test ignoring "*High_Energy*" folder(s)
        defect = "vac_1_Ti_0"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-40.0%",
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_High_Energy",
        )
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                self.EXAMPLE_RESULTS,
            ],
            catch_exceptions=False,
        )
        energies = loadfn(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml")
        self.assertEqual(test_energies, energies)  # no Bond_Distortion_-20.0% results
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]
        shutil.rmtree(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_High_Energy"
        )

        # test parsing energies of calculations that still haven't converged
        defect = "vac_1_Ti_0"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-40.0%",
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_not_converged",
        )
        with open(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_not_converged/OUTCAR",
            "r",
        ) as f:
            lines = f.readlines()
            truncated = lines[:5000]

        with open(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_not_converged/OUTCAR",
            "w+",
        ) as trout:
            for line in truncated:
                trout.write(line)
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                self.EXAMPLE_RESULTS,
            ],
            catch_exceptions=False,
        )
        energies = loadfn(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml")
        self.assertNotEqual(
            test_energies, energies
        )  # Bond_Distortion_-20.0%_not_converged now included
        not_converged_energies = copy.deepcopy(test_energies)
        not_converged_energies["distortions"].update(
            {"Bond_Distortion_-20.0%_not_converged": -1151.8383839}
        )
        self.assertEqual(
            not_converged_energies, energies
        )  # Bond_Distortion_-20.0%_not_converged now included
        # test print statement about not being fully relaxed
        self.assertIn(
            "Bond_Distortion_-20.0%_not_converged not fully relaxed", result.output
        )
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]
        shutil.rmtree(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_not_converged"
        )

        # test parsing energies of residual-forces calculations
        defect = "vac_1_Ti_0"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-40.0%",
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_residual_forces",
        )
        residual_forces_outcar_string = (
            (
                """
                    energy  without entropy=      675.21988502  energy(sigma->0) =    -675.21988502
                """
                * 70
            )
            + """ShakeNBreak: At least 50 ionic steps and energy change < 2 meV for this 
            defect, considering this converged."""
        )
        with open(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_residual_forces/OUTCAR",
            "w+",
        ) as f:
            f.write(residual_forces_outcar_string)

        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                self.EXAMPLE_RESULTS,
            ],
            catch_exceptions=False,
        )
        energies = loadfn(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml")
        self.assertNotEqual(
            test_energies, energies
        )  # Bond_Distortion_-20.0%_residual_forces now included
        residual_forces_energies = copy.deepcopy(test_energies)
        residual_forces_energies["distortions"].update(
            {"Bond_Distortion_-20.0%_residual_forces": -675.21988502}
        )
        self.assertEqual(
            residual_forces_energies, energies
        )  # Bond_Distortion_-20.0%_residual_forces now included
        # test no print statement about not being fully relaxed
        self.assertNotIn("not fully relaxed", result.output)
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]
        shutil.rmtree(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-20.0%_residual_forces"
        )

        # test warning when all parsed distortions are >0.1 eV higher energy than unperturbed
        defect = "vac_1_Ti_3"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0",
            f"{self.EXAMPLE_RESULTS}/{defect}",
        )
        high_energy_outcar_string = """
        energy  without entropy=      -675.21988502  energy(sigma->0) =      -675.21988502
        """  # unperturbed final energy is -1173.02056574 eV
        with open(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-40.0%/OUTCAR",
            "w+",
        ) as f:
            f.write(high_energy_outcar_string)

        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "parse",
                    "-d",
                    defect,
                    "-p",
                    self.EXAMPLE_RESULTS,
                ],
                catch_exceptions=False,
            )
        energies = loadfn(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml")
        high_energies_dict = {
            "distortions": {-0.4: -675.21988502},
            "Unperturbed": -1173.02056574,
        }  # energies still parsed despite being high and not converged (like myself)
        self.assertEqual(
            high_energies_dict, energies
        )  # energies still parsed, but all high energy
        # test print statement about not being fully relaxed
        self.assertIn("not fully relaxed", result.output)
        self.assertTrue(len([i for i in w if i.category == UserWarning]) == 1)
        self.assertTrue(
            any(
                [
                    f"All distortions parsed for {defect} are >0.1 eV higher energy than "
                    f"unperturbed, indicating problems with the relaxations. You should first "
                    f"check if the calculations finished ok for this defect species and if this "
                    f"defect charge state is reasonable (often this is the result of an "
                    f"unreasonable charge state). If both checks pass, you likely need to adjust "
                    f"the `stdev` rattling parameter (can occur for hard/ionic/magnetic "
                    f"materials); see "
                    f"https://shakenbreak.readthedocs.io/en/latest/Tips.html#hard-ionic-materials. "
                    f"– This often indicates a complex PES with multiple minima, "
                    f"thus energy-lowering distortions particularly likely, so important to "
                    f"test with reduced `stdev`!" == str(i.message)
                    for i in w
                    if i.category == UserWarning
                ]
            )
        )
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/{defect}")

        # test warning when all distortions have been renamed to "*High_Energy*"
        defect = "vac_1_Ti_3"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0",
            f"{self.EXAMPLE_RESULTS}/{defect}",
        )
        shutil.move(
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-40.0%",
            f"{self.EXAMPLE_RESULTS}/{defect}/Bond_Distortion_-40.0%_High_Energy",
        )
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]  # remove yaml files so we reparse the energies
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "parse",
                    "-d",
                    defect,
                    "-p",
                    self.EXAMPLE_RESULTS,
                ],
                catch_exceptions=False,
            )
        self.assertFalse(
            os.path.exists(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml")
        )
        # test print statement about not being fully relaxed
        self.assertNotIn("not fully relaxed", result.output)
        self.assertTrue(len([i for i in w if i.category == UserWarning]) == 1)
        print([i.message for i in w])
        print(result.output)
        self.assertTrue(
            any(
                [
                    f"All distortions for {defect} gave positive energies or forces errors, "
                    "indicating problems with these relaxations. You should first check that no "
                    "user INCAR setting is causing this issue. If not, you likely need to adjust "
                    "the `stddev` rattling parameter (can occur for hard/ionic/magnetic "
                    "materials); see https://shakenbreak.readthedocs.io/en/latest/Tips.html#hard"
                    "-ionic-materials." == str(i.message)
                    for i in w
                    if i.category == UserWarning
                ]
            )
        )
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/{defect}")

    def test_parse_codes(self):
        """Test parse() function when using codes different from VASP."""
        runner = CliRunner()
        defect = "vac_1_Cd_0"

        # CP2K
        code = "cp2k"
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                f"{self.DATA_DIR}/{code}",
                "-c",
                code,
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")
        )
        with open(
            f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml", "r"
        ) as test, open(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml", "r") as new:
            test_yaml = yaml.safe_load(test)
            new_yaml = yaml.safe_load(new)
        self.assertDictEqual(test_yaml, new_yaml)
        os.remove(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")

        # CASTEP
        code = "castep"
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                f"{self.DATA_DIR}/{code}",
                "-c",
                code,
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")
        )
        with open(
            f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml", "r"
        ) as test, open(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml", "r") as new:
            test_yaml = yaml.safe_load(test)
            new_yaml = yaml.safe_load(new)
        self.assertDictEqual(test_yaml, new_yaml)
        os.remove(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")

        # Espresso
        code = "quantum_espresso"
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                f"{self.DATA_DIR}/{code}",
                "-c",
                "espresso",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")
        )
        with open(
            f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml", "r"
        ) as test, open(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml", "r") as new:
            test_yaml = yaml.safe_load(test)
            new_yaml = yaml.safe_load(new)
        self.assertDictEqual(test_yaml, new_yaml)
        os.remove(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")

        # FHI-aims
        code = "fhi_aims"
        result = runner.invoke(
            snb,
            [
                "parse",
                "-d",
                defect,
                "-p",
                f"{self.DATA_DIR}/{code}",
                "-c",
                "fhi-aims",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")
        )
        with open(
            f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml", "r"
        ) as test, open(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml", "r") as new:
            test_yaml = yaml.safe_load(test)
            new_yaml = yaml.safe_load(new)
        self.assertDictEqual(test_yaml, new_yaml)
        os.remove(f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml")

    def test_analyse(self):
        "Test analyse() function"
        defect = "vac_1_Ti_0"
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml", "w") as f:
            f.write("")
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "analyse",
                "-d",
                defect,
                "-p",
                self.EXAMPLE_RESULTS,
            ],
            catch_exceptions=False,
        )
        self.assertIn("Comparing structures to Unperturbed...", result.output)
        self.assertIn(
            f"Saved results to {self.EXAMPLE_RESULTS}/{defect}/{defect}.csv",
            result.output,
        )
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.csv") as f:
            file = f.read()
        csv_content = (
            ",Bond Distortion,Σ{Displacements} (Å),Max Distance (Å),Δ Energy (eV)\n"
            + "0,-0.4,5.315,0.88,-3.26\n"
            + "1,Unperturbed,0.0,0.0,0.0\n"
        )

        self.assertEqual(csv_content, file)
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]

        # Test --all flag
        os.mkdir(f"{self.EXAMPLE_RESULTS}/pesky_defects")
        defect_name = "vac_1_Ti_-1"
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0",
            f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}",
        )
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0",
            f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0",
        )
        result = runner.invoke(
            snb,
            [
                "analyse",
                "--all",
                "-p",
                f"{self.EXAMPLE_RESULTS}/pesky_defects",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.csv"
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0/vac_1_Ti_0.csv"
            )
        )
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/pesky_defects/")
        # Test non-existent defect
        name = "vac_1_Ti_-2"
        result = runner.invoke(
            snb,
            [
                "analyse",
                "--defect",
                name,
                "-p",
                f"{self.EXAMPLE_RESULTS}",
            ],
            catch_exceptions=True,
        )
        self.assertIn(
            f"Could not analyse defect '{name}' in directory '{self.EXAMPLE_RESULTS}'. Please "
            "either specify a defect to analyse (with option --defect), run from within a single "
            "defect directory (without setting --defect) or use the --all flag to analyse all "
            "defects in the specified/current directory.",
            str(result.exception),
        )

        # Test when `defect` is present higher up in `path`
        defect = "vac_1_Ti_0"
        os.mkdir(f"{self.EXAMPLE_RESULTS}/{defect}_defect_folder")
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/{defect}",
            f"{self.EXAMPLE_RESULTS}/{defect}_defect_folder/{defect}",
        )
        with open(
            f"{self.EXAMPLE_RESULTS}/{defect}_defect_folder/{defect}/{defect}.yaml", "w"
        ) as f:
            f.write("")
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "analyse",
                "-d",
                defect,
                "-p",
                f"{self.EXAMPLE_RESULTS}/{defect}_defect_folder/{defect}",
            ],
            catch_exceptions=False,
        )
        self.assertIn("Comparing structures to Unperturbed...", result.output)
        self.assertIn(
            f"Saved results to {self.EXAMPLE_RESULTS}/{defect}_defect_folder/{defect}/{defect}.csv",
            result.output,
        )
        with open(
            f"{self.EXAMPLE_RESULTS}/{defect}_defect_folder/{defect}/{defect}.csv"
        ) as f:
            file = f.read()
        csv_content = (
            ",Bond Distortion,Σ{Displacements} (Å),Max Distance (Å),Δ Energy (eV)\n"
            + "0,-0.4,5.315,0.88,-3.26\n"
            + "1,Unperturbed,0.0,0.0,0.0\n"
        )

        self.assertEqual(csv_content, file)
        self.tearDown()

        # Test analysing from inside the defect folder
        defect_name = "vac_1_Ti_0"
        os.chdir(self.VASP_TIO2_DATA_DIR)
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "analyse",
                ],
                catch_exceptions=True,
            )
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message) == "No output file in Bond_Distortion_10.0% "
                    "directory"
                    for warning in w
                ]
            )
        )
        self.assertIn("Comparing structures to Unperturbed...", result.output)
        self.assertIn(
            f"Saved results to {os.path.join(os.getcwd(), defect_name)}.csv",
            result.output,
        )
        self.assertTrue(os.path.exists(f"{defect_name}.csv"))
        self.assertTrue(os.path.exists(f"{defect_name}.yaml"))
        os.remove(f"{defect_name}.csv")
        os.remove(f"{defect_name}.yaml")

        # Test warning when setting path and analysing from inside the defect folder
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "analyse",
                    "-p",
                    self.EXAMPLE_RESULTS,
                ],
                catch_exceptions=True,
            )
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == "`--path` option ignored when running from within defect folder (i.e. "
                    "when `--defect` is not specified."
                    for warning in w
                ]
            )
        )
        self.assertIn("Comparing structures to Unperturbed...", result.output)
        self.assertIn(
            f"Saved results to {os.path.join(os.getcwd(), defect_name)}.csv",
            result.output,
        )
        self.assertTrue(os.path.exists(f"{defect_name}.csv"))
        self.assertTrue(os.path.exists(f"{defect_name}.yaml"))
        os.remove(f"{defect_name}.csv")
        os.remove(f"{defect_name}.yaml")
        os.chdir(file_path)

        # Test exception when run with no arguments in top-level folder
        os.chdir(self.EXAMPLE_RESULTS)
        result = runner.invoke(snb, ["analyse"], catch_exceptions=True)
        self.assertIn(
            f"Could not analyse defect 'example_results' in directory '{self.DATA_DIR}'. Please "
            "either specify a defect to analyse (with option --defect), run from within a single "
            "defect directory (without setting --defect) or use the --all flag to analyse all "
            "defects in the specified/current directory.",
            str(result.exception),
        )
        self.assertNotIn("Saved results to", result.output)
        self.assertFalse(
            any(
                os.path.exists(i)
                for i in os.listdir()
                if (i.endswith(".csv") or i.endswith(".yaml"))
            )
        )
        os.chdir(file_path)

    def test_plot(self):
        """Test plot() function"""
        # Test the following options:
        # --defect, --path, --format,  --units, --colorbar, --metric, --no_title, --verbose
        defect = "vac_1_Ti_0"
        runner = CliRunner()
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "plot",
                    "-d",
                    defect,
                    "-p",
                    self.EXAMPLE_RESULTS,
                    "--units",
                    "meV",
                    "--format",
                    "png",
                    "--colorbar",
                    "--metric",
                    "disp",
                    "-nt",  # No title
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertIn(
            f"{defect}: Energy difference between minimum, found with -0.4 bond distortion, "
            "and unperturbed: -3.26 eV.",
            result.output,
        )  # verbose output
        self.assertIn(f"Plot saved to vac_1_Ti_0/vac_1_Ti_0.png", result.output)
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json or {self.EXAMPLE_RESULTS}/"
            "vac_1_Ti_0/distortion_metadata.json not found. Will not parse "
            "its contents (to specify which neighbour atoms were distorted in plot text).",
            str(w[0].message),
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.EXAMPLE_RESULTS, defect, defect + ".png"))
        )
        # Figures are compared in the local test since on Github Actions images are saved
        # with a different size (raising error when comparing).
        self.tearDown()
        [
            os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file))
            for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect))
            if "yaml" in file or "png" in file
        ]

        # Test --all option, with the distortion_metadata.json file present to parse number of
        # distorted neighbours and their identities
        defect = "vac_1_Ti_0"
        fake_distortion_metadata = {
            "defects": {
                "vac_1_Cd": {
                    "charges": {
                        "0": {
                            "num_nearest_neighbours": 2,
                            "distorted_atoms": [[33, "Te"], [42, "Te"]],
                        },
                        "-1": {
                            "num_nearest_neighbours": 1,
                            "distorted_atoms": [
                                [33, "Te"],
                            ],
                        },
                    }
                },
                "vac_1_Ti": {
                    "charges": {
                        "0": {
                            "num_nearest_neighbours": 3,
                            "distorted_atoms": [[33, "O"], [42, "O"], [40, "O"]],
                        },
                    }
                },
            }
        }
        with open(f"{self.EXAMPLE_RESULTS}/distortion_metadata.json", "w") as f:
            f.write(json.dumps(fake_distortion_metadata, indent=4))
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "plot",
                    "--all",
                    "-p",
                    self.EXAMPLE_RESULTS,
                    "-f",
                    "png",
                ],
                catch_exceptions=False,
            )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, "vac_1_Ti_0/vac_1_Ti_0.png")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, "vac_1_Cd_0/vac_1_Cd_0.png")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, "vac_1_Cd_-1/vac_1_Cd_-1.png")
            )
        )
        if w:
            [
                self.assertNotIn(  # no distortion_metadata.json warning
                    "Will not parse its contents (to specify which neighbour atoms were "
                    "distorted in plot text).",
                    str(w[0].message),
                )
                for warning in w
            ]  # distortion_metadata file is present
        [
            os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file))
            for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect))
            if "yaml" in file
        ]
        os.remove(f"{self.EXAMPLE_RESULTS}/distortion_metadata.json")
        # Figures are compared in the local test.

        # Test plotting from inside the defect folder
        os.chdir(f"{self.EXAMPLE_RESULTS}/{defect}")  # vac_1_Ti_0
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "plot",
                ],
                catch_exceptions=False,
            )
        self.assertNotIn(
            f"{defect}: Energy difference between minimum, found with -0.4 bond distortion, "
            "and unperturbed: -3.26 eV.",
            result.output,
        )  # non-verbose output
        self.assertNotIn(
            "Plot saved to vac_1_Ti_0/vac_1_Ti_0.svg", result.output
        )  # non-verbose
        self.assertTrue(
            len([warning for warning in w if warning.category == UserWarning]) == 0
        )  # non-verbose
        self.assertFalse(
            any(["distortion_metadata.json" in str(warning.message) for warning in w])
        )  # no distortion_metadata.json warning with non-verbose option
        self.assertTrue(os.path.exists("./vac_1_Ti_0.png"))
        self.assertTrue(os.path.exists("./vac_1_Ti_0.yaml"))
        # Figures are compared in the local test since on Github Actions images are saved
        # with a different size (raising error when comparing).
        [
            os.remove(os.path.join(os.getcwd(), file))
            for file in os.listdir(os.getcwd())
            if "yaml" in file or "png" in file
        ]
        self.tearDown()

        # Test when `defect` is present higher up in `path`
        os.chdir(file_path)
        defect_name = "vac_1_Ti_0"
        os.mkdir(f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder")
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/{defect_name}",
            f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder/{defect_name}",
        )
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "plot",
                    "-p",
                    f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder/{defect_name}",
                    "-d",
                    defect_name,
                    "-cb",
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertIn(
            f"{defect}: Energy difference between minimum, found with -0.4 bond distortion, "
            "and unperturbed: -3.26 eV.",
            result.output,
        )  # verbose output
        self.assertIn(
            "Plot saved to vac_1_Ti_0/vac_1_Ti_0.svg", result.output
        )  # verbose
        self.assertTrue(
            len([warning for warning in w if warning.category == UserWarning]) == 1
        )  # verbose
        self.assertTrue(
            any(
                [  # verbose
                    f"Path {self.EXAMPLE_RESULTS}"
                    f"/{defect_name}_defect_folder/distortion_metadata.json or "
                    f"{self.EXAMPLE_RESULTS}/"
                    f"{defect_name}_defect_folder/vac_1_Ti_0/distortion_metadata.json not found. "
                    "Will not parse its contents (to specify which neighbour atoms were "
                    "distorted in plot text)." == str(warning.message)
                    for warning in w
                ]
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/{defect_name}_defect_folder"
                f"/{defect_name}/vac_1_Ti_0.svg"
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{self.EXAMPLE_RESULTS}/"
                f"{defect_name}_defect_folder/"
                f"{defect_name}/vac_1_Ti_0.yaml"
            )
        )
        self.tearDown()

        # Test warning when setting path and plotting from inside the defect folder
        os.chdir(f"{self.EXAMPLE_RESULTS}/{defect}")  # vac_1_Ti_0
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "plot",
                    "-p",
                    self.EXAMPLE_RESULTS,
                    "-v",
                ],
                catch_exceptions=True,
            )
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == "`--path` option ignored when running from within defect folder (i.e. "
                    "when `--defect` is not specified."
                    for warning in w
                ]
            )
        )
        self.assertIn(
            f"{defect}: Energy difference between minimum, found with -0.4 bond distortion, "
            "and unperturbed: -3.26 eV.",
            result.output,
        )  # non-verbose output
        self.assertIn("Plot saved to vac_1_Ti_0/vac_1_Ti_0.svg", result.output)
        self.assertTrue(
            any(
                [
                    f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json or "
                    f"{self.EXAMPLE_RESULTS}/"
                    "vac_1_Ti_0/distortion_metadata.json not found. Will not parse "
                    "its contents (to specify which neighbour atoms were distorted in plot "
                    "text)." == str(warning.message)
                    for warning in w
                ]
            )
        )
        self.assertTrue(os.path.exists("./vac_1_Ti_0.svg"))
        self.assertTrue(os.path.exists(os.getcwd() + "/vac_1_Ti_0.yaml"))
        if_present_rm(os.getcwd() + "/vac_1_Ti_0.yaml")
        self.tearDown()

        # Test exception when run with no arguments in top-level folder
        os.chdir(self.EXAMPLE_RESULTS)
        result = runner.invoke(snb, ["plot"], catch_exceptions=True)
        self.assertIn(
            f"Could not analyse & plot defect 'example_results' in directory '{self.DATA_DIR}'. "
            "Please either specify a defect to analyse (with option --defect), run from within a "
            "single defect directory (without setting --defect) or use the --all flag to analyse "
            "all defects in the specified/current directory.",
            str(result.exception),
        )
        self.assertNotIn("Plot saved to vac_1_Ti_0/vac_1_Ti_0.svg", result.output)
        self.assertFalse(
            any(os.path.exists(i) for i in os.listdir() if i.endswith(".yaml"))
        )
        self.tearDown()

        # Test --all option, with --min_energy option
        defect = "vac_1_Ti_0"
        result = runner.invoke(
            snb,
            [
                "plot",
                "--all",
                "-min",
                "1",
                "-p",
                self.EXAMPLE_RESULTS,
                "-f",
                "png",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(  # energy diff of 3.2 eV larger than min_energy
            os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, "vac_1_Ti_0/vac_1_Ti_0.png")
            )
        )
        self.assertFalse(  # energy diff of 0.75 eV less than min_energy
            os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, "vac_1_Cd_0/vac_1_Cd_0.png")
            )
        )
        self.assertFalse(  # energy diff of 0.9 eV less than min_energy
            os.path.exists(
                os.path.join(self.EXAMPLE_RESULTS, "vac_1_Cd_-1/vac_1_Cd_-1.png")
            )
        )
        [
            os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file))
            for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect))
            if "yaml" in file
        ]

    def test_regenerate(self):
        """Test regenerate() function"""
        with warnings.catch_warnings(record=True) as w:
            runner = CliRunner()
            result = runner.invoke(
                snb,
                [
                    "regenerate",
                    "-p",
                    self.EXAMPLE_RESULTS,
                    "-v",
                ],
                catch_exceptions=False,
            )
        if w:
            self.assertFalse(
                any([war.category == UserWarning for war in w])
            )  # no User Warnings

        self.assertIn(
            "Comparing structures to specified ref_structure (Cd31 Te32)...",
            result.output,
        )
        self.assertIn(
            "Comparing and pruning defect structures across charge states...",
            result.output,
        )
        self.assertIn(
            "Writing low-energy distorted structure to "
            f"{self.EXAMPLE_RESULTS}/vac_1_Cd_0/Bond_Distortion_20.0%_from_-1\n",
            result.output,
        )
        self.assertIn(
            "Writing low-energy distorted structure to "
            f"{self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_20.0%_from_-1\n",
            result.output,
        )
        self.assertIn(
            "Writing low-energy distorted structure to "
            f"{self.EXAMPLE_RESULTS}/vac_1_Cd_-1/Bond_Distortion_-60.0%_from_0\n",
            result.output,
        )
        self.assertIn(
            "Writing low-energy distorted structure to "
            f"{self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_-60.0%_from_0\n",
            result.output,
        )
        self.assertIn(
            f"No subfolders with VASP input files found in {self.EXAMPLE_RESULTS}/vac_1_Cd_-2, "
            "so just writing distorted POSCAR file to "
            f"{self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_-60.0%_from_0 directory.\n",
            result.output,
        )
        self.assertNotIn(  # now we run io.parse_energies() if energy file not present
            "No data parsed for vac_1_Ti_0. This species will be skipped and will not be included"
            " in the low_energy_defects charge state lists (and so energy lowering distortions"
            " found for other charge states will not be applied for this species).",
            result.output,
        )
        self.assertIn("Parsing vac_1_Ti_0...", result.output)
        self.assertIn(
            "vac_1_Ti_0: Energy difference between minimum, found with -0.4 bond "
            "distortion, and unperturbed: -3.26 eV.",
            result.output,
        )
        self.assertIn(
            "Energy lowering distortion found for vac_1_Ti with charge 0. Adding to "
            "low_energy_defects dictionary.",
            result.output,
        )
        self.tearDown()  # Remove generated files

        # test "*High_Energy*" ignored and doesn't cause errors
        shutil.copytree(
            os.path.join(self.EXAMPLE_RESULTS, "vac_1_Cd_0/Bond_Distortion_-60.0%"),
            os.path.join(
                self.EXAMPLE_RESULTS, "vac_1_Cd_0/Bond_Distortion_-48.0%_High_Energy"
            ),
        )
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "regenerate",
                    "-p",
                    self.EXAMPLE_RESULTS,
                    "-v",
                ],
                catch_exceptions=False,
            )
        if w:
            self.assertFalse(any([war.category == UserWarning for war in w]))
        self.assertIn(
            "Comparing structures to specified ref_structure (Cd31 Te32)...",
            result.output,
        )
        self.assertIn(
            "Comparing and pruning defect structures across charge states...",
            result.output,
        )
        self.assertIn(
            "Writing low-energy distorted structure to"
            f" {self.EXAMPLE_RESULTS}/vac_1_Cd_0/Bond_Distortion_20.0%_from_-1\n",
            result.output,
        )
        self.assertIn(
            f"No subfolders with VASP input files found in {self.EXAMPLE_RESULTS}/vac_1_Cd_-2,"
            " so just writing distorted POSCAR file to"
            f" {self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_-60.0%_from_0 directory.\n",
            result.output,
        )
        self.assertFalse("High_Energy" in result.output)

        # test FileNotFoundError raised when no defect folders found
        os.chdir(self.DATA_DIR)
        result = runner.invoke(
            snb,
            [
                "regenerate",
                "-v",
            ],
            catch_exceptions=True,
        )
        self.assertIn(
            f"No defect folders found in directory '{self.DATA_DIR}'. Please check the directory "
            "contains defect folders with names ending in a charge state after an underscore ("
            "e.g. `vac_1_Cd_0` or `Va_Cd_0` etc).",
            str(result.exception),
        )
        self.assertEqual(result.exception.__class__, FileNotFoundError)
        self.tearDown()

    def test_groundstate(self):
        """Test groundstate() function"""
        # Test default behaviour
        defect = "vac_1_Cd_0"
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "groundstate",
                "-p",
                self.VASP_CDTE_DATA_DIR,
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR")
        )
        self.assertIn(
            f"{defect}: Ground state structure (found with -0.55 distortion) saved to"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR",
            result.output,
        )
        gs_structure = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR"
        )
        self.assertEqual(gs_structure, self.V_Cd_minus0pt55_CONTCAR_struc)
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate")

        # Test keywords: groundstate_filename and directory
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "groundstate",
                "-p",
                self.VASP_CDTE_DATA_DIR,
                "-d",
                "My_Groundstate",
                "--groundstate_filename",
                "Groundstate_CONTCAR",
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(
                f"{self.VASP_CDTE_DATA_DIR}/{defect}/My_Groundstate/Groundstate_CONTCAR"
            )
        )
        self.assertIn(
            f"{defect}: Ground state structure (found with -0.55 distortion) saved to"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/My_Groundstate/Groundstate_CONTCAR",
            result.output,
        )
        gs_structure = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/{defect}/My_Groundstate/Groundstate_CONTCAR"
        )
        self.assertEqual(gs_structure, self.V_Cd_minus0pt55_CONTCAR_struc)
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/{defect}/My_Groundstate")

        # Test non existent structure file
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "groundstate",
                "-p",
                self.VASP_CDTE_DATA_DIR,
                "--structure_filename",
                "Fake_structure",
            ],
            catch_exceptions=True,
        )
        self.assertFalse(
            os.path.exists(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate")
        )
        self.assertIsInstance(result.exception, FileNotFoundError)
        self.assertIn(
            "The structure file Fake_structure is not present in the directory"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/Bond_Distortion_-55.0%",
            str(result.exception),
        )

        # test running within a single defect directory and specifying no arguments
        os.chdir(f"{self.VASP_CDTE_DATA_DIR}/{defect}")  # vac_1_Cd_0
        result = runner.invoke(snb, ["groundstate"], catch_exceptions=False)
        self.assertTrue(os.path.exists("Groundstate/POSCAR"))
        self.assertIn(
            f"{defect}: Ground state structure (found with -0.55 distortion) saved to"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR",
            result.output,
        )
        gs_structure = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR"
        )
        self.assertEqual(gs_structure, self.V_Cd_minus0pt55_CONTCAR_struc)
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate")

        # Test warning when setting path and parsing from inside the defect folder
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "groundstate",
                    "-p",
                    self.EXAMPLE_RESULTS,
                ],
                catch_exceptions=False,
            )
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == "`--path` option ignored when running from within defect folder ("
                    "determined to be the case here based on current directory and "
                    "subfolder names)."
                    for warning in w
                ]
            )
        )
        self.assertTrue(os.path.exists("Groundstate/POSCAR"))
        self.assertIn(
            f"{defect}: Ground state structure (found with -0.55 distortion) saved to"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR",
            result.output,
        )
        gs_structure = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR"
        )
        self.assertEqual(gs_structure, self.V_Cd_minus0pt55_CONTCAR_struc)
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate")

        # test error when no defect folders found
        self.tearDown()
        result = runner.invoke(
            snb,
            ["groundstate"],  # use cwd which has no defect directories
            catch_exceptions=True,
        )
        self.assertIsInstance(result.exception, FileNotFoundError)
        self.assertIn(
            "No folders with valid defect names (should end with charge e.g. 'vac_1_Cd_-2') "
            f"found in output_path: '{os.path.abspath('.')}'. Please check the path "
            "and try again.",
            str(result.exception),
        )

        # test "*High_Energy*" ignored and doesn't cause errors
        defect = "vac_1_Cd_0"
        shutil.copytree(
            os.path.join(self.EXAMPLE_RESULTS, f"{defect}/Bond_Distortion_-60.0%"),
            os.path.join(
                self.EXAMPLE_RESULTS, f"{defect}/Bond_Distortion_-48.0%_High_Energy"
            ),
        )
        result = runner.invoke(
            snb,
            [
                "groundstate",
                "-p",
                self.VASP_CDTE_DATA_DIR,
            ],
            catch_exceptions=False,
        )
        self.assertTrue(
            os.path.exists(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR")
        )
        gs_structure = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR"
        )
        self.assertEqual(gs_structure, self.V_Cd_minus0pt55_CONTCAR_struc)
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate")
        self.assertFalse("High_Energy" in result.output)


if __name__ == "__main__":
    unittest.main()
