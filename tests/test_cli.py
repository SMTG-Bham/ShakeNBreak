import unittest
import os
import datetime
import shutil
import pickle
import json
import warnings
from monty.serialization import loadfn
import numpy as np
import filecmp
import subprocess

# Pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar, Incar

# Click
from click.testing import CliRunner

from shakenbreak.cli import snb

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
        self.V_Cd_minus0pt5_struc_local_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_V_Cd_minus0pt5_struc_local_rattled_POSCAR",
            )
        )  # Local rattle is default
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
        with open(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.pickle"), "rb"
        ) as fp:
            self.cdte_defect_dict = pickle.load(fp)
        self.Int_Cd_2_dict = self.cdte_defect_dict["interstitials"][1]

    def tearDown(self):
        os.chdir(os.path.dirname(__file__))
        for i in [
            "parsed_defects_dict.pickle",
            "distortion_metadata.json",
            "test_config.yml",
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
        if os.path.exists(f"{os.getcwd()}/distortion_plots"):
            shutil.rmtree(f"{os.getcwd()}/distortion_plots")

        for defect in os.listdir(self.EXAMPLE_RESULTS):
            if os.path.isdir(f"{self.EXAMPLE_RESULTS}/{defect}"):
                [
                    shutil.rmtree(f"{self.EXAMPLE_RESULTS}/{defect}/{dir}")
                    for dir in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
                    if "_from_" in dir
                ]
            elif os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}"):
                os.remove(f"{self.EXAMPLE_RESULTS}/{defect}")

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
                "-v",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            f"Auto site-matching identified {self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR "
            f"to be type Vacancy with site [0. 0. 0.] Cd",
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
            result.output,
        )
        self.assertIn("--Distortion -40.0%", result.output)
        self.assertIn(
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
            "-50.0%__num_neighbours=2_Vac_Cd_mult32",
        )  # default
        self.assertEqual(
            V_Cd_minus0pt5_rattled_POSCAR.structure,
            self.V_Cd_minus0pt5_struc_local_rattled,
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
            f"Auto site-matching identified"
            f" {self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR "
            f"to be type Vacancy with site [0. 0. 0.] Cd",
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
            "'0.3', '0.4', '0.5', '0.6']. Then, will rattle with a std dev of 0.25 Å",
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
                "--defect-index",
                "4",
                "-v",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(f"Auto site-matching", result.output)
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
                "rattle_stdev": 0.25,
                "local_rattle": True,
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
                                "rattle_stdev": 0.25,
                            },
                        },
                    },
                }
            },
        }
        # check defects from old metadata file are in new metadata file
        with open(f"distortion_metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
        np.testing.assert_equal(metadata, wrong_site_V_Cd_dict)

        # test warning with defect_coords option but wrong site:
        self.tearDown()
        runner = CliRunner()
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
                    0.5,
                    0.5,
                    0.5,
                    "-v",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            warning_message = (
                "Coordinates (0.5, 0.5, 0.5) were specified for (auto-determined) "
                "vacancy defect, but could not find it in bulk structure (found 0 "
                "possible defect sites). Will attempt auto site-matching instead."
            )
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertIn("--Distortion -60.0%", result.output)
            self.assertIn(
                f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
                + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
                + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
                result.output,
            )

        self.tearDown()
        runner = CliRunner()
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
                    0.5,
                    0.5,
                    0.5,
                    "-v",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            warning_message = (
                "Coordinates (0.5, 0.5, 0.5) were specified for (auto-determined) "
                "vacancy defect, but could not find it in bulk structure (found 0 "
                "possible defect sites). Will attempt auto site-matching instead."
            )
            self.assertEqual(w[0].category, UserWarning)
            self.assertIn(warning_message, str(w[0].message))
            self.assertIn("--Distortion -60.0%", result.output)
            self.assertIn(
                f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
                + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
                + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
                result.output,
            )

        # test defect_coords working:
        self.tearDown()
        runner = CliRunner()
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
                    0.1,
                    0.1,
                    0.1,  # close just not quite 0,0,0
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
                f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
                + "            Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
                + "            Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
                result.output,
            )
            self.assertNotIn(f"Auto site-matching", result.output)
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
                    "rattle_stdev": 0.25,
                    "local_rattle": True,
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
                                    "rattle_stdev": 0.25,
                                },
                            },
                        },
                    }
                },
            }
            # check defects from old metadata file are in new metadata file
            with open(f"distortion_metadata.json", "r") as metadata_file:
                metadata = json.load(metadata_file)
            np.testing.assert_equal(metadata, spec_coords_V_Cd_dict)

    def test_snb_generate_config(self):
        # test config file:
        test_yml = f"""
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
                f"test_config.yml",
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

        test_yml = f"""
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
                f"test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(f"Auto site-matching identified", result.output)
        self.assertNotIn("Oxidation states were not explicitly set", result.output)
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ["
            "'-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', "
            "'0.3', '0.4', '0.5', '0.6']. Then, will rattle with a std dev of 0.25 Å",
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
        test_yml = f"""
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
                "-c" "1",
                "--config",
                f"test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(f"Auto site-matching identified", result.output)
        self.assertIn("Oxidation states were not explicitly set", result.output)
        self.assertIn(
            "Applying ShakeNBreak... Will apply the following bond distortions: ['-0.5', "
            "'-0.25', '0.0', '0.25', '0.5']. Then, will rattle with a std dev of 0.25 Å",
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
                "rattle_stdev": 0.25,
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
                                "rattle_stdev": 0.25,
                            },
                        },
                    },
                    "defect_site_index": 65,
                }
            },
        }
        # check defects from old metadata file are in new metadata file
        with open(f"distortion_metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
        np.testing.assert_equal(metadata, kwarged_Int_Cd_2_dict)

        self.tearDown()
        test_yml = f"""
        bond_distortions: [-0.5, -0.25, 0.0, 0.25, 0.5]
        name: Wally_McDoodle
        local_rattle: True
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
                "-c" "0",
                "--config",
                f"test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn(f"Auto site-matching identified", result.output)
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
            "-50.0%__num_neighbours=2_Wally_McDoodle",
        )  # default
        self.assertEqual(
            V_Cd_minus0pt5_rattled_POSCAR.structure,
            self.V_Cd_minus0pt5_struc_local_rattled,
        )

        # test min, max charge setting:
        self.tearDown()
        test_yml = f"""
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
                f"test_config.yml",
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
        test_yml = f"""charge: 1"""
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
                "-c" "0",
                "--name",
                "vac_1_Cd",
                "--config",
                f"test_config.yml",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Defect vac_1_Cd in charge state: 0", result.output)
        self.assertNotIn("Defect vac_1_Cd in charge state: +1", result.output)
        # test parsed defects pickle
        with open("./parsed_defects_dict.pickle", "rb") as fp:
            parsed_defects_dict = pickle.load(fp)
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
        # os.remove("test_config.yml")
        test_yml = f"""charges: [0,]
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
                "vac_1_Cd",  # to match saved pickle
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Defect vac_1_Cd in charge state: 0", result.output)
        self.tearDown()

    def test_snb_generate_all(self):
        """Test generate_all() function."""
        # Test parsing defects from folders with non-standard names
        # And default charge states
        # Create a folder for defect files / directories
        defects_dir = f"pesky_defects"
        defect_name = "vac_1_Cd"
        os.mkdir(defects_dir)
        os.mkdir(f"{defects_dir}/{defect_name}")  # non-standard defect name
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}/POSCAR",
        )
        # CONFIG file
        test_yml = """bond_distortions: [0.3,]"""
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
        self.assertIn(f"Auto site-matching identified", result.output)
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
        self.assertIn(f"Auto site-matching identified", result.output)
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
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
        self.assertIn(f"Auto site-matching identified", result.output)
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
        # Only neutral charge state
        self.assertIn(
            f"Defect {defect_name} in charge state: 0. Number of distorted neighbours: 2",
            result.output,
        )
        self.assertIn("--Distortion 30.0%", result.output)
        self.assertIn(
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
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
            "Job file 'job' not in current directory, so will only submit jobs in folders with "
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
            "Job file 'job_file' not in current directory, so will only submit jobs in folders with "
            "'job_file' present",
            out,
        )
        self.assertIn("Bond_Distortion_-40.0% fully relaxed", out)
        self.assertIn("Unperturbed fully relaxed", out)
        self.assertNotIn("Bond_Distortion_10.0% fully relaxed", out)  # also present
        self.assertIn("Running job for Bond_Distortion_10.0%", out)
        self.assertIn("this vac_1_Ti_0_10.0% job_file", out)  # job submit command
        self.assertTrue(os.path.exists("Bond_Distortion_10.0%/job_file"))

        if_present_rm("job_file")
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
        print(out)
        self.assertIn(
            "No defect folders (with names ending in a number (charge state)) found in "
            "current directory",
            out,
        )

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
            f"No output file in Bond_Distortion_10.0% directory",
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

        # Test warning when run with no arguments in top-level folder
        os.chdir(self.EXAMPLE_RESULTS)
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(snb, ["parse"], catch_exceptions=True)
        self.assertTrue(any([warning.category == UserWarning for warning in w]))
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == f"Energies could not be parsed for defect 'example_results' in"
                    f" {self.DATA_DIR}. If these directories are correct, "
                    f"check calculations have converged, and that distortion subfolders match "
                    f"ShakeNBreak naming (e.g. Bond_Distortion_xxx, Rattled, Unperturbed)"
                    for warning in w
                ]
            )
        )
        self.assertFalse(
            any(os.path.exists(i) for i in os.listdir() if i.endswith(".yaml"))
        )
        os.chdir(file_path)

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
        self.assertTrue(
            filecmp.cmp(
                f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml",
                f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml",
            )
        )
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
        self.assertTrue(
            filecmp.cmp(
                f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml",
                f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml",
            )
        )
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
        self.assertTrue(
            filecmp.cmp(
                f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml",
                f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml",
            )
        )
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
        self.assertTrue(
            filecmp.cmp(
                f"{self.DATA_DIR}/{code}/{defect}/test_{defect}.yaml",
                f"{self.DATA_DIR}/{code}/{defect}/{defect}.yaml",
            )
        )
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
        self.assertIn(f"Comparing structures to Unperturbed...", result.output)
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
            f"either specify a defect to analyse (with option --defect), run from within a single "
            f"defect directory (without setting --defect) or use the --all flag to analyse all "
            f"defects in the specified/current directory.",
            str(result.exception),
        )

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
            f"either specify a defect to analyse (with option --defect), run from within a single "
            f"defect directory (without setting --defect) or use the --all flag to analyse all "
            f"defects in the specified/current directory.",
            str(result.exception),
        )
        self.assertNotIn(f"Saved results to", result.output)
        self.assertFalse(
            any(
                os.path.exists(i)
                for i in os.listdir()
                if (i.endswith(".csv") or i.endswith(".yaml"))
            )
        )
        os.chdir(file_path)

    def test_plot(self):
        "Test plot() function"
        # Test the following options:
        # --defect, --path, --format,  --units, --colorbar, --metric, --no_title, --verbose
        defect = "vac_1_Ti_0"
        wd = (
            os.getcwd()
        )  # plots saved to distortion_plots directory in current directory
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
            f"and unperturbed: -3.26 eV.",
            result.output,
        )  # verbose output
        self.assertIn(f"Plot saved to {wd}/distortion_plots/", result.output)
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json does not exist. "
            f"Will not parse its contents (to specify which neighbour atoms were distorted in "
            f"plot text).",
            str(w[0].message),
        )
        self.assertTrue(os.path.exists(wd + "/distortion_plots/vac_1_Ti_0.png"))
        # Figures are compared in the local test since on Github Actions images are saved
        # with a different size (raising error when comparing).
        self.tearDown()
        [
            os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file))
            for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect))
            if "yaml" in file
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
        self.assertTrue(os.path.exists(wd + "/distortion_plots/vac_1_Ti_0.png"))
        self.assertTrue(os.path.exists(wd + "/distortion_plots/vac_1_Cd_0.png"))
        self.assertTrue(os.path.exists(wd + "/distortion_plots/vac_1_Cd_-1.png"))
        if w:
            [
                self.assertNotEqual(
                    f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json does not exist. Will "
                    f"not parse its contents (to specify which neighbour atoms were distorted in "
                    f"plot text).",
                    str(warning.message),
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
            f"and unperturbed: -3.26 eV.",
            result.output,
        )  # non-verbose output
        self.assertIn(f"Plot saved to {os.getcwd()}/distortion_plots/", result.output)
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json does not exist. Will not parse "
            f"its contents (to specify which neighbour atoms were distorted in plot text).",
            str(w[0].message),
        )
        self.assertTrue(
            os.path.exists(os.getcwd() + "/distortion_plots/vac_1_Ti_0.svg")
        )
        self.assertTrue(os.path.exists(os.getcwd() + "/vac_1_Ti_0.yaml"))
        # Figures are compared in the local test since on Github Actions images are saved
        # with a different size (raising error when comparing).
        shutil.rmtree(os.getcwd() + "/distortion_plots")
        [
            os.remove(os.path.join(os.getcwd(), file))
            for file in os.listdir(os.getcwd())
            if "yaml" in file
        ]
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
            f"and unperturbed: -3.26 eV.",
            result.output,
        )  # non-verbose output
        self.assertIn(f"Plot saved to {os.getcwd()}/distortion_plots/", result.output)
        self.assertTrue(
            any(
                [
                    str(warning.message)
                    == f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json does not exist. Will "
                    f"not parse its contents (to specify which neighbour atoms were distorted in "
                    f"plot text).",
                ]
                for warning in w
            )
        )
        self.assertTrue(
            os.path.exists(os.getcwd() + "/distortion_plots/vac_1_Ti_0.svg")
        )
        self.assertTrue(os.path.exists(os.getcwd() + "/vac_1_Ti_0.yaml"))
        shutil.rmtree(os.getcwd() + "/distortion_plots")
        [
            os.remove(os.path.join(os.getcwd(), file))
            for file in os.listdir(os.getcwd())
            if "yaml" in file
        ]
        self.tearDown()

        # Test exception when run with no arguments in top-level folder
        os.chdir(self.EXAMPLE_RESULTS)
        result = runner.invoke(snb, ["plot"], catch_exceptions=True)
        self.assertIn(
            f"Could not analyse & plot defect 'example_results' in directory '{self.DATA_DIR}'. "
            f"Please either specify a defect to analyse (with option --defect), run from within a "
            f"single defect directory (without setting --defect) or use the --all flag to analyse all "
            f"defects in the specified/current directory.",
            str(result.exception),
        )
        self.assertNotIn(
            f"Plot saved to {os.getcwd()}/distortion_plots/", result.output
        )
        self.assertFalse(
            any(os.path.exists(i) for i in os.listdir() if i.endswith(".yaml"))
        )
        self.tearDown()

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
            self.assertTrue(any([war.category == UserWarning for war in w]))
            self.assertTrue(
                any(
                    [
                        f"Path {self.EXAMPLE_RESULTS}/vac_1_Ti_0/vac_1_Ti_0.yaml does not exist"
                        == str(war.message)
                        for war in w
                    ]
                )
            )
        # self.assertIn(
        #     f"No data parsed for vac_1_Ti_0. This species will be skipped and will not be included"
        #     " in the low_energy_defects charge state lists (and so energy lowering distortions"
        #     " found for other charge states will not be applied for this species).",
        #     result.output,
        # )
        self.assertIn(
            "Comparing structures to specified ref_structure (Cd31 Te32)...",
            result.output,
        )
        self.assertIn(
            "Comparing and pruning defect structures across charge states...",
            result.output,
        )
        self.assertIn(
            f"Writing low-energy distorted structure to {self.EXAMPLE_RESULTS}/vac_1_Cd_0/Bond_Distortion_20.0%_from_-1\n",
            result.output,
        )
        self.assertIn(
            f"Writing low-energy distorted structure to {self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_20.0%_from_-1\n",
            result.output,
        )
        self.assertIn(
            f"Writing low-energy distorted structure to {self.EXAMPLE_RESULTS}/vac_1_Cd_-1/Bond_Distortion_-60.0%_from_0\n",
            result.output,
        )
        self.assertIn(
            f"Writing low-energy distorted structure to {self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_-60.0%_from_0\n",
            result.output,
        )
        self.assertIn(
            f"No subfolders with VASP input files found in {self.EXAMPLE_RESULTS}/vac_1_Cd_-2,"
            f" so just writing distorted POSCAR file to {self.EXAMPLE_RESULTS}/vac_1_Cd_-2/Bond_Distortion_-60.0%_from_0 directory.\n",
            result.output,
        )
        # Remove generated files
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
            f"{defect}: Gound state structure (found with -0.55 distortion) saved to"
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
            f"{defect}: Gound state structure (found with -0.55 distortion) saved to"
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
            f"The structure file Fake_structure is not present in the directory"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/Bond_Distortion_-55.0%",
            str(result.exception),
        )

        # test running within a single defect directory and specifying no arguments
        os.chdir(f"{self.VASP_CDTE_DATA_DIR}/{defect}")  # vac_1_Cd_0
        result = runner.invoke(snb, ["groundstate"], catch_exceptions=False)
        self.assertTrue(os.path.exists("Groundstate/POSCAR"))
        self.assertIn(
            f"{defect}: Gound state structure (found with -0.55 distortion) saved to"
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
            f"{defect}: Gound state structure (found with -0.55 distortion) saved to"
            f" {self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR",
            result.output,
        )
        gs_structure = Structure.from_file(
            f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate/POSCAR"
        )
        self.assertEqual(gs_structure, self.V_Cd_minus0pt55_CONTCAR_struc)
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/{defect}/Groundstate")


if __name__ == "__main__":
    unittest.main()
