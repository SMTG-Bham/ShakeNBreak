import unittest
import os
import datetime
import shutil
import pickle
import json
import warnings
import numpy as np

from matplotlib.testing.compare import compare_images

# Pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar

# Click
from click import exceptions
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
        with open(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.pickle"), "rb"
        ) as fp:
            self.cdte_defect_dict = pickle.load(fp)
        self.Int_Cd_2_dict = self.cdte_defect_dict["interstitials"][1]

    def tearDown(self):
        for i in [
            "parsed_defects_dict.pickle",
            "distortion_metadata.json",
            "test_config.yml",
        ]:
            if_present_rm(i)

        for i in os.listdir("."):
            if "distortion_metadata" in i:
                os.remove(i)
            elif "Vac_Cd" in i or "Int_Cd" in i or "Wally_McDoodle" in i or "pesky_defects" in i:
                shutil.rmtree(i)
        if os.path.exists(f"{os.getcwd()}/distortion_plots"):
            shutil.rmtree(f"{os.getcwd()}/distortion_plots")

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
                "-c 0",
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
                "-c 0",
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
                    "-c", "0",
                    "--defect-coords",
                    0.5, 0.5, 0.5,
                    "-v",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            warning_message = "Coordinates (0.5, 0.5, 0.5) were specified for (auto-determined) " \
                              "vacancy defect, but could not find it in bulk structure (found 0 " \
                              "possible defect sites). Will attempt auto site-matching instead."
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
                    "-c", "0",
                    "--defect-coords",
                    0.5, 0.5, 0.5,
                    "-v",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            warning_message = "Coordinates (0.5, 0.5, 0.5) were specified for (auto-determined) " \
                              "vacancy defect, but could not find it in bulk structure (found 0 " \
                              "possible defect sites). Will attempt auto site-matching instead."
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
                    "-c", "0",
                    "--defect-coords",
                    0.1, 0.1, 0.1,  # close just not quite 0,0,0
                    "-v",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            if w:
                # self.assertNotEqual(w[0].category, UserWarning)  # we have other POTCAR warnings
                # being caught, so just check no UserWarning # TODO: this runs ok locally but not on github actions?
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
                        "unique_site": [0.0, 0.0, 0.0],  # matching final site not slightly-off
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
        test_yml = f"""
                        charge: 1
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
        self.assertIn("Defect Vac_Cd_mult32 in charge state: 0", result.output)
        self.assertNotIn("Defect Vac_Cd_mult32 in charge state: +1", result.output)

        # TODO:
        # test parsed defects pickle
        # test error handling and all print messages
        # only test POSCAR as INCAR, KPOINTS and POTCAR not written on GitHub actions,
        # but tested locally -- add CLI INCAR KPOINTS and POTCAR local tests!

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
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR", f"{defects_dir}/{defect_name}/POSCAR"
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
                    "test_config.yml"
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
            str(w[0].message)
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
        for charge in range(-1,3):
            for dist in ["Unperturbed", "Bond_Distortion_30.0%"]:
                self.assertTrue(os.path.exists(f"{defect_name}_{charge}/{dist}/POSCAR"))
        for dist in ["Unperturbed", "Rattled"]:
            # -2 has 0 electron change -> only Unperturbed & rattled folders
            self.assertTrue(os.path.exists(f"{defect_name}_-2/{dist}/POSCAR"))
        # check POSCAR
        self.assertEqual(
            Structure.from_file(f"{defect_name}_0/Bond_Distortion_30.0%/POSCAR"),
            self.V_Cd_0pt3_local_rattled
        )
        if_present_rm(defects_dir)
        for charge in range(-2,3):
            if_present_rm(f"{defect_name}_{charge}")
        self.tearDown()

        # Test defects not organised in folders
        # Test defect_settings (charges, defect index/coords)
        defect_name = "Vac_Cd"
        os.mkdir(defects_dir)
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}_POSCAR"
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
                    "test_config.yml"
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
            self.V_Cd_0pt3_local_rattled
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
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR", f"{defects_dir}/{defect_name}/POSCAR"
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
                    "test_config.yml"
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
            str(w[0].message)
        )  # Defect name not parsed from config
        self.assertEqual(w[1].category, UserWarning)
        self.assertEqual(
            f"No charge (range) set for defect {defect_name} in config file,"
            " assuming default range of +/-2",
            str(w[1].message)
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
        # if_present_rm(f"{defect_name}_0")
        self.tearDown()

        # Test wrong folder defect name
        defects_dir = "pesky_defects"
        defect_name = "Wally_McDoodle"
        os.mkdir(defects_dir)
        os.mkdir(f"{defects_dir}/{defect_name}")  # non-standard defect name
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR", f"{defects_dir}/{defect_name}/POSCAR"
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
                "test_config.yml"
            ],
            catch_exceptions=True,
        )
        # Test outputs
        self.assertIsInstance(result.exception, ValueError)
        self.assertIn(
            "Error in defect name parsing; could not parse defect name",
            str(result.exception)
        )
        self.tearDown()

    def test_parse(self):
        """Test parse() function"""
        # Specifying defect to parse
        # All OUTCAR's present in distortion directories
        # Energies file already present
        defect = "vac_1_Ti_0"
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.txt", "w") as f:
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
            f"Moving old {self.EXAMPLE_RESULTS}/{defect}/{defect}.txt to ",
            result.output
        )
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.txt") as f:
            lines = [line.strip() for line in f.readlines()]
        energies = {
            "Bond_Distortion_-40.0%": "-1176.28458753",
            "Unperturbed": "-1173.02056574",
        }  # Using dictionary here (rather than file/string), because parsing order
        # is difference on github actions
        self.assertEqual(energies, {lines[0]: lines[1], lines[2]: lines[3]})
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]

        # Test when OUTCAR not present in one of the distortion directories
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
        self.assertIn(f"Bond_Distortion_10.0% not fully relaxed", result.output)
        self.assertTrue(os.path.exists(f"{self.VASP_DIR}/{defect}/{defect}.txt"))
        with open(f"{self.VASP_DIR}/{defect}/{defect}.txt") as f:
            lines = [line.strip() for line in f.readlines()]
        energies = {
            "Bond_Distortion_-40.0%": "-1176.28458753",
            "Unperturbed": "-1173.02056574",
        }
        self.assertEqual(energies, {lines[0]: lines[1], lines[2]: lines[3]})
        os.remove(f"{self.VASP_DIR}/{defect}/{defect}.txt")

        # Test --all option
        os.mkdir(f"{self.EXAMPLE_RESULTS}/pesky_defects")
        defect_name = "vac_1_Ti_-1"
        shutil.copytree(f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0", f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}")
        shutil.copytree(f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0", f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0")
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
        self.assertTrue(os.path.exists(f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.txt"))
        self.assertTrue(os.path.exists(f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0/vac_1_Ti_0.txt"))
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/pesky_defects/")

    def test_analyse(self):
        "Test analyse() function"
        defect = "vac_1_Ti_0"
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.txt", "w") as f:
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
        self.assertIn(
            f"Comparing structures to Unperturbed...",
            result.output
        )
        self.assertIn(
            f"Saved results to {self.EXAMPLE_RESULTS}/{defect}/{defect}.csv",
            result.output
        )
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.csv") as f:
            file = f.read()
        csv_content = ",Bond Distortion,Σ{Displacements} (Å),Max Distance (Å),Δ Energy (eV)\n" \
            +"0,-0.4,5.315,0.88,-3.26\n" \
            +"1,Unperturbed,0.0,0.0,0.0\n"

        self.assertEqual(csv_content, file)
        [
            os.remove(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect}")
            if os.path.isfile(f"{self.EXAMPLE_RESULTS}/{defect}/{file}")
        ]

        # Test --all flag
        os.mkdir(f"{self.EXAMPLE_RESULTS}/pesky_defects")
        defect_name = "vac_1_Ti_-1"
        shutil.copytree(f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0", f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}")
        shutil.copytree(f"{self.EXAMPLE_RESULTS}/vac_1_Ti_0", f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0")
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
        self.assertTrue(os.path.exists(f"{self.EXAMPLE_RESULTS}/pesky_defects/{defect_name}/{defect_name}.csv"))
        self.assertTrue(os.path.exists(f"{self.EXAMPLE_RESULTS}/pesky_defects/vac_1_Ti_0/vac_1_Ti_0.csv"))
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/pesky_defects/")
        # Test non-existent defect
        name =  "vac_1_Ti_-2"
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
        self.assertIsInstance(result.exception, FileNotFoundError)
        self.assertIn(
            f"Could not find {name} in the directory {self.EXAMPLE_RESULTS}.",
            str(result.exception)
        )

    def test_plot(self):
        "Test plot() function"
        # Test the following options:
        # --defect, --path, --format,  --units, --colorbar, --metric, --title, --verbose
        defect = "vac_1_Ti_0"
        wd = os.getcwd()  # plots saved to distortion_plots directory in current directory
        with open(f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.txt", "w") as f:
            f.write("")
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
                    "-t", # No title
                    "-v",
                ],
                catch_exceptions=False,
            )
        self.assertIn(
            f"{defect}: Energy difference between minimum, found with -0.4 bond distortion, and unperturbed: -3.26 eV.",
            result.output
        )  # verbose output
        self.assertIn(
            f"Plot saved to {wd}/distortion_plots/",
            result.output
        )
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(
            f"Path {self.EXAMPLE_RESULTS}/distortion_metadata.json does not exist. Will not parse its contents.",
            str(w[0].message)
        )
        self.assertTrue(os.path.exists(wd + "/distortion_plots/V$_{Ti}^{0}$.png"))
        compare_images(
            wd + "/distortion_plots/V$_{Ti}^{0}$.png",
            f"{file_path}/remote_baseline_plots/"+"V$_{Ti}^{0}$_cli_colorbar_disp.png",
            tol=2.0,
        )
        self.tearDown()
        [os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file)) for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect)) if "txt" in file]

        # Test --all option, with the distortion_metadata.json file present to parse number of
        # distorted neighbours and their identities
        defect = "vac_1_Ti_0"
        fake_distortion_metadata = {
            "defects": {
                "vac_1_Cd": {
                    "charges": {
                        "0": {
                            "num_nearest_neighbours": 2,
                            "distorted_atoms": [[33, "Te"], [42, "Te"]]
                        },
                        "-1": {
                            "num_nearest_neighbours": 1,
                            "distorted_atoms": [[33, "Te"],]
                        },
                    }
                },
                "vac_1_Ti": {
                    "charges": {
                        "0": {
                            "num_nearest_neighbours": 3,
                            "distorted_atoms": [[33, "O"], [42, "O"], [40, "O"]]
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
        self.assertTrue(os.path.exists(wd + "/distortion_plots/V$_{Ti}^{0}$.png"))
        self.assertTrue(os.path.exists(wd + "/distortion_plots/V$_{Cd}^{0}$.png"))
        self.assertTrue(os.path.exists(wd + "/distortion_plots/V$_{Cd}^{-1}$.png"))
        if w:  # distortion_metadata file present, so no warnings
            self.assertNotEqual(w[0].category, UserWarning)
        [os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file)) for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect)) if "txt" in file]
        os.remove(f"{self.EXAMPLE_RESULTS}/distortion_metadata.json")
        # Compare figures
        compare_images(
            wd + "/distortion_plots/V$_{Cd}^{0}$.png",
            f"{file_path}/remote_baseline_plots/"+"V$_{Cd}^{0}$_cli_default.png",
            tol=2.0,
        )
        self.tearDown()

if __name__ == "__main__":
    unittest.main()
