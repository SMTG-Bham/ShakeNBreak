"""
Python test file only to be run locally, when POTCARs are available and the .pmgrc.yaml file is
set up. This cannot be run on GitHub actions as it does not have the POTCARs, preventing POTCAR
and INCAR files from being written.
"""

import copy
import json
import os
import shutil
import unittest
import warnings
from unittest.mock import patch

import numpy as np

# Click
from click.testing import CliRunner
from doped import vasp_input
from matplotlib.testing.compare import compare_images
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects.core import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, UnknownPotcarWarning

from shakenbreak import cli, input, vasp
from shakenbreak.cli import snb

_file_path = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_file_path, "data")


def if_present_rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _update_struct_defect_dict(
    defect_dict: dict, structure: Structure, poscar_comment: str
) -> dict:
    """
    Given a Structure object and POSCAR comment, update the folders dictionary
    (generated with `doped.vasp_input.prepare_vasp_defect_inputs()`) with
    the given values.
    Args:
        defect_dict (:obj:`dict`):
            Dictionary with defect information, as generated with doped
            `prepare_vasp_defect_inputs()`
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


class DistortionLocalTestCase(unittest.TestCase):
    """Test ShakeNBreak structure distortion helper functions"""

    def setUp(self):
        warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.EXAMPLE_RESULTS = os.path.join(self.DATA_DIR, "example_results")

        # Refactor doped defect dict to dict of Defect() objects
        self.cdte_doped_defect_dict = loadfn(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.json")
        )
        self.cdte_defects = {
            defect_dict["name"]: input.generate_defect_object(
                single_defect_dict=defect_dict,
                bulk_dict=self.cdte_doped_defect_dict["bulk"],
            )
            for defects_type, defect_dict_list in self.cdte_doped_defect_dict.items()
            if "bulk" not in defects_type
            for defect_dict in defect_dict_list
        }  # with doped/PyCDT names

        self.V_Cd_dict = self.cdte_doped_defect_dict["vacancies"][0]
        self.Int_Cd_2_dict = self.cdte_doped_defect_dict["interstitials"][1]
        # Refactor to Defect() objects
        self.V_Cd = input.generate_defect_object(
            self.V_Cd_dict, self.cdte_doped_defect_dict["bulk"]
        )
        self.Int_Cd_2 = input.generate_defect_object(
            self.Int_Cd_2_dict, self.cdte_doped_defect_dict["bulk"]
        )

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
        self.Int_Cd_2_minus0pt6_NN_10_struc_rattled = Structure.from_file(
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

        self.parsed_default_incar_settings = {
            k: v for k, v in vasp.default_incar_settings.items() if "#" not in k
        }  # pymatgen doesn't parsed commented lines
        self.parsed_incar_settings_wo_comments = {
            k: v
            for k, v in self.parsed_default_incar_settings.items()
            if "#" not in str(v)
        }  # pymatgen ignores comments after values

    def tearDown(self) -> None:
        for i in self.cdte_defect_folders:
            if_present_rm(i)  # remove test-generated vac_1_Cd_0 folder if present
        if os.path.exists("distortion_metadata.json"):
            os.remove("distortion_metadata.json")

        for i in [
            "parsed_defects_dict.json",
            "distortion_metadata.json",
            "test_config.yml",
        ]:
            if_present_rm(i)

        for i in os.listdir("."):
            if "distortion_metadata" in i:
                os.remove(i)
            if ".png" in i:
                os.remove(i)
            elif (
                "Vac_Cd" in i
                or "v_Cd" in i
                or "vac_1_Cd" in i
                or "Int_Cd" in i
                or "Wally_McDoodle" in i
                or "pesky_defects" in i
            ):
                shutil.rmtree(i)

        for defect_folder in [
            dir for dir in os.listdir(self.EXAMPLE_RESULTS)
            if os.path.isdir(f"{self.EXAMPLE_RESULTS}/{dir}")
        ]:
            for file in os.listdir(f"{self.EXAMPLE_RESULTS}/{defect_folder}"):
                if file.endswith(".png"):
                    os.remove(f"{self.EXAMPLE_RESULTS}/{defect_folder}/{file}")

    # test create_folder and create_vasp_input simultaneously:
    def test_create_vasp_input(self):
        """Test create_vasp_input function for INCARs and POTCARs"""
        vasp_defect_inputs = vasp_input.prepare_vasp_defect_inputs(
            copy.deepcopy(self.cdte_doped_defect_dict)
        )
        V_Cd_updated_charged_defect_dict = _update_struct_defect_dict(
            vasp_defect_inputs["vac_1_Cd_0"],
            self.V_Cd_minus0pt5_struc_rattled,
            "V_Cd Rattled",
        )
        # make unperturbed defect entry:
        V_Cd_unperturbed_dict = _update_struct_defect_dict(
            vasp_defect_inputs["vac_1_Cd_0"],
            self.V_Cd_struc,
            "V_Cd Unperturbed",
        )
        V_Cd_charged_defect_dict = {
            "Unperturbed": V_Cd_unperturbed_dict,
            "Bond_Distortion_-50.0%": V_Cd_updated_charged_defect_dict
        }
        self.assertFalse(os.path.exists("vac_1_Cd_0"))
        input._create_vasp_input(
            "vac_1_Cd_0",
            distorted_defect_dict=V_Cd_charged_defect_dict,
            incar_settings=vasp.default_incar_settings,
        )
        V_Cd_minus50_folder = "vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_minus50_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_minus50_folder + "/POSCAR")
        self.assertEqual(V_Cd_POSCAR.comment, "V_Cd Rattled")
        self.assertEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)

        V_Cd_INCAR = Incar.from_file(V_Cd_minus50_folder + "/INCAR")
        # check if default INCAR is subset of INCAR:
        self.assertTrue(
            self.parsed_incar_settings_wo_comments.items() <= V_Cd_INCAR.items()
        )

        V_Cd_KPOINTS = Kpoints.from_file(V_Cd_minus50_folder + "/KPOINTS")
        self.assertEqual(V_Cd_KPOINTS.kpts, [[1, 1, 1]])

        # check if POTCARs have been written:
        self.assertTrue(os.path.isfile(V_Cd_minus50_folder + "/POTCAR"))

        # test with kwargs: (except POTCAR settings because we can't have this on the GitHub test
        # server)
        kwarg_incar_settings = {
            "NELECT": 3,
            "IBRION": 42,
            "LVHAR": True,
            "LWAVE": True,
            "LCHARG": True,
            "ENCUT": 200,
        }
        kwarged_incar_settings = self.parsed_incar_settings_wo_comments.copy()
        kwarged_incar_settings.update(kwarg_incar_settings)
        input._create_vasp_input(
            "vac_1_Cd_0",
            distorted_defect_dict=V_Cd_charged_defect_dict,
            incar_settings=kwarged_incar_settings,
        )
        V_Cd_kwarg_minus50_folder = "vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_kwarg_minus50_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_kwarg_minus50_folder + "/POSCAR")
        self.assertEqual(V_Cd_POSCAR.comment, "V_Cd Rattled")
        self.assertEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)

        V_Cd_INCAR = Incar.from_file(V_Cd_kwarg_minus50_folder + "/INCAR")
        # check if default INCAR is subset of INCAR:
        self.assertFalse(
            self.parsed_incar_settings_wo_comments.items() <= V_Cd_INCAR.items()
        )
        self.assertTrue(kwarged_incar_settings.items() <= V_Cd_INCAR.items())

        V_Cd_KPOINTS = Kpoints.from_file(V_Cd_kwarg_minus50_folder + "/KPOINTS")
        self.assertEqual(V_Cd_KPOINTS.kpts, [[1, 1, 1]])

        # check if POTCARs have been written:
        self.assertTrue(os.path.isfile(V_Cd_kwarg_minus50_folder + "/POTCAR"))

    @patch("builtins.print")
    def test_write_vasp_files(self, mock_print):
        """Test write_vasp_files method"""
        oxidation_states = {"Cd": +2, "Te": -2}
        bond_distortions = list(np.arange(-0.6, 0.601, 0.05))

        dist = input.Distortions(
            self.cdte_defects,
            oxidation_states=oxidation_states,
            bond_distortions=bond_distortions,
            local_rattle=False,
        )
        distorted_defect_dict, _ = dist.write_vasp_files(
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
            "Then, will rattle with a std dev of 0.28 Å \n",
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
        V_Cd_minus50_folder = "vac_1_Cd_0/Bond_Distortion_-50.0%"
        self.assertTrue(os.path.exists(V_Cd_minus50_folder))
        V_Cd_POSCAR = Poscar.from_file(V_Cd_minus50_folder + "/POSCAR")
        self.assertEqual(
            V_Cd_POSCAR.comment,
            "-50.0%__num_neighbours=2__vac_1_Cd",
        )  # default
        V_Cd_POSCAR.structure.remove_oxidation_states()
        self.assertNotEqual(V_Cd_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled)
        # V_Cd_minus0pt5_struc_rattled was with old default seed = 42 and stdev = 0.25

        # Check INCAR
        V_Cd_INCAR = Incar.from_file(V_Cd_minus50_folder + "/INCAR")
        # check if default INCAR is subset of INCAR: (not here because we set ENCUT)
        self.assertFalse(
            self.parsed_incar_settings_wo_comments.items() <= V_Cd_INCAR.items()
        )
        self.assertEqual(V_Cd_INCAR.pop("ENCUT"), 212)
        self.assertEqual(V_Cd_INCAR.pop("IBRION"), 0)
        self.assertEqual(V_Cd_INCAR.pop("EDIFF"), 1e-4)
        self.assertEqual(V_Cd_INCAR.pop("ROPT"), "1e-3 1e-3")
        parsed_settings = self.parsed_incar_settings_wo_comments.copy()
        parsed_settings.pop("ENCUT")
        self.assertTrue(
            parsed_settings.items()
            <= V_Cd_INCAR.items()  # matches after
            # removing kwarg settings
        )
        # Check KPOINTS
        V_Cd_KPOINTS = Kpoints.from_file(V_Cd_minus50_folder + "/KPOINTS")
        self.assertEqual(V_Cd_KPOINTS.kpts, [[1, 1, 1]])

        # check if POTCARs have been written:
        self.assertTrue(os.path.isfile(V_Cd_minus50_folder + "/POTCAR"))

        # Check POSCARs
        Int_Cd_2_minus60_folder = "Int_Cd_2_0/Bond_Distortion_-60.0%"
        self.assertTrue(os.path.exists(Int_Cd_2_minus60_folder))
        Int_Cd_2_POSCAR = Poscar.from_file(Int_Cd_2_minus60_folder + "/POSCAR")
        self.assertEqual(
            Int_Cd_2_POSCAR.comment,
            "-60.0%__num_neighbours=2__Int_Cd_2",
        )
        struc = Int_Cd_2_POSCAR.structure
        struc.remove_oxidation_states()
        self.assertEqual(struc, self.Int_Cd_2_minus0pt6_struc_rattled)

        # check INCAR
        V_Cd_INCAR = Incar.from_file(V_Cd_minus50_folder + "/INCAR")
        Int_Cd_2_INCAR = Incar.from_file(Int_Cd_2_minus60_folder + "/INCAR")
        # neutral even-electron INCARs the same except for NELECT:
        for incar in [V_Cd_INCAR, Int_Cd_2_INCAR]:
            incar.pop("NELECT")  # https://tenor.com/bgVv9.gif
        self.assertEqual(V_Cd_INCAR, Int_Cd_2_INCAR)
        # Kpoints
        Int_Cd_2_KPOINTS = Kpoints.from_file(Int_Cd_2_minus60_folder + "/KPOINTS")
        self.assertEqual(Int_Cd_2_KPOINTS.kpts, [[1, 1, 1]])
        # check if POTCARs have been written:
        self.assertTrue(os.path.isfile(Int_Cd_2_minus60_folder + "/POTCAR"))

    def test_plot(self):
        """
        Test plot() function.
        The plots used for comparison have been generated with the Montserrat font
        (available in the fonts directory).
        """
        # Test the following options:
        # --defect, --path, --format,  --units, --colorbar, --metric, --no_title, --verbose
        defect = "v_Ti_0"
        dumpfn(
            {
                "distortions": {-0.4: -1176.28458753},
                "Unperturbed": -1173.02056574,
            },
            f"{self.EXAMPLE_RESULTS}/{defect}/{defect}.yaml",
        )
        if os.path.exists(f"{self.EXAMPLE_RESULTS}/distortion_metadata.json"):
            os.remove(f"{self.EXAMPLE_RESULTS}/distortion_metadata.json")
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
        self.assertTrue(
            os.path.exists(os.path.join(self.EXAMPLE_RESULTS, f"{defect}/{defect}.png"))
        )
        compare_images(
            os.path.join(self.EXAMPLE_RESULTS, f"{defect}/{defect}.png"),
            f"{_DATA_DIR}/local_baseline_plots/vac_1_Ti_0_cli_colorbar_disp.png",
            tol=2.0,
        )  # only locally (on Github Actions, saved image has a different size)
        self.tearDown()
        [
            os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file))
            for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect))
            if "yaml" in file or "png" in file
        ]

        # Test --all option, with the distortion_metadata.json file present to parse number of
        # distorted neighbours and their identities
        fake_distortion_metadata = {
            "defects": {
                "v_Cd_s0": {
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
                "v_Ti": {
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
            os.path.exists(os.path.join(self.EXAMPLE_RESULTS, f"{defect}/{defect}.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.EXAMPLE_RESULTS, "v_Cd_s0_0/v_Cd_s0_0.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.EXAMPLE_RESULTS, "v_Cd_s0_-1/v_Cd_s0_-1.png"))
        )
        compare_images(
            os.path.join(self.EXAMPLE_RESULTS, "v_Cd_s0_0/v_Cd_s0_0.png"),
            f"{_DATA_DIR}/local_baseline_plots/vac_1_Cd_0_cli_default.png",
            tol=2.0,
        )  # only locally (on Github Actions, saved image has a different size)
        [
            os.remove(os.path.join(self.EXAMPLE_RESULTS, defect, file))
            for file in os.listdir(os.path.join(self.EXAMPLE_RESULTS, defect))
            if "yaml" in file or "png" in file
        ]

        # generate docs example plots:
        shutil.copytree(
            f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0", f"{self.EXAMPLE_RESULTS}/orig_v_Cd_s0_0"
        )
        for i in range(1,7):
            shutil.copyfile(
                f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/Unperturbed/CONTCAR",
                f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/Bond_Distortion_{i}0.0%/CONTCAR",
            )
        energies_dict = loadfn(f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/v_Cd_s0_0.yaml")
        energies_dict["distortions"][-0.5] = energies_dict["distortions"][-0.6]
        dumpfn(energies_dict, f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/v_Cd_s0_0.yaml")
        shutil.copyfile(
            f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/Bond_Distortion_-60.0%/CONTCAR",
            f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/Bond_Distortion_-50.0%/CONTCAR",
        )

        result = runner.invoke(
            snb,
            [
                "plot",
                "-d",
                "v_Cd_s0_0",
                "-cb",
                "-p",
                self.EXAMPLE_RESULTS,
                "-f",
                "svg",
            ],
        )
        shutil.copyfile(f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/v_Cd_s0_0.svg",
                        "../docs/v_Cd_s0_0_colorbar.svg")
        result = runner.invoke(
            snb,
            [
                "plot",
                "-d",
                "v_Cd_s0_0",
                "-p",
                self.EXAMPLE_RESULTS,
                "-f",
                "svg",
            ],
        )
        shutil.copyfile(f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0/v_Cd_s0_0.svg", "../docs/v_Cd_s0_0.svg")
        shutil.rmtree(f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0")
        shutil.move(
            f"{self.EXAMPLE_RESULTS}/orig_v_Cd_s0_0", f"{self.EXAMPLE_RESULTS}/v_Cd_s0_0"
        )
        os.remove(f"{self.EXAMPLE_RESULTS}/distortion_metadata.json")
        self.tearDown()

    def test_generate_all_input_file(self):
        """Test generate_all() function when user specifies input_file"""
        defects_dir = f"pesky_defects"
        defect_name = "vac_1_Cd"
        os.mkdir(defects_dir)
        os.mkdir(f"{defects_dir}/{defect_name}")  # non-standard defect name
        shutil.copyfile(
            f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
            f"{defects_dir}/{defect_name}/POSCAR",
        )
        test_yml = f"""
        defects:
            {defect_name}:
                charges: [0,]
                defect_coords: [0.0, 0.0, 0.0]
        bond_distortions: [0.3,]
        POTCAR:
          Cd: Cd_GW
        """
        with open("test_config.yml", "w") as fp:
            fp.write(test_yml)

        # Test VASP
        with open("INCAR", "w") as fp:
            fp.write("IBRION = 1 \n GGA = PS")
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate_all",
                "-d",
                f"{defects_dir}/",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "--code",
                "vasp",
                "--input_file",
                "INCAR",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=True,
        )
        dist = "Unperturbed"
        incar_dict = Incar.from_file(f"{defect_name}_0/{dist}/INCAR").as_dict()
        self.assertEqual(incar_dict["GGA"].lower(), "PS".lower())
        self.assertEqual(incar_dict["IBRION"], 1)
        for file in ["KPOINTS", "POTCAR", "POSCAR"]:
            self.assertTrue(os.path.exists(f"{defect_name}_0/{dist}/{file}"))
        # Check POTCAR generation
        with open(f"{defect_name}_0/{dist}/POTCAR") as myfile:
            first_line = myfile.readline()
        self.assertIn("PAW_PBE Cd_GW", first_line)

        shutil.rmtree(f"{defect_name}_0")
        os.remove("INCAR")

        # test warning when input file doesn't match expected format:
        os.remove("distortion_metadata.json")
        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                snb,
                [
                    "generate_all",
                    "-d",
                    f"{defects_dir}/",
                    "-b",
                    f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                    "--code",
                    "vasp",
                    "--input_file",
                    "test_config.yml",
                    "--config",
                    "test_config.yml",
                ],
                catch_exceptions=True,
            )
        dist = "Unperturbed"
        incar_dict = Incar.from_file(f"{defect_name}_0/{dist}/INCAR").as_dict()
        self.assertEqual(incar_dict["IBRION"], 2)  # default setting
        # assert UserWarning about unparsed input file
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        self.assertEqual(len(user_warnings), 1)
        self.assertEqual(
            "Input file test_config.yml specified but no valid INCAR tags found. "
            "Should be in the format of VASP INCAR file.",
            str(user_warnings[-1].message),
        )
        for file in ["KPOINTS", "POTCAR", "POSCAR"]:
            self.assertTrue(os.path.exists(f"{defect_name}_0/{dist}/{file}"))
        shutil.rmtree(f"{defect_name}_0")

        # Test CASTEP
        with open("castep.param", "w") as fp:
            fp.write("XC_FUNCTIONAL: PBE \n MAX_SCF_CYCLES: 100 \n CHARGE: 0")
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate_all",
                "-d",
                f"{defects_dir}/",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "--code",
                "castep",
                "--input_file",
                "castep.param",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=True,
        )
        dist = "Unperturbed"
        with open(f"{defect_name}_0/{dist}/castep.param") as fp:
            castep_lines = [line.strip() for line in fp.readlines()[-3:]]
        self.assertEqual(
            ["XC_FUNCTIONAL: PBE", "MAX_SCF_CYCLES: 100", "CHARGE: 0"], castep_lines
        )
        shutil.rmtree(f"{defect_name}_0")
        os.remove("castep.param")

        # Test CP2K
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate_all",
                "-d",
                f"{defects_dir}/",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "--code",
                "cp2k",
                "--input_file",
                f"{self.DATA_DIR}/cp2k/cp2k_input_mod.inp",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        dist = "Unperturbed"
        self.assertTrue(os.path.exists(f"{defect_name}_0/{dist}"))
        with open(f"{defect_name}_0/{dist}/cp2k_input.inp") as fp:
            input_cp2k = fp.readlines()
        self.assertEqual(
            "CUTOFF [eV] 800 ! PW cutoff",
            input_cp2k[15].strip(),
        )
        shutil.rmtree(f"{defect_name}_0")

        # Test Quantum Espresso
        test_yml = f"""
        defects:
            {defect_name}:
                charges: [0,]
                defect_coords: [0.0, 0.0, 0.0]
        bond_distortions: [0.3,]
        pseudopotentials:
            'Cd': 'Cd_pbe_v1.uspp.F.UPF'
            'Te': 'Te.pbe-n-rrkjus_psl.1.0.0.UPF'
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
                "--code",
                "espresso",
                "--input_file",
                f"{self.DATA_DIR}/quantum_espresso/qe.in",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        dist = "Unperturbed"
        with open(f"{defect_name}_0/{dist}/espresso.pwi") as fp:
            input_qe = fp.readlines()
        self.assertEqual(
            "title            = 'Si bulk'",
            input_qe[2].strip(),
        )
        shutil.rmtree(f"{defect_name}_0")

        # Test FHI-aims
        runner = CliRunner()
        result = runner.invoke(
            snb,
            [
                "generate_all",
                "-d",
                f"{defects_dir}/",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "--code",
                "fhiaims",
                "--input_file",
                f"{self.DATA_DIR}/fhi_aims/control.in",
                "--config",
                "test_config.yml",
            ],
            catch_exceptions=False,
        )
        dist = "Unperturbed"
        with open(f"{defect_name}_0/{dist}/control.in") as fp:
            input_aims = fp.readlines()
        self.assertEqual(
            "xc                                 pbe",
            input_aims[6].strip(),
        )
        self.assertEqual(
            "sc_iter_limit                      100.0",
            input_aims[10].strip(),
        )
        shutil.rmtree(f"{defect_name}_0")
        self.tearDown()

    def test_generate(self):
        "Test generate command"

        test_yml = """
bond_distortions: [-0.5,]
stdev: 0.15
d_min: 2.1250262890187375  # 0.75 * 2.8333683853583165
nbr_cutoff: 3.4
n_iter: 3
active_atoms: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # np.arange(0,31)
width: 0.3
max_attempts: 10000
max_disp: 1.0
seed: 20
local_rattle: False
POTCAR:
  Cd: Cd_GW
"""
        with open("test_config.yml", "w+") as fp:
            fp.write(test_yml)
        defect_name = "v_Cd_s0"  # SnB default name
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
        self.assertTrue(os.path.exists(f"./{defect_name}_0"))
        self.assertTrue(os.path.exists(f"./{defect_name}_0/Bond_Distortion_-50.0%"))
        V_Cd_kwarged_POSCAR = Poscar.from_file(
            f"./{defect_name}_0/Bond_Distortion_-50.0%/POSCAR"
        )
        self.assertEqual(
            V_Cd_kwarged_POSCAR.structure, self.V_Cd_minus0pt5_struc_kwarged
        )
        for file in ["KPOINTS", "POTCAR", "INCAR"]:
            self.assertTrue(
                os.path.exists(f"{defect_name}_0/Bond_Distortion_-50.0%/{file}")
            )
        # Check POTCAR file
        with open(f"{defect_name}_0/Bond_Distortion_-50.0%/POTCAR") as myfile:
            first_line = myfile.readline()
        self.assertIn("PAW_PBE Cd_GW", first_line)
        # Check KPOINTS file
        kpoints = Kpoints.from_file(
            f"{defect_name}_0/Bond_Distortion_-50.0%/" + "KPOINTS"
        )
        self.assertEqual(kpoints.kpts, [[1, 1, 1]])
        # Check INCAR
        incar = Incar.from_file(f"{defect_name}_0/Bond_Distortion_-50.0%/" + "INCAR")
        self.assertEqual(incar.pop("IBRION"), 2)
        self.assertEqual(incar.pop("EDIFF"), 1e-5)
        self.assertEqual(incar.pop("ROPT"), "1e-3 1e-3")

        # Test custom name
        defect_name = "vac_1_Cd"
        result = runner.invoke(
            snb,
            [
                "generate",
                "-d",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_V_Cd_POSCAR",
                "-b",
                f"{self.VASP_CDTE_DATA_DIR}/CdTe_Bulk_Supercell_POSCAR",
                "-c 0",
                "-n",
                "vac_1_Cd",
                "--config",
                f"test_config.yml",
            ],
            catch_exceptions=False,
        )
        cwd = os.getcwd()
        self.assertEqual(result.exit_code, 0)
        # self.assertTrue(os.path.exists(f"{cwd}/vac_1_Cd_0"))
        self.assertTrue(os.path.exists(f"{cwd}/vac_1_Cd_0/Bond_Distortion_-50.0%"))

        # test warning when input file doesn't match expected format:
        os.remove("distortion_metadata.json")
        with warnings.catch_warnings(record=True) as w:
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
                    "--input_file",
                    f"test_config.yml",
                ],
                catch_exceptions=False,
            )
        incar_dict = Incar.from_file(
            f"{defect_name}_0/Bond_Distortion_-50.0%/INCAR"
        ).as_dict()
        self.assertEqual(incar_dict["IBRION"], 2)  # default setting
        # assert UserWarning about unparsed input file
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        self.assertEqual(len(user_warnings), 2)  # wrong INCAR format and overwriting folder
        self.assertTrue(
            any("Input file test_config.yml specified but no valid INCAR tags found. "
            "Should be in the format of VASP INCAR file."
            in str(warning.message) for warning in user_warnings)
        )
        self.assertTrue(  # here we get this warning because no Unperturbed structures were
            # written so couldn't be compared
            any(f"The previously-generated defect folder v_Cd_s0_0 in "
            f"{os.path.basename(os.path.abspath('.'))} has the same Unperturbed defect structure "
            f"as the current defect species: v_Cd_s0_0. ShakeNBreak files in v_Cd_s0_0 will be "
            f"overwritten." in str(warning.message) for warning in user_warnings)
        )
        for file in ["KPOINTS", "POTCAR", "POSCAR"]:
            self.assertTrue(
                os.path.exists(f"{defect_name}_0/Bond_Distortion_-50.0%/{file}")
            )


if __name__ == "__main__":
    unittest.main()
