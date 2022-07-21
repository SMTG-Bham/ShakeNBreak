import unittest
import os
import datetime
import shutil

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar

from click.testing import CliRunner
from shakenbreak.cli import snb


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
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.V_Cd_minus0pt5_struc_rattled = Structure.from_file(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-50%_Distortion_Rattled_POSCAR"
            )
        )

    def tearDown(self):
        for i in [
            "parsed_defects_dict.pickle",
            "Vac_Cd_mult32_0",
            "distortion_metadata.json",
        ]:
            if_present_rm(i)

        for i in os.listdir("."):
            if "distortion_metadata" in i:
                os.remove(i)

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
            + "        Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "        Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
            result.output,
        )
        self.assertIn("--Distortion -40.0%", result.output)
        self.assertIn(
            f"\tDefect Site Index / Frac Coords: [0. 0. 0.]\n"
            + "        Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "        Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
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
            V_Cd_minus0pt5_rattled_POSCAR.structure, self.V_Cd_minus0pt5_struc_rattled
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
            + "        Original Neighbour Distances: [(2.83, 33, 'Te'), (2.83, 42, 'Te')]\n"
            + "        Distorted Neighbour Distances:\n\t[(1.13, 33, 'Te'), (1.13, 42, 'Te')]",
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

        # only test POSCAR as INCAR, KPOINTS and POTCAR not written on GitHub actions,
        # but tested locally -- add CLI INCAR KPOINTS and POTCAR local tests!
        # test all options etc etc


if __name__ == "__main__":
    unittest.main()
