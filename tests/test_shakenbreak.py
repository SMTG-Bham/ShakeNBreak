import copy
import os
import shutil
import unittest
import warnings
from unittest.mock import call, patch

import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import UnknownPotcarWarning

from shakenbreak import cli, energy_lowering_distortions, input, io, plotting

file_path = os.path.dirname(__file__)


def if_present_rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class ShakeNBreakTestCase(unittest.TestCase):  # integration testing ShakeNBreak
    def setUp(self):
        warnings.simplefilter("ignore", UnknownPotcarWarning)
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        # Refactor doped defect dict to dict of Defect() objects
        self.cdte_doped_defect_dict = loadfn(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_defects_dict.json")
        )

        self.V_Cd_dict = self.cdte_doped_defect_dict["vacancies"][0]

        self.V_Cd = input.generate_defect_object(self.V_Cd_dict, self.cdte_doped_defect_dict["bulk"])
        self.V_Cd_minus_0pt55_structure = Structure.from_file(
            self.VASP_CDTE_DATA_DIR + "/vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR"
        )

        # create fake distortion folders for testing functionality:
        for defect_dir in ["vac_1_Cd_-1", "vac_1_Cd_-2"]:
            if_present_rm(defect_dir)
            os.mkdir(f"{defect_dir}")
        V_Cd_1_dict = {"distortions": {-0.075: -206.700}, "Unperturbed": -205.8}
        dumpfn(V_Cd_1_dict, "vac_1_Cd_-1/vac_1_Cd_-1.yaml")
        V_Cd_2_dict = {"distortions": {-0.35: -205.7}, "Unperturbed": -205.8}
        dumpfn(V_Cd_2_dict, "vac_1_Cd_-2/vac_1_Cd_-2.yaml")

        # create fake structures for testing functionality:
        for fake_dir in ["Bond_Distortion_-7.5%", "Unperturbed"]:
            if_present_rm(f"vac_1_Cd_-1/{fake_dir}")
            os.mkdir(f"vac_1_Cd_-1/{fake_dir}")
            shutil.copyfile(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-1_vgam_POSCAR"),
                f"vac_1_Cd_-1/{fake_dir}/CONTCAR",
            )

        for fake_dir in ["Bond_Distortion_-35.0%", "Unperturbed"]:
            if_present_rm(f"vac_1_Cd_-2/{fake_dir}")
            os.mkdir(f"vac_1_Cd_-2/{fake_dir}")
            shutil.copyfile(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_POSCAR"),
                f"vac_1_Cd_-2/{fake_dir}/CONTCAR",
            )

        for charge in [-1,-2]:
            shutil.copyfile(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_POSCAR"),
                f"vac_1_Cd_{charge}/Unperturbed/POSCAR",
            )  # so when we generate SnB files in `test_SnB_integration` it recognises it as
            # being the same defect

        self.defect_charges_dict = (
            energy_lowering_distortions.read_defects_directories()
        )
        self.defect_charges_dict.pop("vac_1_Ti", None)  # Used for magnetization tests

    def tearDown(self):
        for i in os.listdir():
            if "vac_1_Cd" in i:
                if_present_rm(i)
        if_present_rm("distortion_metadata.json")
        if_present_rm("parsed_defects_dict.json")

    def test_SnB_integration(self):
        """Test full ShakeNBreak workflow, for the tricky case where at least 2
        _different_energy-lowering distortions are found for other charge states
        (y and z) that are then to be tested for a different charge state (x),
        then reparsed and plotted successfully
        """
        oxidation_states = {"Cd": +2, "Te": -2}
        reduced_V_Cd = copy.copy(self.V_Cd)
        reduced_V_Cd.user_charges = [-2, -1, 0]
        reduced_V_Cd_enties = [
            input._get_defect_entry_from_defect(reduced_V_Cd, charge)
            for charge in reduced_V_Cd.user_charges
        ]

        # Generate input files
        dist = input.Distortions(
            {"vac_1_Cd": reduced_V_Cd_enties},
            oxidation_states=oxidation_states,
        )
        distortion_defect_dict, structures_defect_dict = dist.write_vasp_files(
            incar_settings={"ENCUT": 212, "IBRION": 0, "EDIFF": 1e-4},
            verbose=False,
        )
        shutil.rmtree("vac_1_Cd_0")
        shutil.copytree(
            os.path.join(self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0"), "vac_1_Cd_0"
        )  # overwrite

        defect_charges_dict = energy_lowering_distortions.read_defects_directories()
        defect_charges_dict.pop("vac_1_Ti", None)  # Used for magnetization tests

        low_energy_defects = (
            energy_lowering_distortions.get_energy_lowering_distortions(
                defect_charges_dict
            )
        )

        self.assertEqual(
            sorted([[0], [-1]]),  # sort to ensure order is the same
            sorted([subdict["charges"] for subdict in low_energy_defects["vac_1_Cd"]]),
        )
        self.assertEqual(
            sorted([sorted(tuple({-2, -1})), sorted(tuple({0, -2}))]),
            sorted(
                [
                    sorted(tuple(subdict["excluded_charges"]))
                    for subdict in low_energy_defects["vac_1_Cd"]
                ]
            ),
        )
        # So the dimer (0) and polaron (-1) structures should be generated and tested for -2

        with patch("builtins.print") as mock_print:
            energy_lowering_distortions.write_retest_inputs(low_energy_defects)

            mock_print.assert_any_call(
                "Writing low-energy distorted structure to "
                "./vac_1_Cd_-2/Bond_Distortion_-55.0%_from_0"
            )
            mock_print.assert_any_call(
                "Writing low-energy distorted structure to "
                "./vac_1_Cd_-1/Bond_Distortion_-55.0%_from_0"
            )
            mock_print.assert_any_call(
                "Writing low-energy distorted structure to "
                "./vac_1_Cd_0/Bond_Distortion_-7.5%_from_-1"
            )
            mock_print.assert_any_call(
                "Writing low-energy distorted structure to "
                "./vac_1_Cd_-2/Bond_Distortion_-7.5%_from_-1"
            )

        # test correct structures written
        gen_struc = Structure.from_file("vac_1_Cd_-2/Bond_Distortion_-55.0%_from_0/POSCAR")
        gen_struc.remove_oxidation_states()
        self.assertEqual(
            self.V_Cd_minus_0pt55_structure,
            gen_struc,
        )
        gen_struc = Structure.from_file("vac_1_Cd_0/Bond_Distortion_-7.5%_from_-1/POSCAR")
        gen_struc.remove_oxidation_states()
        self.assertEqual(
            Structure.from_file(
                os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-1_vgam_POSCAR")
            ),
            gen_struc,
        )

        V_Cd_m1_dict_w_distortion = {
            "distortions": {-0.075: -206.7, "-55.0%_from_0": -207.0},
            "Unperturbed": -205.8,
        }
        dumpfn(V_Cd_m1_dict_w_distortion, "vac_1_Cd_-1/vac_1_Cd_-1.yaml")

        V_Cd_m2_dict_w_distortion = {
            "distortions": {
                -0.35: -205.7,
                "-55.0%_from_0": -207.0,
                "-7.5%_from_-1": -207.7,
            },
            "Unperturbed": -205.8,
        }
        dumpfn(V_Cd_m2_dict_w_distortion, "vac_1_Cd_-2/vac_1_Cd_-2.yaml")

        # note we're not updating vac_1_Cd_0.yaml here, to test the info message that the
        # Bond_Distortion_-7.5%_from_-1 folder is already present in this directory

        shutil.copyfile(
            os.path.join(
                self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/Bond_Distortion_-55.0%/CONTCAR"
            ),
            "vac_1_Cd_-1/Bond_Distortion_-55.0%_from_0/CONTCAR",
        )
        shutil.copyfile(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-1_vgam_POSCAR"),
            "vac_1_Cd_-2/Bond_Distortion_-7.5%_from_-1/CONTCAR",
        )
        shutil.copyfile(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_V_Cd_-1_vgam_POSCAR"),
            "vac_1_Cd_0/Bond_Distortion_-7.5%_from_-1/CONTCAR",
        )

        with patch("builtins.print") as mock_print:
            low_energy_defects = (
                energy_lowering_distortions.get_energy_lowering_distortions(
                    defect_charges_dict
                )
            )
        mock_print.assert_any_call(
            "vac_1_Cd_0: Energy difference between minimum, found with -0.55 bond distortion, "
            "and unperturbed: -0.76 eV."
        )
        mock_print.assert_any_call(
            "Comparing structures to specified ref_structure (Cd31 Te32)..."
        )
        mock_print.assert_any_call(
            "\nComparing and pruning defect structures across charge states..."
        )
        try:
            mock_print.assert_any_call(
                "Low-energy distorted structure for vac_1_Cd_-1 already "
                "found with charge states [0], storing together."
            )
        except AssertionError:  # depends on parsing order, different on GH Actions to local
            mock_print.assert_any_call(
                "Low-energy distorted structure for vac_1_Cd_0 already "
                "found with charge states [-1], storing together."
            )

        # Test that energy_lowering_distortions parsing functions run ok if run on folders where
        # we've already done _some_ re-tests from other structures (-55.0%_from_0 for -1 but not
        # -2 and -7.5%_from_-1 for -2 but not for 0)(i.e. if we did this parsing early when only
        # some of the other charge states had converged etc)
        with patch("builtins.print") as mock_print:
            energy_lowering_distortions.write_retest_inputs(low_energy_defects)

            mock_print.assert_any_call(
                "As ./vac_1_Cd_0/Bond_Distortion_-7.5%_from_-1 already exists, it's assumed this "
                "structure has already been tested. Skipping..."
            )
            mock_print.assert_any_call(
                "As ./vac_1_Cd_-2/Bond_Distortion_-55.0%_from_0 already exists, it's assumed this "
                "structure has already been tested. Skipping..."
            )

        self.assertEqual(
            sorted(
                [sorted(tuple([-2])), sorted(tuple([0, -1]))]
            ),  # sort to make sure order is the same
            sorted(
                [
                    sorted(tuple(subdict["charges"]))
                    for subdict in low_energy_defects["vac_1_Cd"]
                ]
            ),
        )
        self.assertEqual(
            sorted([tuple({0}), tuple({-2})]),
            sorted(
                [
                    tuple(subdict["excluded_charges"])
                    for subdict in low_energy_defects["vac_1_Cd"]
                ]
            ),
        )

    # Now we test parsing of final energies and plotting

    @pytest.mark.mpl_image_compare(
        baseline_dir="data/remote_baseline_plots",
        filename="vac_1_Cd_-2.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_fake_vac_1_Cd_m2(self):
        defect_dir = "vac_1_Cd_-2"
        if_present_rm(defect_dir)
        os.mkdir(defect_dir)
        for dist, energy in {  # Fake energies
            "Bond_Distortion_-35.0%": -205.7,
            "Bond_Distortion_-77.0%_High_Energy": 1000.0,  # positive energy
            "Bond_Distortion_-50.0%_from_0": -206.5,
            "Bond_Distortion_0.0%": -205.6,
            "Unperturbed": -205.4,
        }.items():
            # Just using relevant part of the OUTCAR file to quickly test parsing
            # as parsing of the full file has been extensively tested in test_cli.py
            outcar = f"""
            reached required accuracy - stopping structural energy minimisationss
            FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
            ---------------------------------------------------
            free  energy   TOTEN  =     -1173.02056574 eV

            energy  without entropy=    -1173.02056574  energy(sigma->0) =    {energy}

            d Force = 0.4804267E-04[ 0.582E-05, 0.903E-04]  d Energy = 0.2446833E-04 0.236E-04
            d Force =-0.1081853E+01[-0.108E+01,-0.108E+01]  d Ewald  =-0.1081855E+01 0.235E-05
            """
            os.mkdir(f"{defect_dir}/{dist}")
            with open(f"{defect_dir}/{dist}/OUTCAR", "w") as f:
                f.write(outcar)

        # Parse final energies from OUTCAR files and write them to yaml files
        io.parse_energies(defect=defect_dir, path="./")
        self.assertTrue(os.path.exists(f"{defect_dir}/{defect_dir}.yaml"))
        energies = loadfn(f"{defect_dir}/{defect_dir}.yaml")
        self.assertTrue(-0.35 in energies["distortions"])
        self.assertFalse(-0.77 in energies["distortions"])

        defect_charges_dict = energy_lowering_distortions.read_defects_directories()
        defect_charges_dict.pop("vac_1_Ti", None)  # Used for magnetization tests

        fig_dict = plotting.plot_all_defects(defect_charges_dict, save_format="png")
        return fig_dict["vac_1_Cd_-2"]

    @pytest.mark.mpl_image_compare(
        baseline_dir="data/remote_baseline_plots",
        filename="vac_1_Cd_-1.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_fake_vac_1_Cd_m1(self):
        defect_dir = "vac_1_Cd_-1"
        if_present_rm(defect_dir)
        os.mkdir(defect_dir)
        for dist, energy in {  # Fake energies
            "Bond_Distortion_-35.0%": -205.7,
            "Bond_Distortion_0.0%": -205.6,
            "Unperturbed": -205.4,
            "Bond_Distortion_-50.0%_from_0": -206.5,
            "Bond_Distortion_-20.0%_from_-1": -206.5,
        }.items():
            # Just using relevant part of the OUTCAR file to quickly test parsing
            # as parsing of the full file has been extensively tested in test_cli.py
            outcar = f"""
            reached required accuracy - stopping structural energy minimisation
            FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
            ---------------------------------------------------
            free  energy   TOTEN  =     -1173.02056574 eV

            energy  without entropy=    -1173.02056574  energy(sigma->0) =    {energy}

            d Force = 0.4804267E-04[ 0.582E-05, 0.903E-04]  d Energy = 0.2446833E-04 0.236E-04
            d Force =-0.1081853E+01[-0.108E+01,-0.108E+01]  d Ewald  =-0.1081855E+01 0.235E-05
            """
            os.mkdir(f"{defect_dir}/{dist}")
            with open(f"{defect_dir}/{dist}/OUTCAR", "w") as f:
                f.write(outcar)

        # Parse final energies from OUTCAR files and write them to yaml files
        io.parse_energies(defect=defect_dir, path="./")
        self.assertTrue(os.path.exists(f"{defect_dir}/{defect_dir}.yaml"))

        defect_charges_dict = energy_lowering_distortions.read_defects_directories()
        defect_charges_dict.pop("vac_1_Ti", None)  # Used for magnetization tests

        fig_dict = plotting.plot_all_defects(defect_charges_dict, save_format="png")
        return fig_dict["vac_1_Cd_-1"]

    @pytest.mark.mpl_image_compare(
        baseline_dir="data/remote_baseline_plots",
        filename="vac_1_Cd_0.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_fake_vac_1_Cd_0(self):
        defect_dir = "vac_1_Cd_0"
        if_present_rm(defect_dir)
        os.mkdir(defect_dir)
        for dist, energy in {  # fake energies
            "Bond_Distortion_-40.0%": -205.7,
            "Bond_Distortion_-20.0%": -206.7,
            "Bond_Distortion_0.0%": -205.6,
            "Bond_Distortion_20.0%": -206.7,
            "Bond_Distortion_40.0%": -206.7,
            "Bond_Distortion_-50.0%_from_0": -206.5,
            "Unperturbed": -205.4,
        }.items():
            # Just using relevant part of the OUTCAR file to quickly test parsing
            # as parsing of the full file has been extensively tested in test_cli.py
            outcar = f"""
            reached required accuracy - stopping structural energy minimisation
            FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
            ---------------------------------------------------
            free  energy   TOTEN  =     -1173.02056574 eV

            energy  without entropy=    -1173.02056574  energy(sigma->0) =    {energy}

            d Force = 0.4804267E-04[ 0.582E-05, 0.903E-04]  d Energy = 0.2446833E-04 0.236E-04
            d Force =-0.1081853E+01[-0.108E+01,-0.108E+01]  d Ewald  =-0.1081855E+01 0.235E-05
            """
            os.mkdir(f"{defect_dir}/{dist}")
            with open(f"{defect_dir}/{dist}/OUTCAR", "w") as f:
                f.write(outcar)

        # Parse final energies from OUTCAR files and write them to yaml files
        io.parse_energies(defect=defect_dir, path="./")
        self.assertTrue(os.path.exists(f"{defect_dir}/{defect_dir}.yaml"))

        defect_charges_dict = energy_lowering_distortions.read_defects_directories()
        defect_charges_dict.pop("vac_1_Ti", None)  # Used for magnetization tests

        fig_dict = plotting.plot_all_defects(defect_charges_dict, save_format="png")
        return fig_dict["vac_1_Cd_0"]


if __name__ == "__main__":
    unittest.main()
