import datetime
import os
import shutil
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from monty.serialization import dumpfn, loadfn

from shakenbreak import analysis, plotting


def if_present_rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


_file_path = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_file_path, "data")


class PlottingDefectsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.VASP_CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "vasp/CdTe")
        self.organized_V_Cd_distortion_data = loadfn(
            os.path.join(self.VASP_CDTE_DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.yaml")
        )
        self.organized_V_Cd_distortion_data_no_unperturbed = loadfn(
            os.path.join(
                self.VASP_CDTE_DATA_DIR,
                "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.yaml",
            )
        )
        self.V_Cd_energies_dict = dict(
            analysis.get_energies(
                defect_species="vac_1_Cd_0",
                output_path=self.VASP_CDTE_DATA_DIR,
            )
        )
        self.V_Cd_displacement_dict = analysis.calculate_struct_comparison(
            defect_structures_dict=analysis.get_structures(
                defect_species="vac_1_Cd_0",
                output_path=self.VASP_CDTE_DATA_DIR,
            )
        )
        self.V_O_energies_dict_afm = analysis._sort_data(
            energies_file=f"{self.DATA_DIR}/vasp/rTiO2_vac_2_O_0_nupdown_0.yaml",
        )[0]
        self.V_O_energies_dict_fm = analysis._sort_data(
            energies_file=f"{self.DATA_DIR}/vasp/rTiO2_vac_2_O_0_nupdown_2.yaml",
        )[0]
        self.V_Cd_energies_dict_from_other_charge_states = analysis._sort_data(
            energies_file=f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_0/fake_vac_1_Cd_0.yaml"
        )[0]

        self.V_Cd_m2_energies_dict = analysis._sort_data(
            energies_file=f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2.yaml"
        )[0]

        self.V_Cd_m2_energies_dict_from_other_charge_states = analysis._sort_data(
            energies_file=f"{self.VASP_CDTE_DATA_DIR}/fake_V_Cd_-2_from_other_charge_states.yaml"
        )[0]

        if not os.path.exists(f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2"):
            os.mkdir(f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2")
            shutil.copyfile(
                f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2.yaml",
                f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2/vac_1_Cd_-2.yaml",
            )

    def tearDown(self):
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2")
        for file in os.listdir(f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_0"):
            if file.endswith(".svg") or file.endswith(".png"):
                os.remove(f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_0/{file}")
        for file in os.listdir(self.VASP_CDTE_DATA_DIR):
            if file.endswith(".svg") or file.endswith(".png"):
                os.remove(f"{self.VASP_CDTE_DATA_DIR}/{file}")
        if_present_rm("Int_Se_1_6.png")
        if_present_rm("as_2_O_on_I_1.png")
        if_present_rm("vac_1_Cd_0.png")

    def test_verify_data_directories_exist(self):
        """Test _verify_data_directories_exist() function"""
        self.assertRaises(
            FileNotFoundError,
            plotting._verify_data_directories_exist,
            output_path="./fake_path",
            defect_species="vac_1_Cd_0",
        )
        self.assertRaises(
            FileNotFoundError,
            plotting._verify_data_directories_exist,
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="fake_defect_species",
        )

    def test_format_axis(self):
        """Test _format_axis() function"""
        # Test standard behaviour: labels and ticks
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            list(self.V_Cd_energies_dict["distortions"].values()),
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="$V_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        self.assertEqual(formatted_ax.yaxis.get_label().get_text(), "Energy (eV)")
        self.assertEqual(
            formatted_ax.xaxis.get_label().get_text(),
            "Bond Distortion Factor (for 2 Te near $V_{Cd}^{0}$)",
        )
        self.assertEqual(
            len(formatted_ax.yaxis.get_ticklabels()), 6 + 2
        )  # +2 bc MaxNLocator adds ticks
        # beyond axis limits for autoscaling reasons
        self.assertTrue(
            [
                float(
                    tick.get_text().replace("−", "-")
                )  # weird mpl ticker reformatting
                % 0.3
                == 0.0
                for tick in formatted_ax.xaxis.get_ticklabels()
            ]
        )  # x ticks should be multiples of 0.3
        # check x label if no nearest neighbour info
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            list(self.V_Cd_energies_dict["distortions"].values()),
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="$V_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom=None,
        )
        self.assertEqual(
            formatted_ax.xaxis.get_label().get_text(),
            "Bond Distortion Factor (for 2 NN near $V_{Cd}^{0}$)",
        )
        # check x label if no defect name
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            list(self.V_Cd_energies_dict["distortions"].values()),
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name=None,
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        self.assertEqual(
            formatted_ax.xaxis.get_label().get_text(), "Bond Distortion Factor"
        )

    def test_format_ticks(self):
        "Test format_ticks() function."
        # Test standard behaviour
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            list(self.V_Cd_energies_dict["distortions"].values()),
        )
        semi_formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="$V_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        semi_formatted_ax.set_ylim(
            -0.2, 0.4
        )  # set incorrect y limits (-0.2 rather than ~-0.8)
        semi_formatted_ax.set_yticks(
            ticks=[
                -0.804,
                -0.502,
                0.000,
            ],
            labels=[
                -0.804,
                -0.502,
                0.000,
            ],
            minor=False,
        )  # set incorrect y ticks
        formatted_ax = plotting._format_ticks(
            ax=deepcopy(semi_formatted_ax),
            energies_list=list(self.V_Cd_energies_dict["distortions"].values())
            + [
                self.V_Cd_energies_dict["Unperturbed"],
            ],
        )
        # Check limits of y axis are correct (shouldnt cut off any data point given -
        # note that purging of high energy points has been done before.)
        self.assertTrue(
            formatted_ax.get_ylim()[0]
            < min(
                [
                    self.V_Cd_energies_dict["Unperturbed"],
                ]
                + list(self.V_Cd_energies_dict["distortions"].values())
            ),
        )
        self.assertTrue(
            formatted_ax.get_ylim()[1]
            > max(
                [
                    self.V_Cd_energies_dict["Unperturbed"],
                ]
                + list(self.V_Cd_energies_dict["distortions"].values())
            ),
        )

        # check y tick locators set as expected (0.2 eV spacing for >0.6 eV range)
        np.testing.assert_array_equal(
            mpl.ticker.MultipleLocator(base=0.2).tick_values(-0.3, 0.7),
            formatted_ax.yaxis.get_major_locator().tick_values(-0.3, 0.7),
        )

        # check y tick locators set as expected (0.1 eV spacing for >0.3 eV range)
        fig, ax = plt.subplots(1, 1)
        energies = list(np.arange(-0.31, 0.0, 0.01))
        ax.plot(
            np.arange(len(energies)),
            energies,
        )
        semi_formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="$V_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        semi_formatted_ax.set_yticks(
            ticks=[
                -0.28041,
                -0.05011,
                0.0001,
            ],
            labels=[-0.31041, -0.05011, 0.0001],
            minor=False,
        )  # set incorrect y ticks
        formatted_ax = plotting._format_ticks(
            ax=deepcopy(semi_formatted_ax),
            energies_list=energies,
        )
        np.testing.assert_array_equal(
            mpl.ticker.MultipleLocator(base=0.1).tick_values(-0.3, 0.7),
            formatted_ax.yaxis.get_major_locator().tick_values(-0.3, 0.7),
        )

    def test_cast_energies_to_floats(self):
        """Test _cast_energies_to_floats() function."""
        # Check numbers given as str are succesfully converted to floats
        energies_dict = {
            "Unperturbed": "-0.99",
            "distortions": {-0.2: "-0.2", -0.3: "-0.45"},
        }
        casted_energies_dict = plotting._cast_energies_to_floats(
            energies_dict=energies_dict, defect_species="vac_1_Cd_0"
        )
        [
            self.assertIsInstance(energy, float)
            for energy in casted_energies_dict["distortions"].values()
        ]
        self.assertIsInstance(casted_energies_dict["Unperturbed"], float)

        # Check str letters are not converted to floats, and exception is raised
        self.assertRaises(
            ValueError,
            plotting._cast_energies_to_floats,
            {"distortions": {-0.3: "any_string", -0.2: -0.4}},
            "vac_1_Cd_0",
        )

    def test_change_energy_units_to_meV(self):
        """Test _change_energy_units_to_meV() function."""
        # Test standard behaviour
        (
            energies_dict,
            max_energy_above_unperturbed,
            y_label,
        ) = plotting._change_energy_units_to_meV(
            energies_dict=deepcopy(self.organized_V_Cd_distortion_data),
            max_energy_above_unperturbed=0.2,
            y_label="Energy (eV)",
        )
        self.assertEqual(
            energies_dict["distortions"],
            {
                k: 1000 * v
                for k, v in self.organized_V_Cd_distortion_data["distortions"].items()
            },
        )
        self.assertEqual(
            energies_dict["Unperturbed"],
            1000 * self.organized_V_Cd_distortion_data["Unperturbed"],
        )
        self.assertEqual(max_energy_above_unperturbed, 0.2 * 1000)
        self.assertEqual(y_label, "Energy (meV)")

    def test_purge_data_dict(self):
        """Test _purge_data_dict() function."""
        # Test if dictionaries have same data points when displacement dict is incomplete
        disp_dict = deepcopy(self.V_Cd_displacement_dict)
        disp_dict.pop(-0.6)  # Missing data point
        disp_dict, energies_dict = plotting._purge_data_dicts(
            energies_dict=deepcopy(self.organized_V_Cd_distortion_data),
            disp_dict=deepcopy(disp_dict),
        )
        self.assertEqual(
            set(list(disp_dict.keys()))
            - set(list(energies_dict["distortions"].keys())),
            {"Unperturbed"},
        )  # only difference should be Unperturbed
        # Test behaviour when energy dict is incomplete
        energies_dict = deepcopy(self.organized_V_Cd_distortion_data)
        energies_dict["distortions"].pop(-0.6)
        disp_dict, energies_dict = plotting._purge_data_dicts(
            energies_dict=deepcopy(self.organized_V_Cd_distortion_data),
            disp_dict=deepcopy(disp_dict),
        )
        self.assertEqual(
            set(list(disp_dict.keys()))
            - set(list(energies_dict["distortions"].keys())),
            {"Unperturbed"},
        )  # only difference should be Unperturbed

    def test_remove_high_energy_points(self):
        """Test _remove_high_energy_points() function"""
        purged_energies_dict, disp_dict = plotting._remove_high_energy_points(
            energies_dict={
                "distortions": {-0.4: 1.5, -0.5: 0.2, -0.6: -0.3},
                "Unperturbed": 0.0,
            },
            max_energy_above_unperturbed=0.4,
            disp_dict=None,
        )
        self.assertTrue(-0.4 not in purged_energies_dict["distortions"].keys())
        purged_energies_dict, disp_dict = plotting._remove_high_energy_points(
            energies_dict={
                "distortions": {-0.4: 1.5, -0.5: 0.2, -0.6: -0.3},
                "Unperturbed": 0.0,
            },
            max_energy_above_unperturbed=0.4,
            disp_dict={-0.4: 1.5, -0.5: 0.2, -0.6: -0.3, "Unperturbed": 0.0},
        )
        self.assertTrue(-0.4 not in purged_energies_dict["distortions"].keys())
        self.assertTrue(-0.4 not in disp_dict.keys())

    def test_get_displacement_dict(self):
        """Test _get_displacement_dict() function."""
        # non existent defect directory,
        # `_get_displacement_dict` should catch the FileNotFoundError and set add_colorbar to False
        with warnings.catch_warnings(record=True) as w:
            plotting._get_displacement_dict(
                defect_species="fake_defect_species",
                output_path=self.VASP_CDTE_DATA_DIR,
                metric="max_dist",
                energies_dict=self.V_Cd_energies_dict,
                add_colorbar=True,
            )
            warning_message = f"Could not find structures for {self.VASP_CDTE_DATA_DIR}/fake_defect_species. Colorbar will not be added to plot."
            user_warnings = [
                warning for warning in w if warning.category == UserWarning
            ]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn(warning_message, str(user_warnings[0].message))

        # Test if energies_dict and disp_dict are returned correctly (same keys) and add_colorbar is set to True
        add_colorbar, energies_dict, disp_dict = plotting._get_displacement_dict(
            defect_species="vac_1_Cd_0",
            output_path=self.VASP_CDTE_DATA_DIR,
            metric="max_dist",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=True,
        )
        self.assertTrue(add_colorbar)
        self.assertEqual(  # using sets here to ignore order of dicts
            set(energies_dict["distortions"].keys()).union(
                {
                    "Unperturbed",
                }
            ),
            set(disp_dict.keys()),
        )

    def test_save_plot(self):
        """Test _save_plot() function"""
        # Saving to defect_dir subfolder in output_path
        fig, ax = plt.subplots(1, 1)
        defect_name = "vac_1_Cd_0"
        if_present_rm(
            f"{os.path.join(self.VASP_CDTE_DATA_DIR, defect_name, defect_name)}.png"
        )
        with patch("builtins.print") as mock_print, warnings.catch_warnings(
            record=True
        ) as w:
            plotting._save_plot(
                fig=fig,
                defect_name=defect_name,
                output_path=self.VASP_CDTE_DATA_DIR,
                save_format="png",
            )
        self.assertTrue(
            os.path.exists(
                f"{os.path.join(self.VASP_CDTE_DATA_DIR, defect_name, defect_name)}.png"
            )
        )
        mock_print.assert_called_once_with("Plot saved to vac_1_Cd_0/vac_1_Cd_0.png")
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        self.assertEqual(len(user_warnings), 0)  # No warnings in this case
        if_present_rm(
            f"{os.path.join(self.VASP_CDTE_DATA_DIR, defect_name, defect_name)}.png"
        )

        # Saving to output_path where defect_dir is not in output_path and output_path is not cwd
        if_present_rm(f"./{defect_name}.svg")
        with patch("builtins.print") as mock_print:
            plotting._save_plot(
                fig=fig, defect_name=defect_name, output_path=".", save_format="svg"
            )
        self.assertTrue(os.path.exists(f"./{defect_name}.svg"))
        mock_print.assert_called_once_with("Plot saved to ./vac_1_Cd_0.svg")
        if_present_rm(f"./{defect_name}.svg")

        # Saving to defect_dir subfolder in output_path where output_path is cwd
        os.chdir(self.VASP_CDTE_DATA_DIR)
        if_present_rm(f"{defect_name}/{defect_name}.svg")
        with patch("builtins.print") as mock_print:
            plotting._save_plot(
                fig=fig, defect_name=defect_name, output_path=".", save_format="svg"
            )
        self.assertFalse(
            os.path.exists(f"./{defect_name}.svg")
        )  # not in cwd, in defect directory
        self.assertTrue(os.path.exists(f"{defect_name}/{defect_name}.svg"))
        mock_print.assert_called_once_with(
            f"Plot saved to {defect_name}/{defect_name}.svg"
        )
        if_present_rm(f"{defect_name}/{defect_name}.svg")
        os.chdir(_file_path)

        # test previously saved plot renaming and print statement
        plotting._save_plot(
            fig=fig, defect_name=defect_name, output_path=".", save_format="png"
        )
        self.assertTrue(os.path.exists(f"./{defect_name}.png"))
        with patch("builtins.print") as mock_print:
            plotting._save_plot(
                fig=fig, defect_name=defect_name, output_path=".", save_format="png"
            )
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        current_datetime_minus1min = (
            datetime.datetime.now() - datetime.timedelta(minutes=1)
        ).strftime(
            "%Y-%m-%d-%H-%M"
        )  # in case delay between writing and testing plot generation

        self.assertTrue(os.path.exists(f"./{defect_name}.png"))
        mock_print.assert_any_call("Plot saved to ./vac_1_Cd_0.png")
        self.assertTrue(
            os.path.exists(f"./{defect_name}_{current_datetime}.png")
            or os.path.exists(f"./{defect_name}_{current_datetime_minus1min}.png")
        )
        self.assertTrue(
            f"Previous version of {defect_name}.png found in output_path: './'. Will rename "
            f"old plot to {defect_name}_{current_datetime}.png."
            in mock_print.call_args_list[0][0][0]
            or f"Previous version of {defect_name}.png found in output_path: './'. Will rename "
            f"old plot to {defect_name}_{current_datetime_minus1min}.png."
            in mock_print.call_args_list[0][0][0]
        )
        self._remove_current_and_saved_plots(
            defect_name, current_datetime, current_datetime_minus1min
        )
        # test no print statements with verbose = False
        plotting._save_plot(
            fig=fig, defect_name=defect_name, output_path=".", save_format="png"
        )
        self.assertTrue(os.path.exists(f"./{defect_name}.png"))
        with patch("builtins.print") as mock_print:
            plotting._save_plot(
                fig=fig,
                defect_name=defect_name,
                output_path=".",
                save_format="png",
                verbose=False,
            )
        self.assertTrue(os.path.exists(f"./{defect_name}.png"))
        mock_print.assert_not_called()
        self._remove_current_and_saved_plots(
            defect_name, current_datetime, current_datetime_minus1min
        )

    def _remove_current_and_saved_plots(self, defect_name, current_datetime, current_datetime_minus1min):
        if_present_rm(f"./{defect_name}.png")
        if_present_rm(f"./{defect_name}_{current_datetime}.png")
        if_present_rm(f"./{defect_name}_{current_datetime_minus1min}.png")

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_max_dist.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_max_distance(self):
        """Test plot_colorbar() function with metric=max_dist (default)"""
        return plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_fake_defect_name.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_fake_defect_name(self):
        """Test plot_colorbar() function with wrong defect name"""
        return plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="sub_1_In_on_Cd_1",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_displacement.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_displacement(self):
        """Test plot_colorbar() function with metric=disp and num_nearest_neighbours=None"""
        return plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=None,
            neighbour_atom="Te",
            metric="disp",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="Cd_Te_s32c_2_displacement.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_SnB_naming_w_site_num(self):
        """Test plot_colorbar() function with SnB defect naming and
        `include_site_info_in_name=True`"""
        return plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="Cd_Te_s32c_2",
            include_site_info_in_name=True,
            num_nearest_neighbours=4,
            neighbour_atom="Te",
            metric="disp",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_maxdist_title_linecolor_label.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_legend_label_linecolor_title_saveplot(self):
        """Test plot_colorbar() function with several keyword arguments:
        line_color, title, y_label, save_format, legend_label and neighbour_atom=None"""
        fig = plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=2,
            neighbour_atom=None,
            legend_label="SnB: 2 Te",
            line_color="k",
            title="$V_{Cd}^{0}$",
            save_plot=True,
            y_label="E (eV)",
        )
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), "vac_1_Cd_0.png")))
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="Int_Se_1_6.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_with_rattled_and_imported(self):
        """Test plot_colorbar() function with both rattled and imported charge states"""
        energies_dict = OrderedDict(
            [
                ("Unperturbed", 0.0),
                (
                    "distortions",
                    OrderedDict(
                        [
                            ("-10.0%_from_+3", -0.11105410999999776),
                            ("-10.0%_from_+5", -0.1296013099999982),
                            ("-30.0%_from_-1", -0.005485309999983201),
                            ("40.0%_from_+2", -0.16661439000000655),
                            ("Rattled", -0.006377469999961249),
                            ("Rattled_from_-2", 0.028722570000013548),
                        ]
                    ),
                ),
            ]
        )

        disp_dict = {
            "-10.0%_from_+3": 1.112352762765469,
            "-10.0%_from_+5": 1.171835067840965,
            "Unperturbed": 3.1787764299674696e-15,
            "Rattled_from_-2": 0.2929755991022341,
            "40.0%_from_+2": 1.9083985789588656,
            "-30.0%_from_-1": 0.22806204637713037,
            "Rattled": 0.245274020987905,
        }

        fig = plotting.plot_colorbar(
            energies_dict=energies_dict,
            disp_dict=disp_dict,
            defect_species="Int_Se_1_6",
            save_format="png",
            save_plot=True,
        )
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), "Int_Se_1_6.png")))
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="as_2_O_on_I_1.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_with_multiple_imported_distortions_from_same_charge_state(self):
        """Test plot_datasets() where there are multiple imported distortions from the same charge state"""
        energies_dict = {
            "Unperturbed": -623.27865201,
            "distortions": {
                -0.5: -623.32030465,
                -0.4: -623.57455267,
                -0.3: -623.57592574,
                -0.2: -623.57884127,
                -0.1: -623.32750209,
                0.0: -623.31491063,
                0.1: -623.29844552,
                0.2: -623.3624884,
                0.3: -623.30635978,
                0.4: -623.33427719,
                0.5: -623.33924517,
                0.6: -621.33173607,
                "-10.0%_from_+2": -623.54367663,
                "-20.0%_from_+2": -623.59064927,
                "-30.0%_from_0": -623.49069487,
                "-30.0%_from_+2": -623.64233181,
                "-30.0%_from_+5": -624.65113207,
                "0.0%_from_+4": -624.65313207,
                "-15.0%_from_+7": -624.06113207,
                "30.0%_from_+5": -624.65513207,
                "25.0%_from_+7": -624.06513207,
                "10.0%_from_+2": -624.66110949,
                "20.0%_from_+2": -624.65991904,
                "30.0%_from_+2": -624.66113207,
            },
        }

        fig = plotting.plot_datasets(
            datasets=[energies_dict],
            defect_species="as_2_O_on_I_1",
            save_format="png",
            save_plot=True,
        )
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), "as_2_O_on_I_1.png")))
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_colors.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_keywords(self):
        """Test plot_datasets() function testing several keywords:
        colors, save_format, title, defect_species, title, neighbour_atom, num_nearest_neighbours,
        dataset_labels"""
        fig = plotting.plot_datasets(
            datasets=[self.V_O_energies_dict_fm, self.V_O_energies_dict_afm],
            dataset_labels=["FM", "AFM"],
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=2,
            neighbour_atom="Ti",
            title="$V_{O}^{0}$",
            save_plot=True,
            output_path=self.VASP_CDTE_DATA_DIR,
            save_format="png",
            colors=["green", "orange"],
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.VASP_CDTE_DATA_DIR, "vac_1_Cd_0/vac_1_Cd_0.png")
            )
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_notitle.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_without_saving(self):
        """Test plot_datasets() function testing several keywords:
        title = None, num_nearest_neighbours = None, neighbour_atom = None, save_plot = False
        and user specify style: markers, linestyles, markersize, linewidth"""

        fig = plotting.plot_datasets(
            datasets=[self.V_O_energies_dict_fm, self.V_O_energies_dict_afm],
            dataset_labels=["FM", "AFM"],
            defect_species="vac_1_Cd_0",
            save_plot=False,
            markers=["s", "<"],
            linestyles=[":", "-."],
            markersize=10,
            linewidth=3,
        )
        # Check plot not saved if save_plot=False
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), "vac_1_Cd_0.png")))
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_not_enough_markers.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_not_enough_markers(self):
        """Test plot_datasets() function when user does not provide enough markers and linestyles"""
        return plotting.plot_datasets(
            datasets=[self.V_O_energies_dict_fm, self.V_O_energies_dict_afm],
            dataset_labels=["FM", "AFM"],
            defect_species="vac_1_Cd_0",
            save_plot=False,
            markers=[
                "s",
            ],
            linestyles=[
                ":",
            ],
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_other_charge_states.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_from_other_charge_states(self):
        """Test plot_datasets() function when energy lowering distortions from other
        charge states have been tried"""
        return plotting.plot_datasets(
            datasets=[
                self.V_Cd_energies_dict_from_other_charge_states,
            ],
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_-2_only_rattled.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_only_rattled(self):
        """Test plot_datasets() function when the only distortion is 'Rattled'"""
        return plotting.plot_datasets(
            datasets=[
                self.V_Cd_m2_energies_dict,
            ],
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_-2_rattled_other_charge_states.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_rattled_and_dist_from_other_chargestates(self):
        """Test plot_datasets() function when the distortion is "Rattled"
        and distortions from other charge states have been tried"""
        return plotting.plot_datasets(
            datasets=[
                self.V_Cd_m2_energies_dict_from_other_charge_states,
            ],
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_-2_only_rattled_and_rattled_dist_from_other_charges_states.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_only_rattled_and_rattled_dist_from_other_charge_states(self):
        """Test plot_datasets() function when one of the energy lowering distortions from other
        charge states is Rattled (i.e. `Rattled_from_0`)"""
        # Fake dataset
        datasets = [
            self.V_Cd_m2_energies_dict_from_other_charge_states,
        ]
        datasets[0]["distortions"].pop("-52.5%_from_0")
        datasets[0]["distortions"].update({"Rattled_from_0": -205.92311458})
        return plotting.plot_datasets(
            datasets=datasets,
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
        )

    def test_plot_datasets_value_error(self):
        """Test plot_datasets() function when user provides non-matching `datasets` and
        `dataset_labels`"""
        self.assertRaises(
            ValueError,
            plotting.plot_datasets,
            datasets=[
                self.V_Cd_energies_dict_from_other_charge_states,
            ],
            dataset_labels=["SnB: 2 Te", "Another one"],
            defect_species="$V_{Cd}^{0}$",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    def test_plot_defect_fake_output_directories(self):
        """Test plot_defect() function when directory `output_path` does not exist"""
        self.assertRaises(
            FileNotFoundError,
            plotting.plot_defect,
            output_path="./fake_output_path",
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
        )
        # Test fake defect_species runs fine:
        plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="fake_defect",
            energies_dict=self.V_Cd_energies_dict,
        )
        os.remove(f"{self.VASP_CDTE_DATA_DIR}/fake_defect.png")

    def test_plot_defect_missing_unperturbed_energy(self):
        with warnings.catch_warnings(record=True) as w:
            plotting.plot_defect(
                output_path=self.VASP_CDTE_DATA_DIR,
                defect_species="vac_1_Cd_0",
                energies_dict=self.organized_V_Cd_distortion_data_no_unperturbed,
            )
        warning_message = "Unperturbed energy not present in energies_dict of vac_1_Cd_0! Skipping plot."
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        self.assertEqual(len(user_warnings), 1)
        self.assertEqual(warning_message, str(w[-1].message))

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_plot_defect_add_colorbar_max_dist.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_add_colorbar(self):
        """Test plot_defect() function when add_colorbar = True"""
        return plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=True,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_plot_defect_without_colorbar.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_without_colorbar(self):
        """Test plot_defect() function when add_colorbar = False"""
        return plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=False,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_plot_defect_with_unrecognised_name.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_unrecognised_name(self):
        """Test plot_defect() function when the name cannot be formatted (e.g. if parsing and
        plotting from a renamed folder)"""
        with warnings.catch_warnings(record=True) as w:
            fig = plotting.plot_defect(  # note this also implicitly tests that we can use
                # `plot_defect` with a `defect_species` that is not found in the `output_path` (but
                # is present in the `energies_dict`)
                output_path=self.VASP_CDTE_DATA_DIR,
                defect_species="vac_1_Cd_no_charge",
                energies_dict=self.V_Cd_energies_dict,
                add_colorbar=True,
                num_nearest_neighbours=2,
                neighbour_atom="Te",
            )
        self.assertTrue(
            any(
                f"Cannot add colorbar to plot for vac_1_Cd_no_charge as"
                f" {self.VASP_CDTE_DATA_DIR}/vac_1_Cd_no_charge cannot be found."
                in str(warning.message)
                for warning in w
            )
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="Te_i_Td_Te2.83_+2.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_doped_v2(self):
        """Test plot_defect() function using doped v2 naming"""
        return plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="Te_i_Td_Te2.83_+2",
            energies_dict=self.V_Cd_energies_dict,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
            include_site_info_in_name=False,
            save_plot=False,
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_include_site_info_in_name.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_include_site_info_in_name(self):
        """Test plot_defect() function when include_site_info_in_name = True"""
        return plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=False,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
            include_site_info_in_name=True,
            save_plot=False,
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="Te_i_Td_Te2.83_+2_include_site_info_in_name.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_include_site_info_in_name_doped_v2(self):
        """Test plot_defect() function when include_site_info_in_name = True, using doped v2 naming"""
        return plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="Te_i_Td_Te2.83_+2",
            energies_dict=self.V_Cd_energies_dict,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
            include_site_info_in_name=True,
            save_plot=False,
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_0_plot_defect_without_title_units_meV.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_without_title_units_in_meV(self):
        """Test plot_defect() function when add_title = False and units = 'meV'"""
        return plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=False,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
            add_title=False,
            units="meV",
        )

    # Most keywords of `plot_all_defects` have already been tested by tests of
    # `plot_defect`, `plot_datasets` and `plot_colorbar`. Now we test:
    # incorrect `output_path`, non-existent defect directory,
    # correct output of `plot_all_defects` (dict of figures), behaviour when distortion_metadata
    # not found or missing entry for a given charge state, and finally the keyword `min_e_diff`

    def test_plot_all_defects_incorrect_output_path(self):
        """Test plot_all_defects() function when `output_path` is incorrect"""
        self.assertRaises(
            FileNotFoundError,
            plotting.plot_all_defects,
            output_path="./fake_output_path",
            defects_dict={
                "vac_1_Cd": [
                    0,
                ]
            },
            save_plot=False,
        )

    def test_plot_all_defects_nonexistent_defect_folder(self):
        """Test plot_all_defects() function when one of the defect folders does not exist"""
        with warnings.catch_warnings(record=True) as w:
            plotting.plot_all_defects(
                output_path=self.VASP_CDTE_DATA_DIR,
                defects_dict={
                    "vac_1_Cd": [
                        -1,
                    ]
                },
                save_plot=False,
            )
            self.assertTrue(
                "vac_1_Cd_-1 does not exist! Skipping vac_1_Cd_-1."
                in str(w[-1].message)
            )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{_DATA_DIR}/remote_baseline_plots",
        filename="vac_1_Cd_-2_only_rattled.png",
        style=f"{_file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defects_output(self):
        """Test output of plot_all_defects() function. Test plot still generated when
        distortion_metadata.json does not contain info for a given charge state"""
        fig_dict = plotting.plot_all_defects(
            output_path=self.VASP_CDTE_DATA_DIR,
            defects_dict={"vac_1_Cd": [0, -2]},
            save_plot=False,
            min_e_diff=0.05,
            add_title=False,
        )
        [
            self.assertIsInstance(figure, mpl.figure.Figure)
            for figure in fig_dict.values()
        ]
        self.assertEqual(list(fig_dict.keys()), ["vac_1_Cd_0", "vac_1_Cd_-2"])
        # No info on distortion_metadata.json for charge state -2, so its x label should be 'Bond
        # Distortion Factor'
        return fig_dict["vac_1_Cd_-2"]

    def test_plot_all_defects_min_e_diff(self):
        """Test plot_all_defects() function with keyword min_e_diff set"""
        fig_dict = plotting.plot_all_defects(
            output_path=self.VASP_CDTE_DATA_DIR,
            defects_dict={"vac_1_Cd": [0, -2]},
            save_plot=False,
            min_e_diff=0.15,
        )
        self.assertTrue("vac_1_Cd_-2" not in fig_dict.keys())


if __name__ == "__main__":
    unittest.main()
