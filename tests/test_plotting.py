import unittest
import os
import shutil
import warnings
from copy import deepcopy
import pytest

import numpy as np
from monty.serialization import loadfn, dumpfn

import matplotlib as mpl
import matplotlib.pyplot as plt

from shakenbreak import analysis
from shakenbreak import plotting


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


file_path = os.path.dirname(__file__)


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
        if_present_rm(f"{os.getcwd()}/distortion_plots")
        if_present_rm(f"{self.VASP_CDTE_DATA_DIR}/vac_1_Cd_-2")

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
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        self.assertEqual(formatted_ax.yaxis.get_label().get_text(), "Energy (eV)")
        self.assertEqual(
            formatted_ax.xaxis.get_label().get_text(),
            "Bond Distortion Factor (for 2 Te near V$_{Cd}^{0}$)",
        )
        self.assertEqual(
            len(formatted_ax.yaxis.get_ticklabels()), 6 + 2
        )  # +2 bc MaxNLocator adds ticks
        # beyond axis limits for autoscaling reasons
        # self.assertTrue([float(tick.get_text()) % 0.3 == 0.0 for tick in formatted_ax.xaxis.get_ticklabels()]) # x ticks should be multiples of 0.3
        print(formatted_ax.xaxis.get_ticklabels())
        # check x label if no nearest neighbour info
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            list(self.V_Cd_energies_dict["distortions"].values()),
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom=None,
        )
        self.assertEqual(
            formatted_ax.xaxis.get_label().get_text(),
            "Bond Distortion Factor (for 2 NN near V$_{Cd}^{0}$)",
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

    def test_format_tick_labels(self):
        "Test format_tick_labels() function."
        # Test standard behaviour
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            list(self.V_Cd_energies_dict["distortions"].values()),
        )
        semi_formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        semi_formatted_ax.set_ylim(
            -0.2, 0.4
        )  # set incorrect y limits (-0.2 rather than ~-0.8)
        semi_formatted_ax.set_xticks(
            ticks=[
                -0.201,
                -0.101,
                0.000,
            ],
            labels=[
                -0.201,
                -0.101,
                0.000,
            ],
            minor=False,
        )  # set incorrect x ticks
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
        formatted_ax = plotting._format_tick_labels(
            ax=deepcopy(semi_formatted_ax),
            energy_range=list(self.V_Cd_energies_dict["distortions"].values())
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
        # Check x tick labels have 1 decimal place
        self.assertTrue(
            len(
                formatted_ax.xaxis.get_major_formatter()
                .format_data(0.11111)
                .split(".")[1]
            )
            == 1
        )
        # Check y tick labels have 1 decimal place if energy range is > 0.4 eV
        self.assertTrue(
            len(
                formatted_ax.yaxis.get_major_formatter()
                .format_data(0.11111)
                .split(".")[1]
            )
            == 1
        )

        # Test y tick labels have 3 decimal places if energy range is < 0.1 eV
        fig, ax = plt.subplots(1, 1)
        energies = [
            np.random.uniform(-0.099, 0)
            for i in range(len(self.V_Cd_energies_dict["distortions"].keys()))
        ]
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()),
            energies,
        )
        semi_formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        semi_formatted_ax.set_yticks(
            ticks=[
                -0.08041,
                -0.05011,
                0.0001,
            ],
            labels=[-0.08041, -0.05011, 0.0001],
            minor=False,
        )  # set incorrect y ticks
        formatted_ax = plotting._format_tick_labels(
            ax=deepcopy(semi_formatted_ax),
            energy_range=energies,
        )
        self.assertTrue(
            len(
                formatted_ax.yaxis.get_major_formatter()
                .format_data(0.11111)
                .split(".")[1]
            )
            == 3
        )

    def test_format_defect_name(self):
        """Test _format_defect_name() function."""
        # test standard behaviour
        formatted_name = plotting._format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_num_in_name=False,
        )
        self.assertEqual(formatted_name, "V$_{Cd}^{0}$")
        # test with site number included
        formatted_name = plotting._format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_num_in_name=True,
        )
        self.assertEqual(formatted_name, "V$_{Cd_1}^{0}$")
        # test interstitial case
        formatted_name = plotting._format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_num_in_name=True,
        )
        self.assertEqual(formatted_name, "Cd$_{i_1}^{0}$")
        # check exceptions raised: invalid charge or defect_species
        self.assertRaises(
            ValueError,
            plotting._format_defect_name,
            defect_species="vac_1_Cd_a",
            include_site_num_in_name=True,
        )
        self.assertRaises(
            TypeError,
            plotting._format_defect_name,
            defect_species=2,
            include_site_num_in_name=True,
        )
        # check invalid defect type
        self.assertRaises(
            ValueError,
            plotting._format_defect_name,
            defect_species="kk_Cd_1_0",
            include_site_num_in_name=True,
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
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertEqual(
                str(w[0].message),
                f"Could not find structures for {self.VASP_CDTE_DATA_DIR}/fake_defect_species. Colorbar will not be added to plot.",
            )
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
        fig, ax = plt.subplots(1, 1)
        if_present_rm(f"{os.getcwd()}/distortion_plots/vac_1_Cd_0.svg")
        plotting._save_plot(fig=fig, defect_name="vac_1_Cd_0", save_format="svg")
        self.assertTrue(
            os.path.exists(f"{os.getcwd()}/distortion_plots/vac_1_Cd_0.svg")
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_max_dist.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_max_distance(self):
        """Test plot_colorbar() function with metric=max_dist (default)"""
        fig = plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_fake_defect_name.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_fake_defect_name(self):
        """Test plot_colorbar() function with wrong defect name"""
        fig = plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="sub_1_In_on_Cd_1",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_displacement.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_colorbar_displacement(self):
        """Test plot_colorbar() function with metric=disp and num_nearest_neighbours=None"""
        fig = plotting.plot_colorbar(
            energies_dict=self.V_Cd_energies_dict,
            disp_dict=self.V_Cd_displacement_dict,
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=None,
            neighbour_atom="Te",
            metric="disp",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_maxdist_title_linecolor_label.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
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
            title="V$_{Cd}^{0}$",
            save_format="svg",
            save_plot=True,
            y_label="E (eV)",
        )
        self.assertTrue(
            os.path.exists(os.getcwd() + "/distortion_plots/vac_1_Cd_0.svg")
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{O}^{0}$_colors.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
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
            title="V$_{O}^{0}$",
            save_plot=True,
            save_format="png",
            colors=["green", "orange"],
        )
        self.assertTrue(
            os.path.exists(os.getcwd() + "/distortion_plots/vac_1_Cd_0.png")
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{O}^{0}$_notitle.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_without_saving(self):
        """Test plot_datasets() function testing several keywords:
        title = None, num_nearest_neighbours = None, neighbour_atom = None, save_plot = False
        and user specify style: markers, linestyles, markersize, linewidth"
        if_present_rm(os.getcwd() + "/distortion_plots/") # remove previous plots"""

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
        self.assertFalse(
            os.path.exists(os.getcwd() + "/distortion_plots/V$_{O}^{0}$.png")
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{O}^{0}$_not_enough_markers.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_not_enough_markers(self):
        """Test plot_datasets() function when user does not provide enough markers and linestyles"""
        fig = plotting.plot_datasets(
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
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_other_chargestates.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_from_other_charge_states(self):
        """Test plot_datasets() function when energy lowering distortions from other
        charge states have been tried"""
        fig = plotting.plot_datasets(
            datasets=[
                self.V_Cd_energies_dict_from_other_charge_states,
            ],
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{-2}$_only_rattled.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_only_rattled(self):
        """Test plot_datasets() function when the only distortion is 'Rattled'"""
        fig = plotting.plot_datasets(
            datasets=[
                self.V_Cd_m2_energies_dict,
            ],
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{-2}$_rattled_other_charge_states.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_rattled_and_dist_from_other_chargestates(self):
        """Test plot_datasets() function when the distortion is "Rattled"
        and distortions from other charge states have been tried"""
        fig = plotting.plot_datasets(
            datasets=[
                self.V_Cd_m2_energies_dict_from_other_charge_states,
            ],
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{-2}$_only_rattled_and_rattled_dist_from_other_charges_tates.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_datasets_only_rattled_and_rattled_dist_from_other_chargestates(self):
        """Test plot_datasets() function when one of the energy lowering distortions from other
        charge states is Rattled (i.e. `Rattled_from_0`)"""
        # Fake dataset
        datasets = [
            self.V_Cd_m2_energies_dict_from_other_charge_states,
        ]
        datasets[0]["distortions"].pop("-52.5%_from_0")
        datasets[0]["distortions"].update({"Rattled_from_0": -205.92311458})
        fig = plotting.plot_datasets(
            datasets=datasets,
            dataset_labels=["SnB: 2 Te"],
            defect_species="vac_1_Cd_0",
        )
        return fig

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
            defect_species="V$_{Cd}^{0}$",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )

    def test_plot_defect_fake_output_directories(self):
        """Test plot_defect() function when either directory `output_path` does not exist
        or directory `output_path/defect_species` does not exist"""
        self.assertRaises(
            FileNotFoundError,
            plotting.plot_defect,
            output_path="./fake_output_path",
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
        )
        self.assertRaises(
            FileNotFoundError,
            plotting.plot_defect,
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="fake_defect",
            energies_dict=self.V_Cd_energies_dict,
        )

    def test_plot_defect_missing_unperturbed_energy(self):
        with warnings.catch_warnings(record=True) as w:
            plotting.plot_defect(
                output_path=self.VASP_CDTE_DATA_DIR,
                defect_species="vac_1_Cd_0",
                energies_dict=self.organized_V_Cd_distortion_data_no_unperturbed,
            )
        self.assertEqual(len(w), 1)
        self.assertEqual(
            str(w[-1].message),
            "Unperturbed energy not present in energies_dict of vac_1_Cd_0! Skipping plot.",
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_plot_defect_add_colorbar_max_dist.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_add_colorbar(self):
        """Test plot_defect() function when add_colorbar = True"""
        fig = plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=True,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_plot_defect_without_colorbar.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_without_colorbar(self):
        """Test plot_defect() function when add_colorbar = False"""
        fig = plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=False,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}_^{0}$_include_site_num_in_name.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_include_site_num_in_name(self):
        """Test plot_defect() function when include_site_number_in_name = True"""
        fig = plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=False,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
            include_site_num_in_name=True,
            save_plot=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{0}$_plot_defect_without_title_units_meV.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defect_without_title_units_in_meV(self):
        """Test plot_defect() function when add_title = False and units = 'meV'"""
        fig = plotting.plot_defect(
            output_path=self.VASP_CDTE_DATA_DIR,
            defect_species="vac_1_Cd_0",
            energies_dict=self.V_Cd_energies_dict,
            add_colorbar=False,
            num_nearest_neighbours=2,
            neighbour_atom="Te",
            add_title=False,
            units="meV",
        )
        return fig

    # Most keywords of `plot_all_defects` have already been tested by tests of
    # `plot_defect`, `plot_datasets` and `plot_colorbar`. Now we test:
    # incorrect `output_path`, non-existent defect directory,
    # correct output of `plot_all_defects` (dict of figures), behaviour when distortion_metadata not found
    # or missing entry for a given charge state, and finally the keyword `min_e_diff`

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
        baseline_dir=f"{file_path}/remote_baseline_plots",
        filename="V$_{Cd}^{-2}$_only_rattled.png",
        style=f"{file_path}/../shakenbreak/shakenbreak.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_defects_output(self):
        """Test output of plot_all_defects() function. Test plot still geenrated when
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
        # No info on distortion_metadata.json for charge state -2, so its x label should be 'Bond Distortion Factor'
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
