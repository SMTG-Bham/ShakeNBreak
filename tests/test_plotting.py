import unittest
import os
from unittest.mock import patch
import shutil
import warnings
from copy import deepcopy 

import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure, Element
from shakenbreak import analysis
from shakenbreak import plotting

import matplotlib as mpl
import matplotlib.pyplot as plt


def if_present_rm(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class AnalyseDefectsTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        self.V_Cd_distortion_data = analysis._open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25.txt")
        )
        self.organized_V_Cd_distortion_data = analysis._organize_data(
            self.V_Cd_distortion_data
        )
        self.V_Cd_distortion_data_no_unperturbed = analysis._open_file(
            os.path.join(self.DATA_DIR, "CdTe_vac_1_Cd_0_stdev_0.25_no_unperturbed.txt")
        )
        self.organized_V_Cd_distortion_data_no_unperturbed = analysis._organize_data(
            self.V_Cd_distortion_data_no_unperturbed
        )
        self.V_Cd_energies_dict = analysis.get_energies(
            defect_species='vac_1_Cd_0', 
            output_path=self.DATA_DIR,
        )
        self.V_Cd_displacement_dict = analysis.calculate_struct_comparison(
            defect_structures_dict=analysis.get_structures(
                defect_species='vac_1_Cd_0', 
                output_path=self.DATA_DIR,
            )
        )

    def tearDown(self):
        if_present_rm(f"{os.getcwd()}/distortion_plots")
    
    def test_format_axis(self):
        "Test _format_axis() function"
        # Test standard behaviour: labels and ticks
        fig, ax = plt.subplots(1,1)
        ax.plot(
            self.V_Cd_energies_dict["distortions"].keys(), 
            self.V_Cd_energies_dict["distortions"].values()
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te"
        )
        self.assertEqual(formatted_ax.yaxis.get_label().get_text(), "Energy (eV)")
        self.assertEqual(formatted_ax.xaxis.get_label().get_text(), "Bond Distortion Factor (for 2 Te near V$_{Cd}^{0}$)")
        self.assertEqual(len(formatted_ax.yaxis.get_ticklabels()), 6+2) # +2 bc MaxNLocator adds ticks
        # beyond axis limits for autoscaling reasons
        # self.assertTrue([float(tick.get_text()) % 0.3 == 0.0 for tick in formatted_ax.xaxis.get_ticklabels()]) # x ticks should be multiples of 0.3
        print(formatted_ax.xaxis.get_ticklabels())
        # check x label if no nearest neighbour info
        ax.plot(
            self.V_Cd_energies_dict["distortions"].keys(), 
            self.V_Cd_energies_dict["distortions"].values()
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom=None,
        )
        self.assertEqual(formatted_ax.xaxis.get_label().get_text(), "Bond Distortion Factor (for 2 NN near V$_{Cd}^{0}$)")
        # check x label if no defect name
        ax.plot(
            self.V_Cd_energies_dict["distortions"].keys(), 
            self.V_Cd_energies_dict["distortions"].values()
        )
        formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name=None,
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te",
        )
        self.assertEqual(formatted_ax.xaxis.get_label().get_text(), "Bond Distortion Factor")

    def test_format_tick_labels(self):
        "Test format_tick_labels() function."
        # Test standard behaviour
        fig, ax = plt.subplots(1,1)
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()), 
            list(self.V_Cd_energies_dict["distortions"].values())
        )
        semi_formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te"
        )
        semi_formatted_ax.set_ylim(-0.2, 0.4) # set incorrect y limits (-0.2 rather than ~-0.8)
        semi_formatted_ax.set_xticks(ticks=[-0.201, -0.101, 0.000,], labels=[-0.201, -0.101, 0.000,], minor=False) # set incorrect x ticks
        semi_formatted_ax.set_yticks(ticks=[-0.804, -0.502, 0.000,], labels=[-0.804, -0.502, 0.000,], minor=False) # set incorrect y ticks
        formatted_ax = plotting._format_tick_labels(
            ax=deepcopy(semi_formatted_ax),
            energy_range=list(self.V_Cd_energies_dict["distortions"].values()) + [self.V_Cd_energies_dict["Unperturbed"],],
        )
        # Check limits of y axis are correct (shouldnt cut off any data point given -
        # note that purging of high energy points has been done before.) 
        self.assertTrue(
            formatted_ax.get_ylim()[0] < min([self.V_Cd_energies_dict["Unperturbed"],] + list(self.V_Cd_energies_dict["distortions"].values())),
        )   
        self.assertTrue(
            formatted_ax.get_ylim()[1] > max([self.V_Cd_energies_dict["Unperturbed"],] + list(self.V_Cd_energies_dict["distortions"].values())),
        )
        # Check x tick labels have 1 decimal place
        self.assertTrue( 
            len(formatted_ax.xaxis.get_major_formatter().format_data(0.11111).split('.')[1]) == 1
        )
        # Check y tick labels have 1 decimal place if energy range is > 0.4 eV
        self.assertTrue( 
            len(formatted_ax.yaxis.get_major_formatter().format_data(0.11111).split('.')[1]) == 1
        )
        
        # Test y tick labels have 3 decimal places if energy range is < 0.1 eV
        fig, ax = plt.subplots(1,1)
        energies = [np.random.uniform(-0.099, 0) for i in range(len(self.V_Cd_energies_dict["distortions"].keys()))]
        ax.plot(
            list(self.V_Cd_energies_dict["distortions"].keys()), 
            energies,
        )
        semi_formatted_ax = plotting._format_axis(
            ax=ax,
            defect_name="V$_{Cd}^{0}$",
            y_label="Energy (eV)",
            num_nearest_neighbours=2,
            neighbour_atom="Te"
        )
        semi_formatted_ax.set_yticks(ticks=[-0.08041, -0.05011, 0.0001,], labels=[-0.08041, -0.05011, 0.0001], minor=False) # set incorrect y ticks
        formatted_ax = plotting._format_tick_labels(
            ax=deepcopy(semi_formatted_ax),
            energy_range=energies,
        )
        self.assertTrue( 
            len(formatted_ax.yaxis.get_major_formatter().format_data(0.11111).split('.')[1]) == 3
        )
        
    def test_format_defect_name(self):
        """Test _format_defect_name() function."""
        # test standard behaviour
        formatted_name = plotting._format_defect_name(
            charge = 0,
            defect_species = "vac_1_Cd",
            include_site_num_in_name = False,
        )
        self.assertEqual(formatted_name, "V$_{Cd}^{0}$")
        # test with site number included
        formatted_name = plotting._format_defect_name(
            charge = 0,
            defect_species = "vac_1_Cd",
            include_site_num_in_name = True,
        )
        self.assertEqual(formatted_name, "V$_{Cd_1}^{0}$")
        # test interstitial case
        formatted_name = plotting._format_defect_name(
            charge = 0,
            defect_species = "Int_Cd_1",
            include_site_num_in_name = True,
            )
        self.assertEqual(formatted_name, "Cd$_{i_1}^{0}$")
        # check exceptions raised: invalid charge or defect_species
        self.assertRaises(
            ValueError,
            plotting._format_defect_name,
            charge = "a",
            defect_species = "vac_1_Cd",
            include_site_num_in_name = True,
        )
        self.assertRaises(
            TypeError,
            plotting._format_defect_name,
            charge = 0,
            defect_species = 2,
            include_site_num_in_name = True,
        )
        # check invalid defect type
        self.assertRaises(
            ValueError,
            plotting._format_defect_name,
            charge = 0,
            defect_species = "kk_Cd_1",
            include_site_num_in_name = True,
        )
        
    def test_change_energy_units_to_meV(self):
        """Test _change_energy_units_to_meV() function."""
        # Test standard behaviour
        energies_dict, max_energy_above_unperturbed, y_label = plotting._change_energy_units_to_meV(
            energies_dict=deepcopy(self.organized_V_Cd_distortion_data),
            max_energy_above_unperturbed=0.2,
            y_label="Energy (eV)",
        )
        self.assertEqual(energies_dict["distortions"], {k: 1000*v for k,v in self.organized_V_Cd_distortion_data["distortions"].items()} )
        self.assertEqual(energies_dict["Unperturbed"], 1000*self.organized_V_Cd_distortion_data["Unperturbed"])
        self.assertEqual(max_energy_above_unperturbed, 0.2 * 1000)
        self.assertEqual(y_label, "Energy (meV)")
        
    def test_purge_data_dict(self):
        """Test _purge_data_dict() function."""
        # Test if dictionaries have same data points when displacement dict is incomplete
        disp_dict=deepcopy(self.V_Cd_displacement_dict) 
        disp_dict.pop(-0.6) # Missing data point
        disp_dict, energies_dict = plotting._purge_data_dicts(
            energies_dict=deepcopy(self.organized_V_Cd_distortion_data),
            disp_dict=deepcopy(disp_dict)
        )
        self.assertEqual(
            set(list(disp_dict.keys())) - set(list(energies_dict['distortions'].keys())), {'Unperturbed'}
            ) # only difference should be Unperturbed
        # Test behaviour when energy dict is incomplete
        energies_dict=deepcopy(self.organized_V_Cd_distortion_data)
        energies_dict["distortions"].pop(-0.6)
        disp_dict, energies_dict = plotting._purge_data_dicts(
            energies_dict=deepcopy(self.organized_V_Cd_distortion_data),
            disp_dict=deepcopy(disp_dict)
        )
        self.assertEqual(
            set(list(disp_dict.keys())) - set(list(energies_dict['distortions'].keys())), {'Unperturbed'}
            ) # only difference should be Unperturbed

    def test_save_plot(self):
        "Test _save_plot() function"
        fig, ax = plt.subplots(1,1)
        if_present_rm(f"{os.getcwd()}/distortion_plots/vac_1_Cd_0.svg")
        plotting._save_plot(
            fig=fig,
            defect_name='vac_1_Cd_0',
            save_format='svg'
        )
        self.assertTrue(os.path.exists(f"{os.getcwd()}/distortion_plots/vac_1_Cd_0.svg"))
    
    # def test_plot_datasets(self):
    #     "Test _plot_datasets() function"
        
# if __name__ == "__main__":
#     unittest.main()
