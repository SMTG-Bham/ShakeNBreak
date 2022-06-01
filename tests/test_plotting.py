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
        self.In_Cd_1_distortion_data = analysis._open_file(
            os.path.join(self.DATA_DIR, "CdTe_sub_1_In_on_Cd_1.txt")
        )  # note this was rattled with the old, non-Monte Carlo rattling (ASE"s atoms.rattle())
        self.organized_In_Cd_1_distortion_data = analysis._organize_data(
            self.In_Cd_1_distortion_data
        )
        self.V_Cd_displacement_dict = analysis.calculate_struct_comparison(
            defect_structures_dict=analysis.get_structures(
                defect_species='vac_1_Cd_0', 
                output_path=self.DATA_DIR,
            )
        )

    def tearDown(self):
        if_present_rm(f"{os.getcwd()}/distortion_plots")
    
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
        self.assertEqual(energies_dict["distortions"], {k: v*1000 for k,v in self.organized_V_Cd_distortion_data["distortions"].items()} )
        self.assertEqual(energies_dict["Unperturbed"], 100*self.organized_V_Cd_distortion_data["Unperturbed"])
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

# if __name__ == "__main__":
#     unittest.main()
