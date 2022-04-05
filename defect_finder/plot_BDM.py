"""
Quick and _dirty_ collection of useful functions to plot BDM defect output & spot fancy bond_distortions going on

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from typing import Optional
import os
import sys
import json
from defect_finder.analyse_defects import *

pretty_colors = {"turquoise":"#6CD8AE", "dark_green":"#639483","ligth salmon":"#FFA17A","brownish":"#E46C51"}
dict_colors = {"turquoise":"#80DEB9",  "ligth salmon":"#F9966B", "blue_grey":"#b3d9ff", "grey":"#8585ad", "dark_green":"#4C787E",}
color_palette = sns.cubehelix_palette(start=0.45, rot=-1.1,light=.750, dark=0.35, reverse=True, as_cmap=False , n_colors=4)
colors = list(dict_colors.values())
color1, color2, color3, color4 = list(pretty_colors.values())
    
    
### Matplotlib Style formatting
wd = os.path.dirname(os.path.realpath(__file__))
plt.style.use(f"{wd}/BDM_mpl_style.mplstyle")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,10)))
    
def plot_all_defects(
    defects: dict, 
    base_path: str,                      
    add_colorbar: bool = False,
    metric: str = "max_dist",
    bdm_type : Optional[str] = 'BDM',
    plot_tag: Optional[bool] = True,
    max_energy_above_unperturbed: Optional[float] = 0.5,
    units: Optional[str] = "eV",
    ) -> dict:
    """
    Quick function to easily analyse a myriad of defects and focus on the ones undergoing deep relaxations.
    Args:
        defects (dict): 
            dictionary matching defect name to a list with its charge states. (e.g {"Int_Sb_1":[0,+1,+2]} )
        base_path (str): 
            path to the directory where your BDM defect output and BDM_metadata.txt file are
        add_colorbar (bool): 
            whether to add a colorbar indicating structural similarity between each structure and the unperturbed one.
            (default: False)
        metric (:obj:`str`):
            If add_colorbar is set to True, metric defines the criteria for structural comparison.
            Can choose between root-mean-squared displacement for all sites ('rms') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        bdm_type (str): 
            'BDM' Whether to analyse BDM normal results or champion (when trying the deep distoriton found for another charge state)
        plot_tag (bool):
            whether to plot or not. 
            (default: True) 
        max_energy_above_unperturbed (float):
            maximum E of metastable defects displayed on the graph
            (default: 0.5 eV) 
        units (str):
            Units for energy, either "eV" or "meV"
        """
    with open(f"{base_path}/BDM_metadata.json") as json_file:
        BDM_metadata = json.load(json_file)

    figures = {}
    for defect in defects:
        for charge in defects[defect]:
            defect_name = f"{defect}_{charge}" 
            #print(f"Analysing {defect_species}")
            file_energies = f"{base_path}{defect_name}/{bdm_type}/{defect_name}.txt"
            dict_energies, energy_diff, gs_distortion = sort_data(file_energies)
            
            # if a significant energy lowering distortion was found, and BDM was applied (i.e. not only rattle)
            # then further analyse this fancy defect
            if plot_tag and ("rattled" not in dict_energies['bond_distortions'].keys()) and float(energy_diff) < -0.1:
                num_neighbours = BDM_metadata["defects"][defect]["charges"][str(charge)]["number_neighbours"]  # get number of distorted neighbours                    
                neighbour_atoms = list(i[0] for i in BDM_metadata["defects"][defect]["charges"][str(charge)]["distorted_atoms"]) # get element distorted
                if all(element == neighbour_atoms[0] for element in neighbour_atoms):
                    neighbour_atom = neighbour_atoms[0]
                else:
                    neighbour_atom = "nn" # if different elements were distrorted, just use nearest neighbours (nn) for label
                f = plot_defect(
                    defect_name = defect_name,
                    charge = charge,
                    dict_energies = dict_energies,
                    base_path = base_path, 
                    neighbour_atom = neighbour_atom,
                    num_neighbours = num_neighbours,
                    add_colorbar = add_colorbar,
                    metric = metric,
                    units = units,
                    max_energy_above_unperturbed = max_energy_above_unperturbed,
                    )
                figures[f'{defect_name}_{charge}'] = f
    return figures

def plot_defect(
    defect_name: str,
    charge: int,
    dict_energies: dict,
    base_path: str = None, 
    neighbour_atom: str = None,
    num_neighbours: int = None,
    add_colorbar: bool = False,
    metric: str = "max_dist",
    max_energy_above_unperturbed: float = 0.5,
    include_site_num_in_name: bool = False,
    save_tag: bool = True,
    y_axis: str = "Energy (eV)",
    units: str = "eV",
    ) -> Figure:
    """
    Quick and nasty function to easily plot BDM results for a defect.
    args:
        defect_species (str):
            defect name e.g "Int_Sb_0",
        charge (int) : 
            defect charge state,
        energies_dict (dict):
            dict matching BDM distortion to its energy (eV),
        base_path (str): 
            str with path to the directory where your defect data is (to get structures and calculate RMS),
        neighbour_atom (str): 
            name of distorted neighbours,
        num_neighbours (int): 
            number of distorted neighbours 
        add_colorbar (bool): 
            whether to add a colorbar indicating structural similarity between each structure and the unperturbed one.
        metric (:obj:`str`):
            If add_colorbar is True, determines the criteria used for the structural comparison.
            Can choose between root-mean-squared displacement for all sites ('rms') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (float): 
            maximum E of metastable defects displayed on the graph (relative to Unperturbed). 
            Useful to avoid displaying really high energy bond_distortions
            (default: 0.5)
        include_site_num_in_name (bool): 
            Whether to include the site number (as generated by doped) in defect name.
            Useful for materials with many symmetry inequivalent sites
            (default=False)
        save_tag (bool): 
            whether to save the plot
            (default: True)
        y_axis (str): 
            label of y_axis,
        units (str):
            Units for energy, either "eV" or "meV"
        """
    
    if add_colorbar: # then get structures to compare their similarity
        assert os.isdir(base_path)
        defect_structs = get_structures(defect_name, base_path )
        dict_rms = calculate_struct_comparison(defect_structs, metric=metric) # calculate root mean
        # squared displacement and maximum displacement between paired sites
        if dict_rms : # if rms algorithms worked (sometimes strugles matching lattices)
            for key in list(dict_rms.keys()): 
                # remove any data point if its E is not in the energy dict (this may be due to relaxation not converged)
                if ( key not in dict_energies['bond_distortions'].keys() and key != "Unperturbed" ) or dict_rms[key] == "Not converged" or dict_rms[key] == None:
                    dict_rms.pop(key)
                    if key in dict_energies['bond_distortions'].keys(): # remove it from energy dict as well
                        dict_energies['bond_distortions'].pop(key)
        else:
            print("Rms algorithm strugled matching lattices. Will do normal plot")
            add_colorbar = False # if rms displacement couldn't be calculated (didn't match lattices), then do normal plot
            
    for key in list(dict_energies['bond_distortions'].keys()): # remove high E points
        if dict_energies['bond_distortions'][key]  - dict_energies['Unperturbed'] > max_energy_above_unperturbed:
            dict_energies['bond_distortions'].pop(key)
            if add_colorbar:
                dict_rms.pop(key)

    if charge > 0:
        charge = "+"+str(charge) # show positive charges with a + 
    defect_type = defect_name.split("_")[0] # vac, as or int
    if defect_type == "Int": # for interstials name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
        site_element =  defect_name.split("_")[1]
        site = defect_name.split("_")[2]
        if include_site_num_in_name :
            defect_name= f"{site_element}$_{{i_{site}}}^{{{charge}}}$" # by default include defect site in defect name for interstitials
        else:
            defect_name=f"{site_element}$_i^{{{charge}}}$"
    else:
        site = defect_name.split("_")[1] # number indicating defect site (as generated by doped)
        site_element = defect_name.split("_")[2] # element in defect site
    
    if include_site_num_in_name : # whether to include the site number in defect name
        if defect_type == "vac":
            defect_name=r"V$_{{{}_{}}}^{{{}}}$".format(site_element, site, charge) # double brackets to treat it literally (tex) and then add extra {} for python str formatting
        elif defect_type == "as":
            subs_element = defect_name.split("_")[4]
            defect_name=r"{}$_{{{}_{}}}^{{{}}}$".format(site_element,subs_element , site, charge)
        elif defect_type == "sub":
            subs_element = defect_name.split("_")[4]
            defect_name=r"{}$_{{{}}}^{{{}}}$".format(site_element,subs_element ,  charge)
    else:
        if defect_type == "vac":
            defect_name=r"V$_{{{}}}^{{{}}}$".format(site_element, charge) # double brackets to treat it literally (tex) and then add extra {} for python str formatting
        elif defect_type == "as":
            subs_element = defect_name.split("_")[4]
            defect_name=r"{}$_{{{}}}^{{{}}}$".format(site_element,subs_element , charge)
        elif defect_type == "sub":
            subs_element = defect_name.split("_")[4]
            defect_name=r"{}$_{{{}}}^{{{}}}$".format(site_element, subs_element ,  charge)
    
    if units == "meV" :
        y_axis = y_axis.replace("eV", "meV")
        if max_energy_above_unperturbed < 1 :
            max_energy_above_unperturbed *= 1000 
        for key in dict_energies["bond_distortions"].keys():
            dict_energies["bond_distortions"][key] =   dict_energies["bond_distortions"][key] *1000
        dict_energies["Unperturbed"] = dict_energies["Unperturbed"] * 1000
    
    if add_colorbar:
        max_displacement = max(list(dict_rms.values())) # get maximum displacement between paired sites
        f = plot_bdm_colorbar(
            dict_energies,
            dict_rms,
            number_neighbours = num_neighbours, 
            neighbour_atom = neighbour_atom, 
            dataset_labels = f"BDM: {num_neighbours} {neighbour_atom}", 
            metric = metric,
            defect_name= defect_name,
            title = defect_name, 
            save_tag = save_tag ,
            max_displacement = max_displacement,
            y_axis = y_axis,
            max_energy_above_unperturbed = max_energy_above_unperturbed,
            )
    else:
        f = plot_datasets(
            [dict_energies], 
            number_neighbours = num_neighbours, 
            neighbour_atom = neighbour_atom, 
            dataset_labels = [f"BDM: {num_neighbours} {neighbour_atom}"],
            defect_name= defect_name,
            title= defect_name, 
            save_tag = save_tag ,
            y_axis = y_axis,
            max_energy_above_unperturbed=max_energy_above_unperturbed,
            )
    return f

def plot_bdm_colorbar(
    dict_energies: dict, 
    dict_rms: dict , 
    number_neighbours: int,
    defect_name: str,
    neighbour_atom : str,  
    title: str=None,
    dataset_labels: str="RBDM", 
    metric: str = "max_dist",
    save_tag: bool =False,
    max_displacement: float = None,
    y_axis: str=None,
    max_energy_above_unperturbed : float=0.5,
    units: str = "eV",
    ) -> Figure:
    """
    Plot energy versus BDM distortion, adding a colorbar that shows structural similarity between different final configurations. 
    Args:
        dict_energies (dict): 
            dict maping distortion (format: float, e.g. 0.1) to final E (in eV)
        dict_rms (dict): 
            dict maping distortion to metric displacements (if metric=0) or maximum distance between paired sites (if metric=1)
        defect_name (str): 
            name of defect (e.g '$V_{Cd}^0$')
        title (str): 
            title of plot
            (default: None)
        number_neighbours: 
            number of distorted neighbours
        neighbour_atom (str): 
            name of distorted element (e.g Se)
        dataset_labels (str): 
            label for legend
            (default: RBDM)
        metric (:obj:`str`):
            Defines the criteria for structural comparison, used for the colorbar.
            Can choose between root-mean-squared displacement for all sites ('rms') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        save_tag (bool): 
            whether plot is saved or not
            (default: False)
        max_displacement (float): 
            max displacement displayed in colorbar (in A)
        y_axis (str):
            Label to y_axis 
            (deafult: "Energy (eV)")
        max_energy_above_unperturbed (float): 
            maximum E of metastable defects displayed on the graph (relative to Unperturbed)
            (deafult: 0.5)
        """
    f, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    if title:
        ax.set_title(title, size=20, pad=15)
    if not y_axis:
        y_axis = r"Energy (eV)"

    for key in list(dict_energies['bond_distortions'].keys()): #remove high E points
                    if dict_energies['bond_distortions'][key]  - dict_energies['Unperturbed'] > max_energy_above_unperturbed:
                        dict_energies['bond_distortions'].pop(key)
                        dict_rms.pop(key)
                        
    array_rms = np.array(np.array(list(dict_rms.values())))     
    
    colourmap = sns.cubehelix_palette(start=.65, rot=-.992075, dark=.2755, light=.7205, as_cmap=True) # sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)     #rot=-.952075
    # colormap extremes 
    vmin = round(min(array_rms), 1)
    vmax = round(max(array_rms), 1) 
    vmedium = round((vmin + vmax)/2, 1)
    norm = mpl.colors.Normalize(vmin= vmin, vmax= vmax, clip=False)  
    
    # all E relative to unperturbed one
    for key,i in dict_energies['bond_distortions'].items() :
            dict_energies['bond_distortions'][key] = i - dict_energies['Unperturbed']
    dict_energies['Unperturbed'] = 0.0
    
    ax.set_xlabel(f"BDM Distortion Factor (for {number_neighbours} {neighbour_atom} near {defect_name})", labelpad=10)
    ax.set_ylabel(y_axis, labelpad=10) # vasp_gam
    
    im = ax.scatter(
        dict_energies['bond_distortions'].keys(), dict_energies['bond_distortions'].values(),
        c=array_rms[:-1], 
        ls="-", 
        s=50, 
        marker="o",
        #c=color1, 
        cmap = colourmap,
        norm = norm, alpha = 1
    )
    ax.plot(
        dict_energies['bond_distortions'].keys(),
        dict_energies['bond_distortions'].values(),
        ls="-", markersize=1,
        marker="o",
        color=color1, 
        label = dataset_labels
    )
    color_unperturbed = colourmap(0) # get colour of unperturbed structure (corresponds to 0 as RMS is calculated with respect to this structure)
    ax.scatter(
        0,
        dict_energies['Unperturbed'],
        color = color_unperturbed, #dict_rms[-1],
        ls="None",
        s = 120,
        marker="d", # markersize=9,
        #cmap = colourmap,
        label = 'Unperturbed')
    
    # One decimal point if energy difference between max E and min E >0.4 eV, else 3
    range_energies = [i for i in dict_energies['bond_distortions'].values()]
    range_energies.append(dict_energies['Unperturbed'])
    if (max(range_energies)- min(range_energies)) > 0.4:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    elif (max(range_energies)- min(range_energies)) < 0.2:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.3f}'))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    plt.legend()
    
    # Colorbar formatting
    cbar = f.colorbar(im, ax=ax, boundaries = None, drawedges =False, aspect = 20, fraction =0.1, pad=0.08, shrink=0.8 )
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)
    if metric == "rms": cmap_label = 'RMS'
    elif metric == "max_dist": cmap_label = r"$d_{max}$ $(\AA)$"
    cbar.ax.set_title(cmap_label, size="medium", loc='center',ha='center', va='center', pad=20.5)    
    if vmin != vmax :
        cbar.set_ticks([vmin, vmedium,  vmax])
        cbar.set_ticklabels([vmin, vmedium,  vmax])
    else:
        cbar.set_ticks([vmedium])
        cbar.set_ticklabels([vmedium])
    
    # Save plot?
    if save_tag == True:
        wd  = os.getcwd()
        if not os.path.isdir(wd + "/plots/"):
            os.mkdir(wd + "/plots/")
        print(f"Plot saved to {wd}/plots/")
        plt.savefig(wd + "/plots/" + defect_name + ".svg", format="svg", transparent=True, bbox_inches='tight')
    plt.show()
    return f

def plot_datasets(
    datasets: list,  
    dataset_labels: list ,
    defect_name: str,
    neighbour_atom : str, 
    title: str = None,
    number_neighbours: int = None,                    
    max_energy_above_unperturbed: float=0.6,
    y_axis:str = r"Energy (eV)",
    markers: list = None,
    linestyles: list = None,
    colors: list = None,
    markersize: float = None,
    linewidth: float = None,
    save_tag: bool =False
    ) -> Figure:
    """"
    Quick and easy to plot several datatsets, relative energies (to Unperturbed) against BDM factor.
    Args:
        datasets (list): 
            list of dictionaries to plot (each dictionary matching BDM distortion to final energy)
        dataset_labels (list): 
            labels for each dataset.
        defect_name (str):
            name of defect that will appear in plot labels (e.g "$V_{Cd}^{0}$")
        title (str): 
            title of plot
        number_neighbours (int):
            Number of distorted defect nearest neighbours
        max_energy_above_unperturbed (float):
            maximum energy of BDM bond_distortions displayed in graph (relative to Unperturbed)
        y_axis (str):
            Label for y axis,
        markers (list):
            List of markers to use for each dataset (e.g ["o", "d"])
            (deafult: None)
        linestyles (list):
            List of line styles to use for each dataset (e.g ["-", "-."])
            (deafult: None)
        colors (list):
            List of color codes to use in plot.
            (default: None)
        save_tag (bool):
            Whether to save plot or not
            (default: False)
            """
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.linewidth=0.10
    if not colors:
        colors = list(dict_colors.values())
    if title:
        ax.set_title(title, size=20, pad=15)
    
    unperturbed_energies = {} # energies for unperturbed structure obtained with different methods

    # all E relative to unperturbed one
    for dataset_number, dataset in enumerate(datasets) : 
        
        for key, i in dataset['bond_distortions'].items() :
            dataset['bond_distortions'][key] = i - datasets[0]['Unperturbed'] # Energies relative to unperturbed E of dataset 1

        if dataset_number >= 1:
            dataset['Unperturbed'] = dataset['Unperturbed'] - datasets[0]['Unperturbed']
            unperturbed_energies[dataset_number] = dataset['Unperturbed']
        
        for key in list(dataset['bond_distortions'].keys()): # remove high E points
            if dataset['bond_distortions'][key]  > max_energy_above_unperturbed:
                dataset['bond_distortions'].pop(key)
        
        if number_neighbours and neighbour_atom:
            x_label = f"BDM Distortion Factor (for {number_neighbours} {neighbour_atom} near {defect_name})"
        else:
            x_label = f"BDM Distortion Factor"
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_axis, labelpad=10) 
        
        default_style_settings = {'marker': 'o', 
                                  'linestyle': 'solid',
                                  'linewidth': 1.0,
                                  'markersize': 6}
        for key, optional_style_settings in {'marker': markers, 'linestyle': linestyles, 'linewidth': linewidth,'markersize': markersize}.items():
            if optional_style_settings : # if set by user
                if type(optional_style_settings) == list :
                    default_style_settings[key] = optional_style_settings[dataset_number]
                else:
                    default_style_settings[key] = optional_style_settings
            
        ax.plot(
            dataset['bond_distortions'].keys(),
            dataset['bond_distortions'].values(),
            markersize = default_style_settings['markersize'],
            marker = default_style_settings['marker'],
            linestyle = default_style_settings['linestyle'],
            c = colors[dataset_number], 
            label = dataset_labels[dataset_number],
            linewidth = default_style_settings['linewidth']
        )
    
    datasets[0]['Unperturbed'] = 0.0 # plot unperturbed E of dataset 1, our 0 of E's
    
    for key, value in unperturbed_energies.items():
        if abs(value) > 0.1 :
            print("Energies for unperturbed structures obtained with different methods ({}) differ by {} eV. May wanna take a look ".format(dataset_labels[key], value))
            ax.plot(
            0,
            datasets[key]['Unperturbed'],
            ls="None",
            marker="d", 
            markersize=9,
            c=colors[key] )
    
    ax.plot( # plot our zero of Energies
        0,
        datasets[0]['Unperturbed'],
        ls="None",
        marker="d",
        markersize=9,
        c=colors[0], 
        label = 'Unperturbed'
        ) 
    
    # One decimal point if energy difference between max E and min E >0.4 eV
    range_energies = [i for i in datasets[0]['bond_distortions'].values()]
    range_energies.append(datasets[0]['Unperturbed'])
    if (max(range_energies)- min(range_energies)) > 0.4:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}')) # else 2 decimal points
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}')) # 1 decimal for distortion factor
    
    plt.legend()
    if save_tag == True:
        wd  = os.getcwd()
        if not os.path.isdir(wd + "/plots/"):
            os.mkdir(wd + "/plots/")
        print(f"Plot saved to {wd}/plots/")
        plt.savefig(wd + "/plots/" + defect_name + ".svg", format="svg", transparent=True, bbox_inches='tight')
    plt.show()
    return f

               