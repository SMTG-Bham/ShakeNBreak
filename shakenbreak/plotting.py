"""
Module containing functions to plot distorted defect relaxation outputs and identify 
energy-lowering distortions.
"""
import os
import warnings
import json
from typing import Optional
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from shakenbreak.analysis import (
    _sort_data,
    _read_distortion_metadata,
    get_structures,
    calculate_struct_comparison,
)

# Matplotlib Style formatting
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
plt.style.use(f"{MODULE_DIR}/shakenbreak.mplstyle")


# Helper functions for formatting plots
def _format_tick_labels(
    ax: mpl.axes.Axes,
    energy_range: list,
) -> mpl.axes.Axes:
    """
    Format axis labels of distortion plots and set limits of y axis. 
    For the y-axis (energies), show number with: \
     1 decimal point if energy range is higher than 0.4 eV, 
     3 decimal points if energy range is smaller than 0.1 eV, 
     2 decimal points otherwise.

    Args:
        ax (obj:`mpl.axes.Axes`): 
            matplotlib.axes.Axes of figure to format
        energy_range (:obj:`list`): 
            List of y (energy) values

    Returns:
        mpl.axes.Axes: Formatted axes
    """
    if (max(energy_range) - min(energy_range)) > 0.4:
        ax.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.1f}")  # 1 decimal point
        )
    elif (max(energy_range) - min(energy_range)) < 0.1:
        ax.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.3f}")  # 3 decimal points
        )
    else:
        ax.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.2f}")
        )  # else 2 decimal points
    ax.xaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter("{x:,.1f}")
    )  # 1 decimal for distortion factor (x axis)
    # Limits for y axis:
    ax.set_ylim(
        bottom=min(energy_range) - 0.1 * (max(energy_range) - min(energy_range)),
        top=max(energy_range) + 0.1 * (max(energy_range) - min(energy_range)),
    )  # add some extra space to avoid cutting off some data points
    # (e.g. using the energy range here in case units are meV)
    return ax


def _format_axis(
    ax: mpl.axes.Axes,
    defect_name: str,
    y_label: str,
    num_nearest_neighbours: Optional[int],
    neighbour_atom: Optional[str],
) -> mpl.axes.Axes:
    """
    Format and set axis labels of distortion plots, and set axis locators.

    Args:
        ax (:obj:`mpl.axes.Axes`): 
            current matplotlib.axes.Axes 
        defect_name (:obj:`str`): 
            name of defect (e.g. 'vac_1_Cd_0')
        y_label (:obj:`str`): 
            label for y axis
        num_nearest_neighbours (:obj:`int`): 
            number of distorted nearest neighbours
        neighbour_atom (:obj:`str`):  
            element symbol of distorted nearest neighbour

    Returns:
        mpl.axes.Axes: axes with formatted labels
    """
    if num_nearest_neighbours and neighbour_atom and defect_name:
        x_label = (
            f"Bond Distortion Factor (for {num_nearest_neighbours} {neighbour_atom} near"
            f" {defect_name})"
        )
    elif num_nearest_neighbours and defect_name:
        x_label = (
            f"Bond Distortion Factor (for {num_nearest_neighbours} NN near"
            f" {defect_name})"
        )
    else:
        x_label = "Bond Distortion Factor"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Format axis locators
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    return ax


def _format_defect_name(
    charge: int,
    defect_species: str,
    include_site_num_in_name: bool,
) -> str:
    """
    Format defect name. (i.e. from vac_1_Cd_0 to $V_{Cd}^{0}$)

    Args:
        charge (int): defect charge
        defect_species (str): name of defect without charge state (i.e. vac_1_Cd)
        include_site_num_in_name (bool): whether to include site number in name (i.e. $V_{Cd}^{0}$ or $V_{Cd,1}^{0}$)

    Returns:
        str: formatted defect name
    """
    if not isinstance(defect_species, str): # Check inputs 
        raise(TypeError(f"`defect_species` {defect_species} must be a string"))
    if not isinstance(charge, int):
        try:
            charge = int(charge)
        except ValueError:
            raise(ValueError(f"`charge` {charge} must be an integer!"))
    # Format defect name for title/axis labels
    if charge > 0:
        charge = "+" + str(charge)  # show positive charges with a + sign
    defect_type = defect_species.split("_")[0]  # vac, as or int
    if (
        defect_type == "Int"
    ):  # for interstitials, name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
        site_element = defect_species.split("_")[1]
        site = defect_species.split("_")[2]
        if include_site_num_in_name:
            defect_name = f"{site_element}$_{{i_{site}}}^{{{charge}}}$"  # by default include  # defect site in defect name for interstitials
        else:
            defect_name = f"{site_element}$_i^{{{charge}}}$"
    else:
        site = defect_species.split("_")[
            1
        ]  # number indicating defect site (from doped)
        site_element = defect_species.split("_")[2]  # element at defect site

    if include_site_num_in_name:  # whether to include the site number in defect name
        if defect_type == "vac":
            defect_name = f"V$_{{{site_element}_{site}}}^{{{charge}}}$"
            # double brackets to treat it literally (tex), then extra {} for python str formatting
        elif defect_type in ["as", "sub"]:
            subs_element = defect_species.split("_")[4]
            defect_name = f"{site_element}$_{{{subs_element}_{site}}}^{{{charge}}}$"
        elif defect_type != "Int":
            raise ValueError("Defect type not recognized. Please check spelling.")
    else:
        if defect_type == "vac":
            defect_name = f"V$_{{{site_element}}}^{{{charge}}}$"
        elif defect_type in ["as", "sub"]:
            subs_element = defect_species.split("_")[4]
            defect_name = f"{site_element}$_{{{subs_element}}}^{{{charge}}}$"
        elif defect_type != "Int":
            raise ValueError(f"Defect type {defect_type} not recognized. Please check spelling.")
    return defect_name


def _change_energy_units_to_meV(
    energies_dict: dict,
    max_energy_above_unperturbed: float,
    y_label: str,
) -> tuple:
    """
    Converts energy values from eV to meV and format y label accordingly.

    Args:
        energies_dict (dict): dictionary with energy values for all distortions
        max_energy_above_unperturbed (float): maximum energy value above unperturbed defect
        y_label (str): label for y axis

    Returns:
        tuple: (max_energy_above_unperturbed, energies_dict, y_label) with energy values in meV
    """
    if 'meV' not in y_label:
        y_label = y_label.replace("eV", "meV")
    if max_energy_above_unperturbed < 1:  # assume eV
        max_energy_above_unperturbed = (
            max_energy_above_unperturbed * 1000
        )  # convert to meV
    for key in energies_dict["distortions"].keys():  # convert to meV
        energies_dict["distortions"][key] = energies_dict["distortions"][key] * 1000
    energies_dict["Unperturbed"] = energies_dict["Unperturbed"] * 1000
    return energies_dict, max_energy_above_unperturbed, y_label


def _purge_data_dicts(
    disp_dict: dict,
    energies_dict: dict,
) -> tuple:
    """
    Purges dictionaries of displacements and energies so that they are consistent (i.e. contain
    data for same distortions).
    To achieve this, it removes: 
    - Any data point from disp_dict if its energy is not in the energy dict \
        (this may be due to relaxation not converged).
    - Any data point from energies_dict if its displacement is not in the disp_dict\
        (this might be due to the lattice matching algorithm failing).
    Args:
        disp_dict (dict): 
            dictionary with displacements (for each structure relative to Unperturbed),
            in the output format of `analysis.calculate_struct_comparison()`
        energies_dict (dict): 
            dictionary with final energies (for each structure relative to Unperturbed),
            in the output format of `analysis.get_energies()` or analysis.organize_data()

    Returns:
        (dict, dict): Consistent dictionaries of displacements and energies, containing data for
        same distortions.
    """
    for key in list(disp_dict.keys()):
        if (
            (key not in energies_dict["distortions"].keys() and key != "Unperturbed")
            or disp_dict[key] == "Not converged"
            or disp_dict[key] is None
        ):
            disp_dict.pop(key)
            if (
                key in energies_dict["distortions"].keys()
            ):  # remove it from energy dict as well
                energies_dict["distortions"].pop(key)
    return disp_dict, energies_dict


def _save_plot(
    fig: plt.Figure,
    defect_name: str,
    save_format: str,
) -> None:
    """
    Save plot in directory ´distortion_plots´

    Args:
        fig (:obj:`mpl.figure.Figure`): 
            mpl.figure.Figure object to save
        defect_name (:obj:`std`): 
            Defect name that will be used as file name.
        save_format (:obj:`str`):
            Format to save the plot as, given as string.
    """
    wd = os.getcwd()
    if not os.path.isdir(wd + "/distortion_plots/"):
        os.mkdir(wd + "/distortion_plots/")
    print(f"Plot saved to {wd}/distortion_plots/")
    fig.savefig(
        wd + "/distortion_plots/" + defect_name + f".{save_format}",
        format=save_format,
        transparent=True,
        bbox_inches="tight",
    )

def _get_line_colors(number_of_colors: int):
    """
    Get list of colors for plotting several lines.
    Args:
        number_of_colors (int): 
            Number of colors.
    """
    if  11 > number_of_colors > 1:  # If user didnt specify colors and more than one color needed, use deep color palette
        colors = sns.color_palette("deep", 10)
    elif number_of_colors > 11:  # otherwise use colormap
        colors = list(
            mpl.cm.get_cmap("viridis", number_of_colors + 1).colors
        )  # +1 to avoid yellow color (which is at the end of the colormap)
    else:
        colors = ["#59a590",] # Turquoise by default
    return colors

def _setup_colormap(disp_dict: dict):
    """
    Setup colormap to measure structural similarity between structures.
    Args:
        disp_dict (:obj: `dict`):
            dictionary mapping distortion key to structural similarity between
            the associated structure and the reference structure.
    Returns:
        colormap, vmin, vmedium, vmax, norm
    """
    array_disp = np.array(list(disp_dict.values()))
    colormap = sns.cubehelix_palette(
        start=0.65, rot=-0.992075, dark=0.2755, light=0.7205, as_cmap=True
    )
    # colormap extremes, mapped to min and max displacements
    vmin = round(min(array_disp), 1)
    vmax = round(max(array_disp), 1)
    vmedium = round((vmin + vmax) / 2, 1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    return colormap, vmin, vmedium, vmax, norm

def _format_colorbar(
        fig: mpl.figure.Figure,
        ax: mpl.axes.Axes,
        im: mpl.collections.PathCollection,
        metric: str,
        vmin: float,
        vmax: float,
        vmedium: float,
        ):
        """
        Format colorbar of plot
        Args:
            fig (:obj:`mpl.figure.Figure`):
                matplotlib.figure.Figure object
            ax (:obj:`mpl.axes.Axes`):
                current matplotlib.axes.Axes object
            im (:obj:`mpl.collections.PathCollection`)
            metric (:obj:`str`):
                metric to be plotted: "disp" or "max_dist" 
            vmin (:obj:`float`):
            vmax (:obj:`float`):
            vmedium (:obj:`float`):
        Returns:
            cbar (:obj:`mpl.colorbar.Colorbar`)
        """
        cbar = fig.colorbar(
            im,
            ax=ax,
            boundaries=None,
            drawedges=False,
            aspect=20,
            fraction=0.1,
            pad=0.09,
            shrink=0.8,
        )
        cbar.ax.tick_params(size=0)
        cbar.outline.set_visible(False)
        if metric == "disp":
            cmap_label = "$\Sigma$ Disp $(\AA)$"
        elif metric == "max_dist":
            cmap_label = r"$d_{max}$ $(\AA)$"
        cbar.ax.set_title(
            cmap_label, size="medium", loc="center", ha="center", va="center", pad=20.5
        )
        if vmin != vmax:
            cbar.set_ticks([vmin, vmedium, vmax])
            cbar.set_ticklabels([vmin, vmedium, vmax])
        else:
            cbar.set_ticks([vmedium])
            cbar.set_ticklabels([vmedium])
        return cbar
    
def plot_all_defects(
    defects_dict: dict,
    output_path: str = ".",
    add_colorbar: bool = False,
    metric: str = "max_dist",
    plot_tag: bool = True,
    max_energy_above_unperturbed: float = 0.5,
    units: str = "eV",
    min_e_diff: float = 0.05,
    line_color: Optional[str] = None,
    save_format: str = "svg",
) -> dict:
    """
    Convenience function to quickly analyse a range of defects and identify those which undergo
    energy-lowering distortions.

    Args:
        defects_dict (:obj:`dict`):
            Dictionary matching defect names to lists of their charge states. (e.g {"Int_Sb_1":[
            0,+1,+2]} etc)
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations and distortion_metadata.txt
            (Default: current directory)
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between each structure and
            the unperturbed one.
            (Default: False)
        metric (:obj:`str`):
            If add_colorbar is set to True, metric defines the criteria for structural comparison.
            Can choose between root-mean-squared displacement for all sites ('disp') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        plot_tag (:obj:`bool`):
            Whether to plot the results or not.
            (Default: True)
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in eV), relative to the unperturbed structure, to show on the plot.
            (Default: 0.5 eV)
        units (:obj:`str`):
            Units for energy, either "eV" or "meV".
        min_e_diff (:obj:`float`):
            Minimum energy difference (in eV) between the ground-state defect structure,
             relative to the `Unperturbed` structure, to consider it as having found a new
             energy-lowering distortion. Default is 0.05 eV.
        line_color (:obj:`str`):
            Color of the line connecting points.
            (Default: ShakeNBreak base style)
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'svg')
    Returns:
        Dictionary of {Defect Species (Name & Charge): Energy vs Distortion Plot}

    """
    if not os.path.isdir(output_path):  # check if output_path exists
        raise FileNotFoundError(f"Path {output_path} does not exist!")

    distortion_metadata = _read_distortion_metadata(output_path=output_path)

    figures = {}
    for defect in defects_dict:
        for charge in defects_dict[defect]:
            defect_species = f"{defect}_{charge}"
            # Parse energies
            if not os.path.isdir(f"{output_path}/{defect_species}"):
                warnings.warn(
                    f"Path {output_path}/{defect_species} does not exist! "
                    f"Skipping {defect_species}."
                )  # if defect directory doesn't exist, skip defect
                continue
            energies_file = f"{output_path}/{defect_species}/{defect_species}.txt"
            if not os.path.exists(energies_file):
                warnings.warn(
                    f"Path {energies_file} does not exist. Skipping {defect_species}."
                )  # skip defect
                continue
            energies_dict, energy_diff, gs_distortion = _sort_data(energies_file, verbose=False)

            if not energy_diff:  # if Unperturbed calc is not converged, warn user
                warnings.warn(f"Unperturbed calculation for {defect}_{charge} not converged! "
                              f"Skipping plot.")
                continue
            # If a significant energy lowering was found with bond distortions (not just rattling),
            # then further analyse this defect
            # TODO: Also do two-datapoint plot for defects that were just rattled
            if (
                plot_tag
                and ("Rattled" not in energies_dict["distortions"].keys())
                and abs(float(energy_diff)) > abs(min_e_diff)
            ):
                # Get number and element symbol of the distorted site(s)
                num_nearest_neighbours = distortion_metadata["defects"][defect][
                    "charges"
                ][str(charge)][
                    "num_nearest_neighbours"
                ]  # get number of distorted neighbours
                try:
                    neighbour_atoms = list(
                        i[1]  # element symbol
                        for i in distortion_metadata["defects"][defect]["charges"][
                            str(charge)
                        ]["distorted_atoms"]
                    )  # get element of the distorted site

                    if all(
                        element == neighbour_atoms[0] for element in neighbour_atoms
                    ):
                        neighbour_atom = neighbour_atoms[0]
                    else:
                        neighbour_atom = "NN"  # if different elements were distorted, just use nearest  # neighbours (NN) for label

                except TypeError:
                    neighbour_atom = (
                        "NN"  # if distorted_elements wasn't set, set label to "NN"
                    )

                fig= plot_defect(
                    defect_species=defect_species,
                    charge=charge,
                    energies_dict=energies_dict,
                    output_path=output_path,
                    neighbour_atom=neighbour_atom,
                    num_nearest_neighbours=num_nearest_neighbours,
                    add_colorbar=add_colorbar,
                    metric=metric,
                    units=units,
                    max_energy_above_unperturbed=max_energy_above_unperturbed,
                    line_color=line_color,
                    save_format=save_format,
                )
                figures[defect_species] = fig

            elif (
                    plot_tag
                    and ("Rattled" in energies_dict["distortions"].keys())
                    and abs(float(energy_diff)) > abs(min_e_diff)
            ):
                _sort_data(energies_file, verbose=True)  # Print energy-lowering info (remove
                # this once 'plot rattling' TODO above is added)

    return figures


def plot_defect(
    defect_species: str,
    charge: int,
    energies_dict: dict,
    output_path: Optional[str] = ".",
    neighbour_atom: Optional[str] = None,
    num_nearest_neighbours: Optional[int] = None,
    add_colorbar: Optional[bool] = False,
    metric: Optional[str] = "max_dist",
    max_energy_above_unperturbed: Optional[float] = 0.5,
    include_site_num_in_name: Optional[bool] = False,
    y_label: Optional[str] = "Energy (eV)",
    line_color: Optional[str] = None,
    units: Optional[str] = "eV",
    save_plot: Optional[bool] = True,
    save_format: Optional[str] = "svg",
) -> Figure:
    """
    Convenience function to plot energy vs distortion for a defect, to identify any energy-lowering
    distortions.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        charge (:obj:`int`):
            Defect charge state
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by `_organize_data()` or
            `analysis.get_energies()`)..
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to calculate structure
            comparisons)
            (Default: current directory)
        neighbour_atom (:obj:`str`):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between each structure and
            the unperturbed one.
        metric (:obj:`str`):
            If add_colorbar is True, determines the criteria used for the structural comparison.
            Can choose between root-mean-squared displacement for all sites ('disp') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed structure, to show on
            the plot.
            (Default: 0.5 eV)
        include_site_num_in_name (:obj:`bool`):
            Whether to include the site number (as generated by doped) in the defect name.
            Useful for materials with many symmetry-inequivalent sites
            (Default = False)
        save_plot (:obj:`bool`):
            Whether to save the plot as an SVG file.
            (Default: True)
        y_label (:obj:`str`):
            Y axis label (Default: 'Energy (eV)')
        units (:obj:`str`):
            Units for energy, either "eV" or "meV" (Default: "eV")
        line_color (:obj:`str`):
            Color of the line conneting points.
            (Default: ShakeNBreak base style)
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'svg')
    Returns:
        Energy vs distortion plot, as a mpl Figure object
    """
    # Check directories and input
    if not os.path.isdir(output_path):  # if output_path does not exist, raise error
        raise FileNotFoundError(
            f"Path {output_path} does not exist! Skipping {defect_species}."
        )
    if not os.path.isdir(
        f"{output_path}/{defect_species}"
    ):  # check if defect directory exists
        raise FileNotFoundError(
            f"Path {output_path}/{defect_species} does not exist! Skipping {defect_species}."
        )
    if not "Unperturbed" in energies_dict.keys():  # check if unperturbed energies exist
        warnings.warn(f"Unperturbed energy not present in energies_dict of {defect_species}! Skipping plot.")
        return None
    
    if not all(
        isinstance(energy, float)
        for energy in list(energies_dict["distortions"].values())
    ) and isinstance(energies_dict["Unperturbed"], float):  # check energies_dict values are floats
        try:
            energies_dict["distortions"] = {
                k: float(v) for k, v in energies_dict["distortions"].items()
            }
            energies_dict["Unperturbed"] = float(energies_dict["Unperturbed"])
        except ValueError:
            raise ValueError(
                f"Values of energies_dict are not floats! Skipping {defect_species}."
            )

    if add_colorbar:  # then get structures to compare their similarity
        try:
            defect_structs = get_structures(
                defect_species=defect_species, output_path=output_path
            )
            disp_dict = calculate_struct_comparison(
                defect_structs, metric=metric
            )  # calculate sum of atomic displacements and maximum displacement between paired sites
            if (
                disp_dict
            ):  # if struct_comparison algorithms worked (sometimes struggles matching
                # lattices)
                disp_dict, energies_dict = _purge_data_dicts(
                    disp_dict=disp_dict,
                    energies_dict=energies_dict,
                )  # make disp and energies dict consistent
                # by removing any data point if its energy is not in the energy dict and viceversa
            else:
                print(
                    "Structure comparison algorithm struggled matching lattices. Colorbar will not "
                    "be added to plot."
                )
                add_colorbar = False
        except FileNotFoundError:  # raised by analysis.get_structures() if defect_directory or distortion subdirectories do not exist
            warnings.warn(
                f"Could not find structures for {defect_species}. Colorbar will not be added to plot."
            )
            add_colorbar = False

    for key in list(energies_dict["distortions"].keys()):  # remove high energy points
        if (
            energies_dict["distortions"][key] - energies_dict["Unperturbed"]
            > max_energy_above_unperturbed
        ):
            energies_dict["distortions"].pop(key)
            if add_colorbar:
                disp_dict.pop(key)

    defect_name = _format_defect_name(
        charge=charge,
        defect_species=defect_species,
        include_site_num_in_name=include_site_num_in_name,
    )  # Format defect name for title and axis labels

    if units == "meV":
        (
            energies_dict,
            max_energy_above_unperturbed,
            y_label,
        ) = _change_energy_units_to_meV(
            energies_dict=energies_dict,
            max_energy_above_unperturbed=max_energy_above_unperturbed,
            y_label=y_label,
        )  # convert energy units from eV to meV, and update y label

    if add_colorbar:
       fig= plot_colorbar(
            energies_dict=energies_dict,
            disp_dict=disp_dict,
            defect_name=defect_name,
            num_nearest_neighbours=num_nearest_neighbours,
            neighbour_atom=neighbour_atom,
            title=defect_name,
            dataset_label=f"ShakeNBreak: {num_nearest_neighbours} {neighbour_atom}"
            if num_nearest_neighbours != None
            else f"ShakeNBreak: {neighbour_atom}",
            metric=metric,
            save_plot=save_plot,
            y_label=y_label,
            max_energy_above_unperturbed=max_energy_above_unperturbed,
            line_color=line_color,
            save_format=save_format,
        )
    else:
       fig= plot_datasets(
            datasets=[energies_dict],
            defect_name=defect_name,
            num_nearest_neighbours=num_nearest_neighbours,
            neighbour_atom=neighbour_atom,
            title=defect_name,
            dataset_labels=[f"ShakeNBreak: {num_nearest_neighbours} {neighbour_atom}"]
            if num_nearest_neighbours != None
            else [f"ShakeNBreak: {neighbour_atom}"],
            save_plot=save_plot,
            y_label=y_label,
            max_energy_above_unperturbed=max_energy_above_unperturbed,
            save_format=save_format,
        )
    return fig


def plot_colorbar(
    energies_dict: dict,
    disp_dict: dict,
    defect_name: str,
    num_nearest_neighbours: int = None,
    neighbour_atom: str = "NN",
    title: Optional[str] = None,
    dataset_label: str = "SnB",
    metric: Optional[str] = "max_dist",
    max_energy_above_unperturbed: Optional[float] = 0.5,
    save_plot: Optional[bool] = False,
    y_label: Optional[str] = "Energy (eV)",
    line_color: Optional[str] = None,
    save_format: Optional[str] = "svg",
) -> Figure:
    """
    Plot energy versus bond distortion, adding a colorbar to show structural similarity between
    different final configurations.

    Args:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by `analysis.get_energies()`
            or `analysis._organize_data()`.
        disp_dict (:obj:`dict`):
            Dictionary matching bond distortions to structure comparison metric (metric = 'disp' or
            'max_dist'), as produced by `analysis.calculate_struct_comparison()`.
        defect_name (:obj:`str`):
            Specific defect name that will appear in plot labels and file names (e.g '$V_{Cd}^0$')
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
            (Default: None)
        neighbour_atom (:obj:`str`, optional):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
            (Default: "NN")
        title (:obj:`str`, optional):
            Plot title
            (Default: None)
        dataset_label (:obj:`str`):
            Label for plot legend
            (Default: 'SnB')
        metric (:obj:`str`):
            Defines the criteria for structural comparison, used for the colorbar.
            Can choose between root-mean-squared displacement for all sites ('disp') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed structure, to show on
            the plot.
            (Default: 0.5 eV)
        line_color (:obj:`str`):
            Color of the line conneting points.
            (Default: ShakeNBreak base style)
        save_plot (:obj:`bool`):
            Whether to save the plot as an SVG file.
            (Default: True)
        y_label (:obj:`str`):
            Y axis label (Default: 'Energy (eV)')
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'svg')

    Returns:
        Energy vs distortion plot with colorbar for structural similarity, as a matplotlib.figure.Figure
        object
    """
    fig, ax = plt.subplots(1,1)

    # Title and format axis labels and locators
    if title:
        ax.set_title(title)
    ax = _format_axis(
        ax=ax,
        y_label=y_label,
        defect_name=defect_name,
        num_nearest_neighbours=num_nearest_neighbours,
        neighbour_atom=neighbour_atom,
    )

    # Remove high energy points
    for key in list(energies_dict["distortions"].keys()):
        if (
            energies_dict["distortions"][key] - energies_dict["Unperturbed"]
            > max_energy_above_unperturbed
        ):
            energies_dict["distortions"].pop(key)
            disp_dict.pop(key)

    # Setting line color and colorbar
    if not line_color:
        line_color = "#59a590"  # By default turquoise
    colormap, vmin, vmedium, vmax, norm = _setup_colormap(disp_dict) # colormap to measure structural similarity
    
    # all energies relative to unperturbed one
    for key, i in energies_dict["distortions"].items():
        energies_dict["distortions"][key] = i - energies_dict["Unperturbed"]
    energies_dict["Unperturbed"] = 0.0

    # Plotting
    # store indices of imported structures ("X%_from_Y") to plot differently later comparison
    imported_indices = []
    for i, entry in enumerate(energies_dict["distortions"].keys()):
        if isinstance(entry, str) and "_from_" in entry:
            imported_indices.append(i)
    
    # sort displacements in same order as distortions and energies, for proper color mapping         
    sorted_disp = [disp_dict[k] for k in energies_dict["distortions"].keys() if k in disp_dict.keys()] 
    # reformat any "X%_from_Y" distortions to corresponding (X) distortion factor
    keys = [
        float(entry.split("%")[0]) / 100
        if isinstance(entry, str) and "_from_" in entry
        else entry
        for entry in energies_dict["distortions"].keys()
    ]
    # sort keys and values
    sorted_distortions, sorted_energies, sorted_disp = zip(
        *sorted(zip(keys, energies_dict["distortions"].values(), sorted_disp))
    )
    
    im = ax.scatter(
        sorted_distortions,
        sorted_energies,
        c=sorted_disp,
        ls="-",
        s=50,
        marker="o",
        cmap=colormap,
        norm=norm,
        alpha=1,
    )
    ax.plot(
        sorted_distortions,
        sorted_energies,
        ls="-",
        markersize=1,
        marker="o",
        color=line_color,
        label=dataset_label,
    )
    for i in imported_indices:  # datapoints from other charge states
        ax.scatter(
            np.array(keys)[i],
            list(energies_dict["distortions"].values())[i],
            c=sorted_disp,
            edgecolors="k",
            ls="-",
            s=50,
            marker="s",
            zorder=10,  # make sure it's on top of the other points
            cmap=colormap,
            norm=norm,
            alpha=1,
            label=f"From "
                  f"{list(energies_dict['distortions'].keys())[i].split('_')[-1]} charge state",
        )
    unperturbed_color = colormap(
        0
    )  # get color of unperturbed structure (corresponding to 0 as disp is calculated with respect
    # to this structure)
    ax.scatter(
        0,
        energies_dict["Unperturbed"],
        color=unperturbed_color,
        ls="None",
        s=120,
        marker="d",
        label="Unperturbed",
    ) # plot reference energy

    # Formatting of tick labels.
    # For yaxis (i.e. energies): 1 decimal point if deltaE = (max E - min E) > 0.4 eV, 2 if deltaE > 0.1 eV, otherwise 3.
    ax = _format_tick_labels(ax=ax, energy_range=list(energies_dict["distortions"].values()) + [energies_dict["Unperturbed"],] )
    
    plt.legend(frameon=True)

    cbar = _format_colorbar(fig=fig, ax=ax, im=im, metric=metric, vmin=vmin, vmax=vmax, vmedium=vmedium) # Colorbar formatting
    
    # Save plot?
    if save_plot:
        _save_plot(
            fig=fig,
            defect_name=defect_name,
            save_format=save_format,
        )
    return fig


def plot_datasets(
    datasets: list,
    dataset_labels: list,
    defect_name: str,
    title: Optional[str] = None,
    neighbour_atom: Optional[str] = None,
    num_nearest_neighbours: Optional[int] = None,
    max_energy_above_unperturbed: Optional[float] = 0.6,
    y_label: str = r"Energy (eV)",
    markers: Optional[list] = None,
    linestyles: Optional[list] = None,
    colors: Optional[list] = None,
    markersize: Optional[float] = None,
    linewidth: Optional[float] = None,
    save_plot: Optional[bool] = False,
    save_format: Optional[str] = "svg",
) -> Figure:
    """
    Generate energy versus bond distortion plots for multiple datasets. 

    Args:
        datasets (:obj:`list`):
            List of {distortion: energy} dictionaries to plot (each dictionary matching
            distortion to final energy (eV), as produced by `analysis._organize_data()` or
            `analysis.get_energies()`)
        dataset_labels (:obj:`list`):
            Labels for each dataset plot legend.
        defect_name (:obj:`str`):
            Specific defect name that will appear in plot labels and file names (e.g '$V_{Cd}^0$')
        neighbour_atom (:obj:`str`):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
        title (:obj:`str`, optional):
            Plot title
            (Default: None)
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed structure, to show on
            the plot.
            (Default: 0.5 eV)
        y_label (:obj:`str`):
            Y axis label (Default: 'Energy (eV)')
        colors (:obj:`list`):
            List of color codes to use for each dataset (e.g ["C1", "C2"])
            (Default: None)
        markers (:obj:`list`):
            List of markers to use for each dataset (e.g ["o", "d"])
            (Default: None)
        linestyles (:obj:`list`):
            List of line styles to use for each dataset (e.g ["-", "-."])
            (Default: None)
        markersize (:obj:`float`):
            Marker size to use for plots (single value, or list of values for each dataset)
            (Default: None)
        linewidth (:obj:`float`):
            Linewidth to use for plots (single value, or list of values for each dataset)
            (Default: None)
        save_plot (:obj:`bool`):
            Whether to save the plots.
            (Default: True)
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'svg')
    Returns:
        Energy vs distortion plot for multiple datasets, as a matplotlib.figure.Figure object
    """
    # Validate input
    if len(datasets) != len(dataset_labels):
        raise ValueError(
           f"Number of datasets and labels must match! You gave me {len(datasets)} datasets and"
           f" {len(dataset_labels)} labels."
    )
    
    fig, ax = plt.subplots(1,1)
    # Line colors
    if not colors:
        colors = _get_line_colors(number_of_colors=len(datasets)) # get list of colors to use for each dataset
    elif len(colors) < len(datasets):
        warnings.warn(f"Insufficient colors provided for {len(datasets)} datasets. Using default colors.")
        colors = _get_line_colors(number_of_colors=len(datasets))
    # Title and labels of axis
    if title:
        ax.set_title(title)
    ax = _format_axis(
        ax=ax,
        y_label=y_label,
        defect_name=defect_name,
        num_nearest_neighbours=num_nearest_neighbours,
        neighbour_atom=neighbour_atom,
    )

    # Plot data points for each dataset
    unperturbed_energies = (
        {}
    )  # energies for unperturbed structure obtained with different methods

    # all energies relative to the unperturbed energy of first dataset
    for dataset_number, dataset in enumerate(datasets):

        for key, energy in dataset["distortions"].items():
            dataset["distortions"][key] = (
                energy - datasets[0]["Unperturbed"]
            )  # Energies relative to unperturbed E of dataset 1

        if dataset_number >= 1:
            dataset["Unperturbed"] = dataset["Unperturbed"] - datasets[0]["Unperturbed"]
            unperturbed_energies[dataset_number] = dataset["Unperturbed"]

        for key in list(dataset["distortions"].keys()):  # remove high E points
            if dataset["distortions"][key] > max_energy_above_unperturbed:
                dataset["distortions"].pop(key)

        default_style_settings = {
            "marker": "o",
            "linestyle": "solid",
            "linewidth": 1.0,
            "markersize": 6,
        }
        for key, optional_style_settings in {
            "marker": markers,
            "linestyle": linestyles,
            "linewidth": linewidth,
            "markersize": markersize,
        }.items():
            if optional_style_settings:  # if set by user
                if isinstance(optional_style_settings, list):
                    try:
                        default_style_settings[key] = optional_style_settings[
                            dataset_number
                        ]
                    except IndexError:
                        default_style_settings[key] = optional_style_settings[
                            0
                        ]  # in case not enough for each dataset
                else:
                    default_style_settings[key] = optional_style_settings

        # Plotting
        # store indices of imported structures ("X%_from_Y") to plot differently later
        # comparison
        imported_indices = []
        for i, entry in enumerate(dataset["distortions"].keys()):
            if isinstance(entry, str) and "_from_" in entry:
                imported_indices.append(i)

        # reformat any "X%_from_Y" distortions to corresponding (X) distortion factor
        keys = [
            float(entry.split("%")[0]) / 100
            if isinstance(entry, str) and "_from_" in entry
            else entry
            for entry in dataset["distortions"].keys()
        ]
        # sort keys and values
        sorted_distortions, sorted_energies = zip(
            *sorted(zip(keys, dataset["distortions"].values()))
        )
        ax.plot(
            sorted_distortions,
            sorted_energies,
            c=colors[dataset_number],
            markersize=default_style_settings["markersize"],
            marker=default_style_settings["marker"],
            linestyle=default_style_settings["linestyle"],
            label=dataset_labels[dataset_number],
            linewidth=default_style_settings["linewidth"],
        )
        for i in imported_indices:
            ax.scatter(
                np.array(keys)[i],
                list(dataset["distortions"].values())[i],
                c=colors[dataset_number],
                edgecolors="k",
                ls="-",
                s=50,
                zorder=10,  # make sure it's on top of the other lines
                marker="s",
                alpha=1,
                label=f"From "
                      f"{list(dataset['distortions'].keys())[i].split('_')[-1]} charge state",
            )

    datasets[0][
        "Unperturbed"
    ] = 0.0  # unperturbed energy of first dataset (our reference energy)

    # Plot Unperturbed point for every dataset, relative to the unperturbed energy of first dataset
    for key, value in unperturbed_energies.items():
        if abs(value) > 0.1: # Only plot if different energy from the reference Unperturbed
            print(
                f"Energies for unperturbed structures obtained with different methods "
                f"({dataset_labels[key]}) differ by {value:.2f}. If testing different "
                "magnetic states (FM, AFM) this is normal, otherwise you may want to check this!"
            )
            ax.plot(
                0,
                datasets[key]["Unperturbed"],
                ls="None",
                marker="d",
                markersize=9,
                c=colors[key],
            )

    ax.plot(  # plot our reference energy
        0,
        datasets[0]["Unperturbed"],
        ls="None",
        marker="d",
        markersize=9,
        c=colors[0],
        label="Unperturbed",
    )

    # Format tick labels:
    # For yaxis, 1 decimal point if energy difference between max E and min E > 0.4 eV, 3 if E < 0.1 eV, 2 otherwise
    ax = _format_tick_labels(
        ax=ax, 
        energy_range=list(datasets[0]["distortions"].values()) + [datasets[0]["Unperturbed"],] 
        )

    ax.legend(frameon=True)  # show legend

    if save_plot: # Save plot?
        _save_plot(
            fig=fig,
            defect_name=defect_name,
            save_format=save_format,
        )
    return fig
