"""
Module containing functions to plot distorted defect relaxation outputs and identify 
energy-lowering distortions.
"""
import os
import json
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from defect_finder.analyse_defects import (
    sort_data,
    get_structures,
    calculate_struct_comparison,
)

pretty_colors = {
    "turquoise": "#6CD8AE",
    "dark_green": "#639483",
    "light salmon": "#FFA17A",
    "brownish": "#E46C51",
}
colors_dict = {
    "turquoise": "#80DEB9",
    "light salmon": "#F9966B",
    "blue_grey": "#b3d9ff",
    "grey": "#8585ad",
    "dark_green": "#4C787E",
}
color_palette = sns.cubehelix_palette(
    start=0.45,
    rot=-1.1,
    light=0.750,
    dark=0.35,
    reverse=True,
    as_cmap=False,
    n_colors=4,
)


## Matplotlib Style formatting
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
plt.style.use(f"{MODULE_DIR}/defect_finder_style.mplstyle")
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.viridis(np.linspace(0, 1, 10))
)


def plot_all_defects(
    defects_dict: dict,
    base_path: str,
    add_colorbar: bool = False,
    metric: str = "max_dist",
    distortion_type: str = "BDM",
    plot_tag: bool = True,
    max_energy_above_unperturbed: float = 0.5,
    units: str = "eV",
) -> dict:
    """
    Convenience function to quickly analyse a range of defects and identify those which undergo
    energy-lowering distortions.

    Args:
        defects_dict (:obj:`dict`):
            Dictionary matching defect names to lists of their charge states. (e.g {"Int_Sb_1":[
            0,+1,+2]} etc)
        base_path (:obj:`str`):
            Path to directory with your distorted defect calculations and distortion_metadata.txt
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between each structure and
            the unperturbed one.
            (Default: False)
        metric (:obj:`str`):
            If add_colorbar is set to True, metric defines the criteria for structural comparison.
            Can choose between root-mean-squared displacement for all sites ('rms') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        distortion_type (:obj:`str`) :
            Type of distortion method used.
            Either 'BDM' (bond distortion method (standard)) or 'champion'. The option 'champion'
            is used when relaxing a defect from the relaxed structure(s) found for other charge
            states of that defect – in which case only the unperturbed and rattled configurations of
            the relaxed other-charge defect structure(s) are calculated.
            (Default: 'BDM')
        plot_tag (:obj:`bool`):
            Whether to plot the results or not.
            (Default: True)
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in eV), relative to the unperturbed structure, to show on the plot.
            (Default: 0.5 eV)
        units (:obj:`str`):
            Units for energy, either "eV" or "meV".

    Returns:
        Dictionary of {Defect Species (Name & Charge): Energy vs Distortion Plot}

    """
    with open(f"{base_path}/distortion_metadata.json") as json_file:
        distortion_metadata = json.load(json_file)

    figures = {}
    for defect in defects_dict:
        for charge in defects_dict[defect]:
            defect_species = f"{defect}_{charge}"
            # print(f"Analysing {defect_species}")
            energies_file = (
                f"{base_path}{defect_species}/{distortion_type}/{defect_species}.txt"
            )
            energies_dict, energy_diff, gs_distortion = sort_data(energies_file)

            # If a significant energy lowering was found with bond distortions (not just rattling),
            # then further analyse this defect
            if (
                plot_tag
                and ("rattled" not in energies_dict["bond_distortions"].keys())
                and float(energy_diff)
                < -0.1  # TODO: Have energy lowering tolerance as an
                # optional parameter
            ):
                num_nearest_neighbours = distortion_metadata["defects"][defect]["charges"][
                    str(charge)
                ][
                    "num_nearest_neighbours"
                ]  # get number of distorted neighbours
                neighbour_atoms = list(
                    i[0]
                    for i in distortion_metadata["defects"][defect]["charges"][
                        str(charge)
                    ]["distorted_atoms"]
                )  # get element distorted
                if all(element == neighbour_atoms[0] for element in neighbour_atoms):
                    neighbour_atom = neighbour_atoms[0]
                else:
                    neighbour_atom = (
                        "nn"  # if different elements were distorted, just use nearest
                    )
                    # neighbours (nn) for label
                f = plot_defect(
                    defect_species=defect_species,
                    charge=charge,
                    energies_dict=energies_dict,
                    base_path=base_path,
                    neighbour_atom=neighbour_atom,
                    num_nearest_neighbours=num_nearest_neighbours,
                    add_colorbar=add_colorbar,
                    metric=metric,
                    units=units,
                    max_energy_above_unperturbed=max_energy_above_unperturbed,
                )
                figures[defect_species] = f
    return figures


def plot_defect(
    defect_species: str,
    charge: int,
    energies_dict: dict,
    base_path: Optional[str] = None,
    neighbour_atom: Optional[str] = None,
    num_nearest_neighbours: Optional[int] = None,
    add_colorbar: bool = False,
    metric: str = "max_dist",
    max_energy_above_unperturbed: float = 0.5,
    include_site_num_in_name: bool = False,
    save_tag: bool = True,
    y_axis: str = "Energy (eV)",
    units: str = "eV",
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
            Dictionary matching distortion to final energy (eV), as produced by `organize_data()`.
        base_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to calculate structure
            comparisons)
        neighbour_atom (:obj:`str`):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between each structure and
            the unperturbed one.
        metric (:obj:`str`):
            If add_colorbar is True, determines the criteria used for the structural comparison.
            Can choose between root-mean-squared displacement for all sites ('rms') or the
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
        save_tag (:obj:`bool`):
            Whether to save the plot as an SVG file.
            (Default: True)
        y_axis (:obj:`str`):
            Y axis label (Default: 'Energy (eV)')
        units (:obj:`str`):
            Units for energy, either "eV" or "meV" (Default: "eV")

    Returns:
        Energy vs distortion plot, as a Matplotlib Figure object
    """
    # TODO: Add save_format option
    if add_colorbar:  # then get structures to compare their similarity
        assert os.path.isdir(base_path)
        defect_structs = get_structures(defect_species, base_path)
        rms_dict = calculate_struct_comparison(
            defect_structs, metric=metric
        )  # calculate root mean squared displacement and maximum displacement between paired sites
        if (
            rms_dict
        ):  # if struct_comparison algorithms worked (sometimes struggles matching
            # lattices)
            for key in list(rms_dict.keys()):
                # remove any data point if its energy is not in the energy dict (this may be due to
                # relaxation not converged)
                if (
                    (
                        key not in energies_dict["bond_distortions"].keys()
                        and key != "Unperturbed"
                    )
                    or rms_dict[key] == "Not converged"
                    or rms_dict[key] is None
                ):
                    rms_dict.pop(key)
                    if (
                        key in energies_dict["bond_distortions"].keys()
                    ):  # remove it from energy dict as well
                        energies_dict["bond_distortions"].pop(key)
        else:
            print(
                "Structure comparison algorithm struggled matching lattices. Colorbar will not "
                "be added to plot."
            )
            add_colorbar = False

    for key in list(
        energies_dict["bond_distortions"].keys()
    ):  # remove high energy points
        if (
            energies_dict["bond_distortions"][key] - energies_dict["Unperturbed"]
            > max_energy_above_unperturbed
        ):
            energies_dict["bond_distortions"].pop(key)
            if add_colorbar:
                rms_dict.pop(key)

    if charge > 0:
        charge = "+" + str(charge)  # show positive charges with a + sign
    defect_type = defect_species.split("_")[0]  # vac, as or int
    if (
        defect_type == "Int"
    ):  # for interstitials, name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
        site_element = defect_species.split("_")[1]
        site = defect_species.split("_")[2]
        if include_site_num_in_name:
            defect_name = (
                f"{site_element}$_{{i_{site}}}^{{{charge}}}$"  # by default include
            )
            # defect site in defect name for interstitials
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
        else:
            raise ValueError("Defect type not recognized. Please check spelling.")
    else:
        if defect_type == "vac":
            defect_name = f"V$_{{{site_element}}}^{{{charge}}}$"
        elif defect_type in ["as", "sub"]:
            subs_element = defect_species.split("_")[4]
            defect_name = f"{site_element}$_{{{subs_element}}}^{{{charge}}}$"
        else:
            raise ValueError("Defect type not recognized. Please check spelling.")

    if units == "meV":
        y_axis = y_axis.replace("eV", "meV")
        if max_energy_above_unperturbed < 1:  # assume eV
            max_energy_above_unperturbed *= 1000  # convert to meV
        for key in energies_dict["bond_distortions"].keys():  # convert to meV
            energies_dict["bond_distortions"][key] *= 1000
            energies_dict["bond_distortions"][key] = (
                energies_dict["bond_distortions"][key] * 1000
            )
        energies_dict["Unperturbed"] = energies_dict["Unperturbed"] * 1000

    if add_colorbar:
        f = plot_bdm_colorbar(
            energies_dict=energies_dict,
            rms_dict=rms_dict,
            defect_name=defect_name,
            num_nearest_neighbours=num_nearest_neighbours,
            neighbour_atom=neighbour_atom,
            title=defect_name,
            dataset_label=f"BDM: {num_nearest_neighbours} {neighbour_atom}",
            metric=metric,
            save_tag=save_tag,
            y_axis=y_axis,
            max_energy_above_unperturbed=max_energy_above_unperturbed,
        )
    else:
        f = plot_datasets(
            datasets=[energies_dict],
            defect_name=defect_name,
            num_nearest_neighbours=num_nearest_neighbours,
            neighbour_atom=neighbour_atom,
            title=defect_name,
            dataset_labels=[f"BDM: {num_nearest_neighbours} {neighbour_atom}"],
            save_tag=save_tag,
            y_axis=y_axis,
            max_energy_above_unperturbed=max_energy_above_unperturbed,
        )
    return f


def plot_bdm_colorbar(
    energies_dict: dict,
    rms_dict: dict,
    defect_name: str,
    num_nearest_neighbours: int,
    neighbour_atom: str,
    title: Optional[str] = None,
    dataset_label: str = "RBDM",
    metric: str = "max_dist",
    max_energy_above_unperturbed: float = 0.5,
    save_tag: bool = False,
    y_axis: Optional[str] = None,
) -> Figure:
    """
    Plot energy versus bond distortion, adding a colorbar to show structural similarity between
    different final configurations.

    Args:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by `organize_data()`.
        rms_dict (:obj:`dict`):
            Dictionary matching bond distortions to structure comparison metric (metric = 'rms' or
            'max_dist').
        defect_name (:obj:`str`):
            Specific defect name that will appear in plot labels and file names (e.g '$V_{Cd}^0$')
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
        neighbour_atom (:obj:`str`):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
        title (:obj:`str`, optional):
            Plot title
            (Default: None)
        dataset_label (:obj:`str`):
            Label for plot legend
            (Default: 'RBDM')
        metric (:obj:`str`):
            Defines the criteria for structural comparison, used for the colorbar.
            Can choose between root-mean-squared displacement for all sites ('rms') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed structure, to show on
            the plot.
            (Default: 0.5 eV)
        save_tag (:obj:`bool`):
            Whether to save the plot as an SVG file.
            (Default: True)
        y_axis (:obj:`str`):
            Y axis label (Default: 'Energy (eV)')

    Returns:
        Energy vs distortion plot with colorbar for structural similarity, as a Matplotlib Figure
        object
    """
    f, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    if title:
        ax.set_title(title, size=20, pad=15)
    if not y_axis:
        y_axis = r"Energy (eV)"

    for key in list(energies_dict["bond_distortions"].keys()):  # remove high energy points
        if (
            energies_dict["bond_distortions"][key] - energies_dict["Unperturbed"]
            > max_energy_above_unperturbed
        ):
            energies_dict["bond_distortions"].pop(key)
            rms_dict.pop(key)

    array_rms = np.array(np.array(list(rms_dict.values())))

    colormap = sns.cubehelix_palette(
        start=0.65, rot=-0.992075, dark=0.2755, light=0.7205, as_cmap=True
    )  # sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)     #rot=-.952075
    # colormap extremes
    vmin = round(min(array_rms), 1)
    vmax = round(max(array_rms), 1)
    vmedium = round((vmin + vmax) / 2, 1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    # all energies relative to unperturbed one
    for key, i in energies_dict["bond_distortions"].items():
        energies_dict["bond_distortions"][key] = i - energies_dict["Unperturbed"]
    energies_dict["Unperturbed"] = 0.0

    ax.set_xlabel(
        f"BDM Distortion Factor (for {num_nearest_neighbours} {neighbour_atom} near {defect_name})",
        labelpad=10,
    )
    ax.set_ylabel(y_axis, labelpad=10)

    im = ax.scatter(
        energies_dict["bond_distortions"].keys(),
        energies_dict["bond_distortions"].values(),
        c=array_rms[:-1],
        ls="-",
        s=50,
        marker="o",
        # c=pretty_colors["turquoise"],
        cmap=colormap,
        norm=norm,
        alpha=1,
    )
    ax.plot(
        energies_dict["bond_distortions"].keys(),
        energies_dict["bond_distortions"].values(),
        ls="-",
        markersize=1,
        marker="o",
        color=pretty_colors["turquoise"],
        label=dataset_label,
    )
    unperturbed_color = colormap(
        0
    )  # get color of unperturbed structure (corresponding to 0 as RMS is calculated with respect
    # to this structure)
    ax.scatter(
        0,
        energies_dict["Unperturbed"],
        color=unperturbed_color,
        ls="None",
        s=120,
        marker="d",  # markersize=9,
        # cmap = colormap,
        label="Unperturbed",
    )

    # One decimal point if deltaE = (max E - min E) > 0.4 eV, 2 if deltaE > 0.2 eV, otherwise 3
    energy_range = list(energies_dict["bond_distortions"].values())
    energy_range.append(energies_dict["Unperturbed"])
    if (max(energy_range) - min(energy_range)) > 0.4:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.1f}"))
    elif (max(energy_range) - min(energy_range)) < 0.2:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.3f}"))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.2f}"))
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.1f}"))
    plt.legend()

    # Colorbar formatting
    cbar = f.colorbar(
        im,
        ax=ax,
        boundaries=None,
        drawedges=False,
        aspect=20,
        fraction=0.1,
        pad=0.08,
        shrink=0.8,
    )
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)
    if metric == "rms":
        cmap_label = "RMS"
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

    # Save plot?
    if save_tag:
        wd = os.getcwd()
        if not os.path.isdir(wd + "/distortion_plots/"):
            os.mkdir(wd + "/distortion_plots/")
        print(f"Plot saved to {wd}/distortion_plots/")
        plt.savefig(
            wd + "/distortion_plots/" + defect_name + ".svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()
    return f


def plot_datasets(
    datasets: list,
    dataset_labels: list,
    defect_name: str,
    neighbour_atom: str,
    title: Optional[str] = None,
    num_nearest_neighbours: Optional[int] = None,
    max_energy_above_unperturbed: float = 0.6,
    y_axis: str = r"Energy (eV)",
    markers: Optional[list] = None,
    linestyles: Optional[list] = None,
    colors: Optional[list] = None,
    markersize: Optional[float] = None,
    linewidth: Optional[float] = None,
    save_tag: bool = False,
) -> Figure:
    """
    Generate energy versus bond distortion plots for multiple datasets.

    Args:
        datasets (:obj:`list`):
            List of {distortion: energy} dictionaries to plot (each dictionary matching
            distortion to final energy (eV), as produced by `organize_data()`)
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
        y_axis (:obj:`str`):
            Y axis label (Default: 'Energy (eV)')
        markers (:obj:`list`):
            List of markers to use for each dataset (e.g ["o", "d"])
            (Default: None)
        linestyles (:obj:`list`):
            List of line styles to use for each dataset (e.g ["-", "-."])
            (Default: None)
        colors (:obj:`list`):
            List of color codes to use for each dataset (e.g ["C1", "C2"])
            (Default: None)
        markersize (:obj:`float`):
            Marker size to use for plots (single value, or list of values for each dataset)
            (Default: None)
        linewidth (:obj:`float`):
            Linewidth to use for plots (single value, or list of values for each dataset)
            (Default: None)
        save_tag (:obj:`bool`):
            Whether to save the plots as SVG files.
            (Default: True)

    Returns:
        Energy vs distortion plot for multiple datasets, as a Matplotlib Figure object
    """
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.linewidth = 0.10
    if not colors:
        colors = list(colors_dict.values())
    if title:
        ax.set_title(title, size=20, pad=15)

    unperturbed_energies = (
        {}
    )  # energies for unperturbed structure obtained with different methods

    # all energies relative to unperturbed one
    for dataset_number, dataset in enumerate(datasets):

        for key, i in dataset["bond_distortions"].items():
            dataset["bond_distortions"][key] = (
                i - datasets[0]["Unperturbed"]
            )  # Energies relative to unperturbed E of dataset 1

        if dataset_number >= 1:
            dataset["Unperturbed"] = dataset["Unperturbed"] - datasets[0]["Unperturbed"]
            unperturbed_energies[dataset_number] = dataset["Unperturbed"]

        for key in list(dataset["bond_distortions"].keys()):  # remove high E points
            if dataset["bond_distortions"][key] > max_energy_above_unperturbed:
                dataset["bond_distortions"].pop(key)

        if num_nearest_neighbours and neighbour_atom:
            x_label = f"BDM Distortion Factor (for {num_nearest_neighbours} {neighbour_atom} near" \
                      f" {defect_name})"
        else:
            x_label = "BDM Distortion Factor"
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_axis, labelpad=10)

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
                    default_style_settings[key] = optional_style_settings[
                        dataset_number
                    ]
                else:
                    default_style_settings[key] = optional_style_settings

        ax.plot(
            dataset["bond_distortions"].keys(),
            dataset["bond_distortions"].values(),
            markersize=default_style_settings["markersize"],
            marker=default_style_settings["marker"],
            linestyle=default_style_settings["linestyle"],
            c=colors[dataset_number],
            label=dataset_labels[dataset_number],
            linewidth=default_style_settings["linewidth"],
        )

    datasets[0]["Unperturbed"] = 0.0  # unperturbed energy of first dataset (our reference energy)

    for key, value in unperturbed_energies.items():
        if abs(value) > 0.1:
            print(
                f"Energies for unperturbed structures obtained with different methods "
                f"({dataset_labels[key]}) differ by {value:.2f} eV. You may want to check this!"
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

    # One decimal point if energy difference between max E and min E > 0.4 eV, otherwise two
    range_energies = list(datasets[0]["bond_distortions"].values())
    range_energies.append(datasets[0]["Unperturbed"])
    if (max(range_energies) - min(range_energies)) > 0.4:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.1f}"))
    else:
        ax.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.2f}")
        )  # else 2 decimal points
    ax.xaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter("{x:,.1f}")
    )  # 1 decimal for distortion factor

    plt.legend()
    if save_tag:
        wd = os.getcwd()
        if not os.path.isdir(wd + "/distortion_plots/"):
            os.mkdir(wd + "/distortion_plots/")
        print(f"Plot saved to {wd}/distortion_plots/")
        plt.savefig(
            wd + "/distortion_plots/" + defect_name + ".svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()
    return f
