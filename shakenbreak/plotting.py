"""
Module containing functions to plot distorted defect relaxation outputs and identify
energy-lowering distortions.
"""
import datetime
import os
import re
import shutil
import warnings
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from matplotlib.figure import Figure
from pymatgen.core.periodic_table import Element

from shakenbreak import analysis

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _install_custom_font():
    """Check if SnB custom font has been installed, and install it otherwise."""
    # Find where matplotlib stores its True Type fonts
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_fonts_dir = os.path.join(mpl_data_dir, "fonts", "ttf")
    custom_fonts = [
        font
        for font in font_manager.findSystemFonts(fontpaths=mpl_fonts_dir, fontext="ttf")
        if "montserrat" in font.lower()
    ]
    if not custom_fonts:  # If custom hasn't been installed, install it
        print("Trying to install ShakeNBreak custom font...")
        try:
            # Copy the font file to matplotlib's True Type font directory
            fonts_dir = f"{MODULE_DIR}/../fonts/"
            try:
                for file_name in os.listdir(fonts_dir):
                    if ".ttf" in file_name:  # must be in ttf format for matplotlib
                        old_path = os.path.join(fonts_dir, file_name)
                        new_path = os.path.join(mpl_fonts_dir, file_name)
                        shutil.copyfile(old_path, new_path)
                        print("Copying " + old_path + " -> " + new_path)
                    else:
                        print(f"No ttf fonts found in the {fonts_dir} directory.")
            except Exception:
                pass

            # Try to delete matplotlib's fontList cache
            mpl_cache_dir = mpl.get_cachedir()
            mpl_cache_dir_ls = os.listdir(mpl_cache_dir)
            if "fontList.cache" in mpl_cache_dir_ls:
                fontList_path = os.path.join(mpl_cache_dir, "fontList.cache")
                if fontList_path:
                    os.remove(fontList_path)
                    print("Deleted the matplotlib fontList.cache.")
            else:
                print("Couldn't find matplotlib cache, so will continue.")

            # Add font to MAtplotlib Fontmanager
            for font in os.listdir(fonts_dir):
                font_manager._load_fontmanager(try_read_cache=False)
                font_manager.fontManager.addfont(f"{fonts_dir}/{font}")
                print(f"Adding {font} font to matplotlib fonts.")
        except Exception:
            warning_msg = """WARNING: An issue occured while installing the custom font for ShakeNBreak.
                The widely available Helvetica font will be used instead."""
            warnings.warn(warning_msg)


# Helper functions for formatting plots
def _verify_data_directories_exist(
    output_path: str,
    defect_species: str,
) -> None:
    """Check top-level directory (e.g. `output_path`) and defect folders exist."""
    # Check directories and input
    if not os.path.isdir(output_path):  # if output_path does not exist, raise error
        raise FileNotFoundError(
            f"Path {output_path} does not exist! Skipping {defect_species}."
        )
    if not os.path.isdir(
        f"{output_path}/{defect_species}"
    ):  # check if defect directory exists
        raise FileNotFoundError(
            f"Path {output_path}/{defect_species} does not exist! "
            f"Skipping {defect_species}."
        )


def _parse_distortion_metadata(distortion_metadata, defect, charge) -> tuple:
    """
    Parse the number and type of distorted nearest neighbours for a
    given defect from the distortion_metadata dictionary.
    """
    if defect in distortion_metadata["defects"].keys():
        try:
            # Get number and element symbol of the distorted site(s)
            num_nearest_neighbours = distortion_metadata["defects"][defect]["charges"][
                str(charge)
            ][
                "num_nearest_neighbours"
            ]  # get number of distorted neighbours
        except KeyError:
            num_nearest_neighbours = None
        try:
            neighbour_atoms = list(
                i[1]  # element symbol
                for i in distortion_metadata["defects"][defect]["charges"][str(charge)][
                    "distorted_atoms"
                ]
            )  # get element of the distorted site

            if all(element == neighbour_atoms[0] for element in neighbour_atoms):
                neighbour_atom = neighbour_atoms[0]
            else:
                neighbour_atom = "NN"  # if different elements were
                # distorted, just use nearest neighbours (NN) for label

        except (KeyError, TypeError, ValueError):
            neighbour_atom = (
                "NN"  # if distorted_elements wasn't set, set label
                # to "NN"
            )
    else:
        num_nearest_neighbours, neighbour_atom = None, None
    return num_nearest_neighbours, neighbour_atom


def _format_defect_name(
    defect_species: str,
    include_site_num_in_name: bool,
) -> str:
    """
    Format defect name for plot titles. (i.e. from vac_1_Cd_0 to $V_{Cd}^{0}$).
    Note this assumes "V_" means vacancy not Vanadium.

    Args:
        defect_species (:obj:`str`):
            name of defect including charge state (e.g. vac_1_Cd_0)
        include_site_num_in_name (:obj:`bool`):
            whether to include site number in name (e.g. $V_{Cd}^{0}$ or
            $V_{Cd,1}^{0}$)

    Returns:
        :obj:`str`:
            formatted defect name
    """
    if not isinstance(defect_species, str):  # Check inputs
        raise (TypeError(f"`defect_species` {defect_species} should be a string"))
    try:
        charge = defect_species.split("_")[-1]  # charge comes last
        charge = int(charge)
    except ValueError:
        raise (
            ValueError(
                f"Problem reading defect name {defect_species}, should end with charge state "
                f"after underscore (e.g. vac_1_Cd_0)"
            )
        )
    # Format defect name for title/axis labels
    if charge > 0:
        charge = "+" + str(charge)  # show positive charges with a + sign

    recognised_pre_vacancy_strings = sorted(
        [
            "V",
            "v",
            "V_",
            "v_",
            "Vac",
            "vac",
            "Vac_",
            "vac_",
            "Va",
            "va",
            "Va_",
            "va_",
        ],
        key=len,
        reverse=True,
    )
    recognised_post_vacancy_strings = sorted(
        [
            "_v",  # but not '_V' as could be vanadium
            "v",  # but not 'V' as could be vanadium
            "_vac",
            "_Vac",
            "vac",
            "Vac",
            "va",
            "Va",
            "_va",
            "_Va",
        ],
        key=len,
        reverse=True,
    )
    recognised_pre_interstitial_strings = sorted(
        [
            "i",  # but not 'I' as could be iodine
            "i_",  # but not 'I_' as could be iodine
            "Int",
            "int",
            "Int_",
            "int_",
            "Inter",
            "inter",
            "Inter_",
            "inter_",
        ],
        key=len,
        reverse=True,
    )
    recognised_post_interstitial_strings = sorted(
        [
            "_i",  # but not '_I' as could be iodine
            "i",  # but not 'I' as could be iodine
            "_int",
            "_Int",
            "int",
            "Int",
            "inter",
            "Inter",
            "_inter",
            "_Inter",
        ],
        key=len,
        reverse=True,
    )

    def _check_matching_defect_format(
        element, name, pre_def_type_list, post_def_type_list
    ):
        if any(
            f"{pre_def_type}{element}" in name for pre_def_type in pre_def_type_list
        ) or any(
            f"{element}{post_def_type}" in name for post_def_type in post_def_type_list
        ):
            return True
        else:
            return False

    def _check_matching_defect_format_with_site_num(
        element, name, pre_def_type_list, post_def_type_list
    ):
        for site_preposition in ["s", "m", "mult", ""]:
            for site_postposition in [r"[a-z]", ""]:
                match = re.match(
                    r"([a-z_]+)("
                    + site_preposition
                    + r"[0-9]+"
                    + site_postposition
                    + r")",
                    name,
                    re.I,
                )

                if match:
                    items = match.groups()
                    for match_generator in [
                        (
                            fstring in name
                            for pre_def_type in pre_def_type_list
                            for fstring in [
                                f"{pre_def_type}{items[1]}{element}",
                                f"{pre_def_type}{element}{items[1]}",
                                f"{pre_def_type}{items[1]}_{element}",
                                f"{pre_def_type}{element}_{items[1]}",
                            ]
                        ),
                    ]:
                        if any(match_generator):
                            return True, items[1].replace("mult", "m")

                    for match_generator in [
                        (
                            fstring in name
                            for post_def_type in post_def_type_list
                            for fstring in [
                                f"{element}{items[1]}{post_def_type}",
                                f"{items[1]}{element}{post_def_type}",
                                f"{element}{items[1]}_{post_def_type}",
                                f"{items[1]}_{element}{post_def_type}",
                            ]
                        ),
                    ]:
                        if any(match_generator):
                            return True, items[1].replace("mult", "m")

        return False, None

    def _try_vacancy_interstitial_match(
        element,
        name,
        include_site_num_in_name,
        pre_vacancy_strings=None,
        post_vacancy_strings=None,
        pre_interstitial_strings=None,
        post_interstitial_strings=None,
    ):
        if pre_vacancy_strings is None:
            pre_vacancy_strings = recognised_pre_vacancy_strings
        if post_vacancy_strings is None:
            post_vacancy_strings = recognised_post_vacancy_strings
        if pre_interstitial_strings is None:
            pre_interstitial_strings = recognised_pre_interstitial_strings
        if post_interstitial_strings is None:
            post_interstitial_strings = recognised_post_interstitial_strings
        defect_name = None
        defect_name_without_site_num = None
        defect_name_with_site_num = None

        match_found, site_num = _check_matching_defect_format_with_site_num(
            element,
            name,
            pre_vacancy_strings,
            post_vacancy_strings,
        )
        if match_found:
            defect_name_with_site_num = f"$V_{{{element}_{{{site_num}}}}}^{{{charge}}}$"
            defect_name_without_site_num = f"$V_{{{element}}}^{{{charge}}}$"

        else:
            match_found, site_num = _check_matching_defect_format_with_site_num(
                element,
                name,
                pre_interstitial_strings,
                post_interstitial_strings,
            )
            if match_found:
                defect_name_with_site_num = (
                    f"{element}$_{{i_{{{site_num}}}}}^{{{charge}}}$"
                )
                defect_name_without_site_num = f"{element}$_i^{{{charge}}}$"

        if include_site_num_in_name and defect_name_with_site_num is not None:
            defect_name = defect_name_with_site_num

        if (
            _check_matching_defect_format(
                element, name, pre_vacancy_strings, post_vacancy_strings
            )
            and defect_name is None
        ):
            defect_name = f"$V_{{{element}}}^{{{charge}}}$"
        elif (
            _check_matching_defect_format(
                element,
                name,
                pre_interstitial_strings,
                post_interstitial_strings,
            )
            and defect_name is None
        ):
            defect_name = f"{element}$_i^{{{charge}}}$"

        if defect_name is None and defect_name_without_site_num is not None:
            defect_name = defect_name_without_site_num

        return defect_name

    def _try_substitution_match(
        substituting_element, orig_site_element, name, include_site_num_in_name
    ):
        defect_name = None
        if (
            f"{substituting_element}_{orig_site_element}" in name
            or f"{substituting_element}_on_{orig_site_element}" in name
        ):
            defect_name = (
                f"{substituting_element}$_{{{orig_site_element}}}^{{{charge}}}$"
            )

        if (
            defect_name and include_site_num_in_name
        ):  # if we have a match, check if we can add the site number
            for site_preposition in ["s", "m", "mult", ""]:
                for site_postposition in [r"[a-z]", ""]:
                    match = re.match(
                        r"([a-z_]+)("
                        + site_preposition
                        + r"[0-9]+"
                        + site_postposition
                        + r")",
                        name,
                        re.I,
                    )

                    if match:
                        items = match.groups()
                        if any(
                            fstring in name
                            for fstring in [
                                f"{items[1]}_{substituting_element}_{orig_site_element}",
                                f"{substituting_element}_{orig_site_element}_{items[1]}",
                                f"{items[1]}_{substituting_element}_on_{orig_site_element}",
                                f"{substituting_element}_on_{orig_site_element}_{items[1]}",
                            ]
                        ):
                            defect_name = (
                                f"{substituting_element}$_{{{orig_site_element}_{{{items[1]}}}}}^"
                                f"{{{charge}}}$"
                            )
                            return defect_name.replace("mult", "m")

        if defect_name:
            defect_name = defect_name.replace("mult", "m")
        return defect_name

    def _defect_name_from_matching_elements(
        element_matches, name, include_site_num_in_name
    ):
        defect_name = None
        if len(element_matches) == 1:  # vacancy or interstitial?
            defect_name = _try_vacancy_interstitial_match(
                element_matches[0], name, include_site_num_in_name
            )
        elif len(element_matches) == 2:
            # try substitution/antisite match, if not try vacancy/interstitial with first element
            defect_name = _try_substitution_match(
                element_matches[0], element_matches[1], name, include_site_num_in_name
            )
            if defect_name is None:
                defect_name = _try_vacancy_interstitial_match(
                    element_matches[0], name, include_site_num_in_name
                )
        else:
            # try use first match and see if we match vacancy or interstitial format
            # if not, try first and second matches and see if we match substitution format
            # otherwise fail
            defect_name = _try_vacancy_interstitial_match(
                element_matches[0], name, include_site_num_in_name
            )
            if defect_name is None:
                defect_name = _try_substitution_match(
                    element_matches[0],
                    element_matches[1],
                    name,
                    include_site_num_in_name,
                )

        return defect_name

    defect_name = None
    dummy_h = Element("H")
    pre_charge_name = defect_species.rsplit("_", 1)[
        0
    ]  # defect name without charge state

    # trim any matching pre or post vacancy/interstitial strings from defect name
    trimmed_pre_charge_name = pre_charge_name
    for substring in (
        recognised_pre_vacancy_strings
        + recognised_post_vacancy_strings
        + recognised_pre_interstitial_strings
        + recognised_post_interstitial_strings
    ):
        if substring in trimmed_pre_charge_name and not (
            substring.endswith("i") or substring.startswith("i")
        ):
            trimmed_pre_charge_name = trimmed_pre_charge_name.replace(substring, "")

    two_character_pairs_in_name = [
        trimmed_pre_charge_name[
            i : i + 2
        ]  # trimmed_pre_charge_name name for finding elements,
        # pre_charge_name for matching defect format
        for i in range(0, len(trimmed_pre_charge_name), 1)
        if len(trimmed_pre_charge_name[i : i + 2]) == 2
    ]
    possible_two_character_elements = [
        two_char_string
        for two_char_string in two_character_pairs_in_name
        if dummy_h.is_valid_symbol(two_char_string)
    ]

    if len(possible_two_character_elements) > 0:
        defect_name = _defect_name_from_matching_elements(
            possible_two_character_elements,
            pre_charge_name,  # trimmed_pre_charge_name name for finding elements, pre_charge_name
            # for matching defect format
            include_site_num_in_name,
        )

        if defect_name is None and len(possible_two_character_elements) == 1:
            # possibly one single-character element and one two-character element
            possible_one_character_elements = [
                character
                for character in trimmed_pre_charge_name.replace(
                    possible_two_character_elements[0], ""
                )
                if dummy_h.is_valid_symbol(character)
            ]
            if len(possible_one_character_elements) > 0:
                defect_name = _defect_name_from_matching_elements(
                    possible_one_character_elements + possible_two_character_elements,
                    pre_charge_name,  # trimmed_pre_charge_name name for finding elements,
                    # pre_charge_name for matching defect format
                    include_site_num_in_name,
                )

    if defect_name is None:
        # try single-character element match
        possible_one_character_elements = [
            character
            for character in trimmed_pre_charge_name  # trimmed_pre_charge_name name for finding
            # elements, pre_charge_name for matching defect format
            if dummy_h.is_valid_symbol(character)
        ]

        if len(possible_one_character_elements) > 0:
            defect_name = _defect_name_from_matching_elements(
                possible_one_character_elements,
                pre_charge_name,  # trimmed_pre_charge_name name for finding elements,
                # pre_charge_name for matching defect format
                include_site_num_in_name,
            )

    if defect_name is None:
        # try matching to PyCDT/doped style:
        try:
            defect_type = defect_species.split("_")[0]  # vac, as or int
            if (
                defect_type.capitalize() == "Int"
            ):  # for interstitials, name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
                site_element = defect_species.split("_")[1]
                site = defect_species.split("_")[2]
                if include_site_num_in_name:
                    # by default include defect site in defect name for interstitials
                    defect_name = f"{site_element}$_{{i_{{{site}}}}}^{{{charge}}}$"
                else:
                    defect_name = f"{site_element}$_i^{{{charge}}}$"
            else:
                site = defect_species.split("_")[
                    1
                ]  # number indicating defect site (from doped)
                site_element = defect_species.split("_")[2]  # element at defect site

            if (
                include_site_num_in_name
            ):  # whether to include the site number in defect name
                if defect_type.lower() == "vac":
                    defect_name = f"$V_{{{site_element}_{{{site}}}}}^{{{charge}}}$"
                    # double brackets to treat it literally (tex), then extra {} for
                    # python str formatting
                elif defect_type.lower() in ["as", "sub"]:
                    subs_element = defect_species.split("_")[4]
                    defect_name = (
                        f"{site_element}$_{{{subs_element}_{{{site}}}}}^{{{charge}}}$"
                    )
                elif defect_type.capitalize() != "Int":
                    raise ValueError(
                        "Defect type not recognized. Please check spelling."
                    )
            else:
                if defect_type.lower() == "vac":
                    defect_name = f"$V_{{{site_element}}}^{{{charge}}}$"
                elif defect_type.lower() in ["as", "sub"]:
                    subs_element = defect_species.split("_")[4]
                    defect_name = f"{site_element}$_{{{subs_element}}}^{{{charge}}}$"
                elif defect_type.capitalize() != "Int":
                    raise ValueError(
                        f"Defect type {defect_type} not recognized. Please check spelling."
                    )
        except Exception:
            defect_name = None

    return defect_name


def _cast_energies_to_floats(
    energies_dict: dict,
    defect_species: str,
) -> dict:
    """
    If values of the `energies_dict` are not floats, convert them to floats.
    If any problem encountered during conversion, raise ValueError.

    Args:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by
            `_organize_data()` or `analysis.get_energies()`)..
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')

    Returns:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), with all energy
            values as floats.
    """
    if not all(
        isinstance(energy, float)
        for energy in list(energies_dict["distortions"].values())
    ) or not isinstance(energies_dict["Unperturbed"], float):
        # check energies_dict values are floats
        try:
            energies_dict["distortions"] = {
                k: float(v) for k, v in energies_dict["distortions"].items()
            }
            energies_dict["Unperturbed"] = float(energies_dict["Unperturbed"])
        except ValueError:
            raise ValueError(
                f"Values of energies_dict are not floats! Skipping {defect_species}."
            )
    return energies_dict


def _change_energy_units_to_meV(
    energies_dict: dict,
    max_energy_above_unperturbed: float,
    y_label: str,
) -> Tuple[dict, float, str]:
    """
    Converts energy values from eV to meV and format y label accordingly.

    Args:
        energies_dict (dict):
            dictionary with energy values for all distortions
        max_energy_above_unperturbed (float):
            maximum energy value above unperturbed defect
        y_label (str):
            label for y axis

    Returns:
        Tuple[dict, float, str]: (max_energy_above_unperturbed, energies_dict, y_label)
        with energy values in meV
    """
    if "meV" not in y_label:
        y_label = y_label.replace("eV", "meV")
    if max_energy_above_unperturbed < 4:  # assume eV
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
) -> Tuple[dict, dict]:
    """
    Purges dictionaries of displacements and energies so that they are consistent
    (i.e. contain data for same distortions).
    To achieve this, it removes:
    - Any data point from disp_dict if its energy is not in the energy dict
        (this may be due to relaxation not converged).
    - Any data point from energies_dict if its displacement is not in the disp_dict
        (this might be due to the lattice matching algorithm failing).

    Args:
        disp_dict (dict):
            dictionary with displacements (for each structure relative to
            Unperturbed), in the output format of
            `analysis.calculate_struct_comparison()`
        energies_dict (dict):
            dictionary with final energies (for each structure relative to ¡
            Unperturbed), in the output format of `analysis.get_energies()` or
            analysis.organize_data()

    Returns:
        Tuple[dict, dict]: Consistent dictionaries of displacements and energies,
        containing data for same distortions.
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


def _remove_high_energy_points(
    energies_dict: dict,
    max_energy_above_unperturbed: float,
    disp_dict: Optional[dict] = None,
) -> Tuple[dict, dict]:
    """
    Remove points whose energy is higher than the reference (Unperturbed) by
    more than `max_energy_above_unperturbed`.

    Args:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by
            `analysis._organize_data()` or `analysis.get_energies()`
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed
            structure, to show on plot
        disp_dict (:obj:`dict`):
            Dictionary matching distortion to sum of atomic displacements, as
            produced by `analysis.calculate_struct_comparison()`
            (Default: None)

    Returns:
        Tuple[dict, dict]: energies_dict, disp_dict
    """
    for key in list(energies_dict["distortions"].keys()):
        # remove high energy points
        if (
            energies_dict["distortions"][key] - energies_dict["Unperturbed"]
            > max_energy_above_unperturbed
        ):
            energies_dict["distortions"].pop(key)
            if disp_dict:  # only exists if user selected `add_colorbar=True`
                disp_dict.pop(key)
    return energies_dict, disp_dict


def _get_displacement_dict(
    defect_species: str,
    output_path: str,
    metric: str,
    energies_dict: dict,
    add_colorbar: bool,
    code: Optional[str] = "vasp",
) -> Tuple[bool, dict, dict]:
    """
    Parses structures of `defect_species` to calculate displacements between each
    of them and the reference configuration (Unperturbed). These displacements
    are stored in a dictionary matching distortion key to displacement value.
    Then, ensures `energies_dict` and `disp_dict` are consistent (same keys),
    and makes them consistent otherwise.
    If any problems encountered when parsing or calculating structural
    similarity, warning will be raised and `add_colorbar` will be set to False.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to
            calculate structure comparisons)
            (Default: current directory)
        metric (:obj:`str`):
            If add_colorbar is True, determines the criteria used for the
            structural comparison. Can choose between root-mean-squared
            displacement for all sites ('disp') or the
            maximum distance between matched sites ('max_dist', default).
            (Default: 'max_dist')
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by
            `_organize_data()` or `analysis.get_energies()`)
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between
            each structure and the unperturbed one.
        code (:obj:`str`, optional):
            Code used for the geometry relaxations. Valid code names are:
            "vasp", "espresso", "cp2k" and "fhi-aims" (case insensitive).
            (Default: "vasp")

    Returns:
        Tuple[bool, dict, dict]: tuple of `add_colorbar`, `energies_dict` and
        `disp_dict`
    """
    try:
        defect_structs = analysis.get_structures(
            defect_species=defect_species,
            output_path=output_path,
            code=code,
        )
        disp_dict = analysis.calculate_struct_comparison(
            defect_structs, metric=metric
        )  # calculate sum of atomic displacements and maximum displacement
        # between paired sites
        if (
            disp_dict
        ):  # if struct_comparison algorithms worked (sometimes struggles matching
            # lattices)
            disp_dict, energies_dict = _purge_data_dicts(
                disp_dict=disp_dict,
                energies_dict=energies_dict,
            )  # make disp and energies dict consistent by removing any data point
            # if its energy is not in the energy dict and viceversa
        else:
            warnings.warn(
                "Structure comparison algorithm struggled matching lattices. "
                "Colorbar will not be added to plot."
            )
            add_colorbar = False
            return add_colorbar, energies_dict, None
    except FileNotFoundError:  # raised by analysis.get_structures() if
        # defect_directory or distortion subdirectories do not exist
        warnings.warn(
            f"Could not find structures for {output_path}/{defect_species}. "
            "Colorbar will not be added to plot."
        )
        add_colorbar = False
        return add_colorbar, energies_dict, None
    return add_colorbar, energies_dict, disp_dict


def _format_datapoints_from_other_chargestates(
    energies_dict: dict, disp_dict: Optional[dict] = None
) -> tuple:
    """
    Format distortions keys of the energy lowering distortions imported from
    other charge states.

    Args:
        energies_dict (dict):
            Dictionary matching distortion to final energy.
        disp_dict (Optional[dict], optional):
            Dictionary matching distortion to displacement value.
            Defaults to None.

    Returns:
        tuple: imported_indices, sorted_distortions, sorted_energies,
        sorted_disp (if disp_dict is not None)
    """
    # Store indices of imported structures ("X%_from_Y") to plot differently later
    # comparison
    imported_indices = []
    imported_energies = []
    for i, entry in enumerate(energies_dict["distortions"].keys()):
        if isinstance(entry, str) and "_from_" in entry:
            imported_indices.append(i)
            imported_energies.append(energies_dict["distortions"][entry])

    # Reformat any "X%_from_Y" or "Rattled_from_Y" distortions to corresponding
    # (X) distortion factor or 0.0 for "Rattled"
    keys = []
    for entry in energies_dict["distortions"].keys():
        if isinstance(entry, str) and "%_from_" in entry:
            keys.append(float(entry.split("%")[0]) / 100)
        elif isinstance(entry, str) and "Rattled_from_" in entry:
            keys.append(0.0)  # Rattled will be plotted at x = 0.0
        elif entry == "Rattled":  # add 0.0 for Rattled
            # (to avoid problems when sorting distortions)
            keys.append(0.0)
        else:
            keys.append(entry)

    if disp_dict:
        # Sort displacements in same order as distortions and energies,
        # for proper color mapping
        sorted_disp = [
            disp_dict[k]
            for k in energies_dict["distortions"].keys()
            if k in disp_dict.keys()
        ]
        # Save the values of the displacements from *other charge states*
        # As the displacements will be re-sorted -> we'll need to
        # find the index of t
        disps_from_other_charges = (sorted_disp[i] for i in imported_indices)
        try:
            # sort keys and values
            sorted_distortions, sorted_energies, resorted_disp = zip(
                *sorted(zip(keys, energies_dict["distortions"].values(), sorted_disp))
            )
            # Indexes of the displacements values for other charge states
            # We need both the indexes for the unsorted lists
            # and for the sorted ones
            imported_indices = {  # unsorted_index: sorted_index
                unsorted_index: resorted_disp.index(d)
                for unsorted_index, d in zip(imported_indices, disps_from_other_charges)
            }
            return (
                imported_indices,
                keys,
                sorted_distortions,
                sorted_energies,
                resorted_disp,
            )
        except ValueError:  # if keys and energies_dict["distortions"] are empty
            # (i.e. the only distortion is Rattled)
            return {}, [], None, None, None
    # Sort keys and values
    try:
        sorted_distortions, sorted_energies = zip(
            *sorted(zip(keys, energies_dict["distortions"].values()))
        )
        imported_indices = {  # unsorted_index: sorted_index
            unsorted_index: sorted_energies.index(d)
            for unsorted_index, d in zip(imported_indices, imported_energies)
        }
        return imported_indices, keys, sorted_distortions, sorted_energies
    except ValueError:  # if keys and energies_dict["distortions"] are empty
        # (i.e. the only distortion is Rattled)
        return [], [], None, None


def _save_plot(
    fig: plt.Figure,
    defect_name: str,
    output_path: str,
    save_format: str,
    verbose: bool = True,
) -> None:
    """
    Save plot in the defect directory. If defect directory not present/recognised, save to cwd.
    If previous saved plots with the same name exist, rename to <defect>_<datetime>.<format> to
    prevent overwriting.

    Args:
        fig (:obj:`mpl.figure.Figure`):
            mpl.figure.Figure object to save.
        defect_name (:obj:`str`):
            Defect name that will be used as file name and for identifying defect folder.
        output_path (:obj:`str`):
            Path to top-level directory containing the defect directory (in which to save the plot).
        save_format (:obj:`str`):
            Format to save the plot as, given as string.
        verbose (:obj:`bool`, optional):
            Whether to print information about the saved plot. Defaults to True.

    Returns:
        None
    """
    # Locate defect directory; either subfolder in output_path or cwd
    defect_dir = os.path.join(output_path, defect_name)
    if not os.path.isdir(defect_dir):
        defect_dir = output_path

    plot_filepath = f"{os.path.join(defect_dir, defect_name)}.{save_format}"
    # If plot already exists, rename to <defect>_<datetime>.<format>
    if os.path.exists(plot_filepath):
        current_datetime = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M"
        )  # keep copy of old plot file
        os.rename(
            plot_filepath,
            f"{os.path.join(defect_dir, defect_name)}_{current_datetime}.{save_format}",
        )
        if verbose:
            print(
                f"Previous version of {os.path.basename(plot_filepath)} found in "
                f"output_path: '{os.path.basename(os.path.dirname(plot_filepath))}/'. Will rename "
                f"old plot to {defect_name}_{current_datetime}.{save_format}."
            )

    # use pycairo as backend if installed and save_format is pdf:
    backend = None
    if "pdf" in save_format:
        try:
            import cairo

            backend = "cairo"
        except ImportError:
            warnings.warn(
                "pycairo not installed. Defaulting to matplotlib's pdf backend, so default "
                "ShakeNBreak fonts may not be used – try setting `save_format` to 'png' or "
                "`pip install pycairo` if you want ShakeNBreak's default font."
            )

    fig.savefig(
        plot_filepath,
        format=save_format,
        transparent=True,
        bbox_inches="tight",
        backend=backend,
    )
    if verbose:
        print(
            f"Plot saved to {os.path.basename(os.path.dirname(plot_filepath))}"
            f"/{os.path.basename(plot_filepath)}"
        )


def _format_ticks(
    ax: mpl.axes.Axes,
    energies_list: list,
) -> mpl.axes.Axes:
    """
    Format axis ticks of distortion plots and set limits of y axis.
    For the y-axis (energies), show number with:
    - 1 decimal point if energy range is higher than 0.4 eV,
    - 3 decimal points if energy range is smaller than 0.1 eV,
    - 2 decimal points otherwise.

    Args:
        ax (obj:`mpl.axes.Axes`):
            matplotlib.axes.Axes of figure to format
        energies_list (:obj:`list`):
            List of y (energy) values

    Returns:
        mpl.axes.Axes: Formatted axes
    """
    energy_range = max(energies_list) - min(energies_list)

    tick_interval = None
    if energy_range > 0.3:
        tick_interval = 0.1
    if energy_range > 0.6:
        tick_interval = 0.2
    if energy_range > 1.5:
        tick_interval = 0.5
    if energy_range > 2.5:
        # probably something wrong... revert to default
        tick_interval = None

    ylim_lower = min(energies_list) - 0.1 * energy_range  # default lower limit

    if tick_interval:  # set locator to tick_interval
        loc = mpl.ticker.MultipleLocator(base=tick_interval)
        # want the bottom tick to be no more thant (5%)*energy_range above the minimum energy to
        # allow easy visual estimation of energy lowering and scale:
        if (
            min(energies_list) + 0.05 * energy_range
        ) // tick_interval == ylim_lower // tick_interval:
            # means bottom tick is at or above the minimum energy
            ylim_lower = min(energies_list) - tick_interval

    else:  # set locator to default (either <0.1 eV or >2.5 eV energy difference)
        loc = mpl.ticker.AutoLocator()  # default locator

    ax.yaxis.set_major_locator(loc)

    # Limits for y axis:
    ax.set_ylim(
        bottom=ylim_lower,
        top=max(energies_list) + 0.1 * energy_range,
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
    Format and set axis labels and locators of distortion plots.

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
            f"Bond Distortion Factor (for {num_nearest_neighbours} "
            f"{neighbour_atom} near {defect_name})"
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


def _get_line_colors(number_of_colors: int) -> list:
    """
    Get list of colors for plotting several lines.

    Args:
        number_of_colors (int):
            Number of colors.

    Returns:
        list
    """
    if 11 > number_of_colors > 1:
        # If user didnt specify colors and more than one color needed,
        # use deep color palette
        colors = sns.color_palette("deep", 10)
    elif number_of_colors > 11:  # otherwise use colormap
        colors = list(
            mpl.cm.get_cmap("viridis", number_of_colors + 1).colors
        )  # +1 to avoid yellow color (which is at the end of the colormap)
    else:
        colors = [
            "#59a590",
        ]  # Turquoise by default
    return colors


def _setup_colormap(
    disp_dict: dict,
) -> Tuple[mpl.colors.Colormap, float, float, float, mpl.colors.Normalize]:
    """
    Setup colormap to measure structural similarity between structures.

    Args:
        disp_dict (:obj: `dict`):
            dictionary mapping distortion key to structural similarity between
            the associated structure and the reference structure.

    Returns:
        Tuple[mpl.colors.Colormap, float, float, float, mpl.colors.Normalize]:
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
) -> mpl.figure.Figure.colorbar:
    """
    Format colorbar of plot.

    Args:
        fig (:obj:`mpl.figure.Figure`):
            matplotlib.figure.Figure object
        ax (:obj:`mpl.axes.Axes`):
            current matplotlib.axes.Axes object
        im (:obj:`mpl.collections.PathCollection`)
        metric (:obj:`str`):
            metric to be plotted: "disp" or "max_dist"
        vmin (:obj:`float`):
            tick label for the colorbar
        vmax (:obj:`float`):
            tick label for the colorbar
        vmedium (:obj:`float`):
            tick label for the colorbar

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
        cmap_label = r"$\Sigma$ Disp $(\AA)$"
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


# Main plotting functions:
def plot_all_defects(
    defects_dict: dict,
    output_path: str = ".",
    add_colorbar: bool = False,
    metric: str = "max_dist",
    max_energy_above_unperturbed: float = 0.5,
    units: str = "eV",
    min_e_diff: float = 0.05,
    line_color: Optional[str] = None,
    add_title: Optional[bool] = True,
    save_plot: bool = True,
    save_format: str = "png",
    verbose: bool = True,
) -> dict:
    """
    Convenience function to quickly analyse a range of defects and identify those
    which undergo energy-lowering distortions.

    Args:
        defects_dict (:obj:`dict`):
            Dictionary matching defect names to lists of their charge states.
            (e.g {"Int_Sb_1": [0,+1,+2]} etc)
        output_path (:obj:`str`):
            Path to top-level directory with your distorted defect calculations and
            distortion_metadata.json file.
            (Default: current directory)
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between
            each structure and the unperturbed one.
            (Default: False)
        metric (:obj:`str`):
            If add_colorbar is set to True, metric defines the criteria for
            structural comparison. Can choose between root-mean-squared
            displacement for all sites ('disp') or the maximum distance between
            matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed structure,
            to show on the plot.
            (Default: 0.5 eV)
        units (:obj:`str`):
            Units for energy, either "eV" or "meV".
            (Default: "eV")
        min_e_diff (:obj:`float`):
            Minimum energy difference (in eV) between the ground-state defect
            structure and the `Unperturbed` structure to generate the distortion plot.
            (Default: 0.05 eV)
        line_color (:obj:`str`):
            Color of the line connecting points.
            (Default: ShakeNBreak base style)
        add_title (:obj:`bool`):
            Whether to add a title to the plot. By default, the title is the
            formatted defect name (i.e. V$_{Cd}^{0}$).
            (Default: True)
        save_plot (:obj:`bool`):
            Whether to plot the results or not.
            (Default: True)
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'png')
        verbose (:obj:`bool`):
            Whether to print information about the plots (warnings and where they're saved).

    Returns:
        :obj:`dict`:
            Dictionary of {Defect Species (Name & Charge): Energy vs Distortion Plot}

    """
    if not os.path.isdir(output_path):  # check if output_path exists
        raise FileNotFoundError(f"Path {output_path} does not exist!")

    try:
        distortion_metadata = analysis._read_distortion_metadata(
            output_path=output_path
        )
    except FileNotFoundError:
        # check if any defect_species folders have distortion_metadata.json files
        defect_species_list = [
            f"{defect}_{charge}"
            for defect in defects_dict
            for charge in defects_dict[defect]
        ]
        if any(
            os.path.isfile(
                os.path.join(output_path, defect_species, "distortion_metadata.json")
            )
            for defect_species in defect_species_list
        ):
            distortion_metadata = "in_defect_species_folders"

        else:
            if verbose:
                warnings.warn(
                    f"Path {output_path}/distortion_metadata.json does not exist, "
                    f"and distortion_metadata.json files not found in defect folders. "
                    "Will not parse its contents (to specify which neighbour atoms were distorted "
                    "in plot text)."
                )
            distortion_metadata = None
            num_nearest_neighbours = None
            neighbour_atom = None

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
            energies_file = f"{output_path}/{defect_species}/{defect_species}.yaml"
            if not os.path.exists(energies_file):
                warnings.warn(
                    f"Path {energies_file} does not exist. "
                    f"Skipping {defect_species}."
                )  # skip defect
                continue
            energies_dict, energy_diff, gs_distortion = analysis._sort_data(
                energies_file, verbose=False
            )

            if not energy_diff:  # if Unperturbed calc is not converged, warn user
                warnings.warn(
                    f"Unperturbed calculation for {defect}_{charge} not converged! "
                    f"Skipping plot."
                )
                continue
            # If a significant energy lowering was found, then further analyse this defect
            if float(-1 * energy_diff) > abs(min_e_diff):
                # energy_diff is negative if energy is lowered
                if verbose:
                    print(
                        f"Energy lowering distortion found for {defect} with "
                        f"charge {charge}. Generating distortion plot..."
                    )
                if distortion_metadata == "in_defect_species_folders":
                    try:
                        (
                            num_nearest_neighbours,
                            neighbour_atom,
                        ) = analysis._read_distortion_metadata(
                            output_path=f"{output_path}/{defect_species}"
                        )
                    except (
                        FileNotFoundError
                    ):  # distortion_metadata.json not found in this folder
                        pass
                if distortion_metadata and type(distortion_metadata) == dict:
                    num_nearest_neighbours, neighbour_atom = _parse_distortion_metadata(
                        distortion_metadata, defect, charge
                    )
                else:
                    num_nearest_neighbours = None
                    neighbour_atom = None

                figures[defect_species] = plot_defect(
                    defect_species=defect_species,
                    energies_dict=energies_dict,
                    output_path=output_path,
                    neighbour_atom=neighbour_atom,
                    num_nearest_neighbours=num_nearest_neighbours,
                    add_colorbar=add_colorbar,
                    metric=metric,
                    units=units,
                    max_energy_above_unperturbed=max_energy_above_unperturbed,
                    line_color=line_color,
                    add_title=add_title,
                    save_plot=save_plot,
                    save_format=save_format,
                    verbose=verbose,
                )

    return figures


def plot_defect(
    defect_species: str,
    energies_dict: dict,
    output_path: Optional[str] = ".",
    neighbour_atom: Optional[str] = None,
    num_nearest_neighbours: Optional[int] = None,
    add_colorbar: Optional[bool] = False,
    metric: Optional[str] = "max_dist",
    max_energy_above_unperturbed: Optional[float] = 0.5,
    include_site_num_in_name: Optional[bool] = False,
    y_label: Optional[str] = "Energy (eV)",
    add_title: Optional[bool] = True,
    line_color: Optional[str] = None,
    units: Optional[str] = "eV",
    save_plot: Optional[bool] = True,
    save_format: Optional[str] = "png",
    verbose: bool = True,
) -> Optional[Figure]:
    """
    Convenience function to plot energy vs distortion for a defect, to identify
    any energy-lowering distortions.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by
            `_organize_data()` or `analysis.get_energies()`)
        output_path (:obj:`str`):
            Path to top-level directory with your distorted defect calculations
            (to calculate structure comparisons and save plots)
            (Default: current directory)
        neighbour_atom (:obj:`str`):
            Name(s) of distorted neighbour atoms (e.g. 'Cd'). If not specified,
            will be parsed from distortion_metadata.json file.
            (Default: None)
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2). If not specified,
            will be parsed from distortion_metadata.json file.
            (Default: None)
        add_colorbar (:obj:`bool`):
            Whether to add a colorbar indicating structural similarity between
            each structure and the unperturbed one.
            (Default: False)
        metric (:obj:`str`):
            If add_colorbar is True, determines the criteria used for the
            structural comparison. Can choose between the summed of atomic
            displacements ('disp') or the maximum distance between
            matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed
            structure, to show on the plot.
            (Default: 0.5 eV)
        units (:obj:`str`):
            Units for energy, either "eV" or "meV".
            (Default: "eV")
        include_site_num_in_name (:obj:`bool`):
            Whether to include the site number (as generated by doped) in the
            defect name. Useful for materials with many symmetry-inequivalent
            sites.
            (Default: False)
        y_label (:obj:`str`):
            Y axis label
            (Default: "Energy (eV)")
        add_title (:obj:`bool`):
            Whether to add a title to the plot. By default, the title is the
            formatted defect name (i.e. V$_{Cd}^{0}$).
            (Default: True)
        line_color (:obj:`str`):
            Color of the line connecting points.
            (Default: ShakeNBreak base style)
        save_plot (:obj:`bool`):
            Whether to save the plot as an SVG file.
            (Default: True)
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: "png")
        verbose (:obj:`bool`):
            Whether to print information about the plot (warnings and where it's saved).

    Returns:
        :obj:`mpl.figure.Figure`:
            Energy vs distortion plot, as a mpl.figure.Figure object
    """
    # Ensure necessary directories exist, and raise error if not
    try:
        _verify_data_directories_exist(
            output_path=output_path, defect_species=defect_species
        )
    except FileNotFoundError:
        if add_colorbar:
            warnings.warn(
                f"Cannot add colorbar to plot for {defect_species} as {output_path}/"
                f"{defect_species} cannot be found."
            )
            add_colorbar = False

    if "Unperturbed" not in energies_dict.keys():
        # check if unperturbed energies exist
        warnings.warn(
            f"Unperturbed energy not present in energies_dict of {defect_species}! "
            f"Skipping plot."
        )
        return None

    # If not specified, try to parse from distortion_metadata.json file
    if not neighbour_atom and not num_nearest_neighbours:
        try:
            try:
                distortion_metadata = analysis._read_distortion_metadata(
                    output_path=output_path
                )
            except FileNotFoundError:
                distortion_metadata = analysis._read_distortion_metadata(
                    output_path=f"{output_path}/{defect_species}"  # if user moved file
                )
            if distortion_metadata:
                num_nearest_neighbours, neighbour_atom = _parse_distortion_metadata(
                    distortion_metadata=distortion_metadata,
                    defect=defect_species.rsplit("_", 1)[0],
                    charge=defect_species.rsplit("_", 1)[1],
                )
        except FileNotFoundError:
            if verbose:
                warnings.warn(
                    f"Path {output_path}/distortion_metadata.json or {output_path}/"
                    f"{defect_species}/distortion_metadata.json not found. Will not parse "
                    f"its contents (to specify which neighbour atoms were distorted in plot text)."
                )
            pass

    energies_dict = _cast_energies_to_floats(
        energies_dict=energies_dict, defect_species=defect_species
    )  # ensure all energy values are floats, otherwise cast them to floats

    if add_colorbar:  # then get structures and calculate structural similarity
        add_colorbar, energies_dict, disp_dict = _get_displacement_dict(
            defect_species=defect_species,
            output_path=output_path,
            add_colorbar=add_colorbar,
            metric=metric,
            energies_dict=energies_dict,
        )

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

    energies_dict, disp_dict = _remove_high_energy_points(
        energies_dict=energies_dict,
        max_energy_above_unperturbed=max_energy_above_unperturbed,
        disp_dict=disp_dict if add_colorbar else None,
    )  # remove high energy points
    if not energies_dict["distortions"]:
        warnings.warn(
            f"No distortion energies within {max_energy_above_unperturbed} {units} above "
            f"unperturbed structure for {defect_species}. Skipping plot."
        )
        return None

    try:
        defect_name = _format_defect_name(
            defect_species=defect_species,
            include_site_num_in_name=include_site_num_in_name,
        )  # Format defect name for title and axis labels
    except Exception:  # if formatting fails, just use the defect_species name
        defect_name = defect_species

    if num_nearest_neighbours and neighbour_atom:
        legend_label = f"Distortions: {num_nearest_neighbours} {neighbour_atom}"
    elif neighbour_atom:
        legend_label = f"Distortions: {neighbour_atom}"
    else:
        legend_label = "Distortions"

    with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
        if add_colorbar:
            fig = plot_colorbar(
                energies_dict=energies_dict,
                disp_dict=disp_dict,
                defect_species=defect_species,
                include_site_num_in_name=include_site_num_in_name,
                title=defect_name if add_title else None,
                num_nearest_neighbours=num_nearest_neighbours,
                neighbour_atom=neighbour_atom,
                legend_label=legend_label,
                metric=metric,
                y_label=y_label,
                max_energy_above_unperturbed=max_energy_above_unperturbed,
                line_color=line_color,
                save_plot=save_plot,
                output_path=output_path,
                save_format=save_format,
                verbose=verbose,
            )
        else:
            fig = plot_datasets(
                datasets=[energies_dict],
                defect_species=defect_species,
                include_site_num_in_name=include_site_num_in_name,
                title=defect_name if add_title else None,
                num_nearest_neighbours=num_nearest_neighbours,
                neighbour_atom=neighbour_atom,
                dataset_labels=[legend_label],
                y_label=y_label,
                max_energy_above_unperturbed=max_energy_above_unperturbed,
                save_plot=save_plot,
                output_path=output_path,
                save_format=save_format,
                verbose=verbose,
            )
    return fig


def plot_colorbar(
    energies_dict: dict,
    disp_dict: dict,
    defect_species: str,
    include_site_num_in_name: Optional[bool] = False,
    num_nearest_neighbours: int = None,
    neighbour_atom: str = "NN",
    title: Optional[str] = None,
    legend_label: str = "SnB",
    metric: Optional[str] = "max_dist",
    max_energy_above_unperturbed: Optional[float] = 0.5,
    save_plot: Optional[bool] = False,
    output_path: Optional[str] = ".",
    y_label: Optional[str] = "Energy (eV)",
    line_color: Optional[str] = None,
    save_format: Optional[str] = "png",
    verbose: Optional[bool] = True,
) -> Optional[Figure]:
    """
    Plot energy versus bond distortion, adding a colorbar to show structural
    similarity between different final configurations.

    Args:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by
            `analysis.get_energies()` or `analysis._organize_data()`.
        disp_dict (:obj:`dict`):
            Dictionary matching bond distortions to structure comparison metric
            (metric = 'disp' or 'max_dist'), as produced by
            `analysis.calculate_struct_comparison()`.
        defect_species (:obj:`str`):
            Specific defect name that will appear in plot labels (in LaTeX form)
             and file names (e.g 'vac_1_Cd_0')
        include_site_num_in_name (:obj:`bool`):
            Whether to include the site number (as generated by doped) in the
            defect name. Useful for materials with many symmetry-inequivalent
            sites.
            (Default: False)
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
            (Default: None)
        neighbour_atom (:obj:`str`, optional):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
            (Default: "NN")
        title (:obj:`str`, optional):
            Plot title
            (Default: None)
        legend_label (:obj:`str`):
            Label for plot legend
            (Default: 'SnB')
        metric (:obj:`str`):
            Defines the criteria for structural comparison, used for the colorbar.
            Can choose between root-mean-squared displacement for all sites ('disp')
            or the maximum distance between matched sites ('max_dist', default).
            (Default: "max_dist")
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed
            structure, to show on the plot.
            (Default: 0.5 eV)
        line_color (:obj:`str`):
            Color of the line conneting points.
            (Default: ShakeNBreak base style)
        save_plot (:obj:`bool`):
            Whether to save the plot as an SVG file.
            (Default: True)
        output_path (:obj:`str`):
            Path to top-level directory containing the defect directory (in which to save the
            plot).
            (Default: ".")
        y_label (:obj:`str`):
            Y axis label
            (Default: 'Energy (eV)')
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'png')
        verbose (:obj:`bool`):
            Whether to print information about the plot (warnings and where it's saved).

    Returns:
        :obj:`mpl.figure.Figure`:
            Energy vs distortion plot with colorbar for structural similarity,
            as a mpl.figure.Figure object
    """
    _install_custom_font()
    with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))

        # Title and format axis labels and locators
        if title:
            ax.set_title(title)

        try:
            formatted_defect_name = _format_defect_name(
                defect_species, include_site_num_in_name=include_site_num_in_name
            )
        except Exception:
            formatted_defect_name = "defect"

        ax = _format_axis(
            ax=ax,
            y_label=y_label,
            defect_name=formatted_defect_name,
            num_nearest_neighbours=num_nearest_neighbours,
            neighbour_atom=neighbour_atom,
        )

    # All energies relative to unperturbed one
    for key, i in energies_dict["distortions"].items():
        energies_dict["distortions"][key] = i - energies_dict["Unperturbed"]
    energies_dict["Unperturbed"] = 0.0

    energies_dict, disp_dict = _remove_high_energy_points(
        energies_dict=energies_dict,
        disp_dict=disp_dict,
        max_energy_above_unperturbed=max_energy_above_unperturbed,
    )  # Remove high energy points
    if not energies_dict["distortions"]:
        warnings.warn(
            f"No distortion energies within {max_energy_above_unperturbed} eV above "
            f"unperturbed structure for {defect_species}. Skipping plot."
        )
        return None

    # Setting line color and colorbar
    if not line_color:
        line_color = "#59a590"  # By default turquoise
    # colormap to measure structural similarity
    colormap, vmin, vmedium, vmax, norm = _setup_colormap(disp_dict)

    # Format distortion keys from other charge states
    (
        imported_indices,
        keys,
        sorted_distortions,
        sorted_energies,
        sorted_disp,
    ) = _format_datapoints_from_other_chargestates(
        energies_dict=energies_dict, disp_dict=disp_dict
    )

    # Plotting
    line = None  # to later check if line was plotted, for legend formatting
    with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
        if (
            "Rattled" in energies_dict["distortions"].keys()
            and "Rattled" in disp_dict.keys()
        ):
            # Plot Rattled energy
            im = ax.scatter(
                0.0,
                energies_dict["distortions"]["Rattled"],
                c=disp_dict["Rattled"],
                label="Rattled",
                s=50,
                marker="o",
                cmap=colormap,
                norm=norm,
                alpha=1,
            )

        if (
            len(sorted_distortions) > 0
            and len([key for key in energies_dict["distortions"] if key != "Rattled"])
            > 0
        ):  # more than just Rattled
            if imported_indices:  # Exclude datapoints from other charge states
                non_imported_sorted_indices = [
                    i
                    for i in range(len(sorted_distortions))
                    if i not in imported_indices.values()
                ]
            else:
                non_imported_sorted_indices = range(len(sorted_distortions))

            # Plot non-imported distortions
            im = ax.scatter(  # Points for each distortion
                [sorted_distortions[i] for i in non_imported_sorted_indices],
                [sorted_energies[i] for i in non_imported_sorted_indices],
                c=[sorted_disp[i] for i in non_imported_sorted_indices],
                ls="-",
                s=50,
                marker="o",
                cmap=colormap,
                norm=norm,
                alpha=1,
            )
            if len(non_imported_sorted_indices) > 1:  # more than one point
                # Plot line connecting points
                (line,) = ax.plot(
                    [sorted_distortions[i] for i in non_imported_sorted_indices],
                    [sorted_energies[i] for i in non_imported_sorted_indices],
                    ls="-",
                    markersize=1,
                    marker="o",
                    color=line_color,
                    label=legend_label,
                )

            # Datapoints from other charge states
            if imported_indices:
                other_charges = len(
                    set(
                        list(energies_dict["distortions"].keys())[i].split("_")[-1]
                        for i in imported_indices.keys()
                    )
                )  # number of other charge states whose distortions have been imported
                for i, j in zip(imported_indices.keys(), range(other_charges)):
                    other_charge_state = int(
                        list(energies_dict["distortions"].keys())[i].split("_")[-1]
                    )
                    sorted_i = imported_indices[i]  # index for the sorted dicts
                    ax.scatter(
                        np.array(keys)[i],
                        sorted_energies[sorted_i],
                        c=sorted_disp[sorted_i],
                        edgecolors="k",
                        ls="-",
                        s=50,
                        marker=["s", "v", "<", ">", "^", "p", "X"][j],
                        zorder=10,  # make sure it's on top of the other points
                        cmap=colormap,
                        norm=norm,
                        alpha=1,
                        label=f"From {'+' if other_charge_state > 0 else ''}{other_charge_state} "
                        f"charge state",
                    )

        # Plot reference energy
        unperturbed_color = colormap(
            0
        )  # get color of unperturbed structure (corresponding to 0 as disp is calculated with
        # respect to this structure)
        ax.scatter(
            0,
            energies_dict["Unperturbed"],
            color=unperturbed_color,
            ls="None",
            s=120,
            marker="d",
            label="Unperturbed",
        )

        # distortion_range is sorted_distortions range, including 0 if above/below this range
        distortion_range = (
            min(sorted_distortions + (0,)),
            max(sorted_distortions + (0,)),
        )
        # set xlim to distortion_range + 5% (matplotlib default padding), if distortion_range is
        # not zero (only rattled and unperturbed)
        if distortion_range[1] - distortion_range[0] > 0:
            ax.set_xlim(
                distortion_range[0]
                - 0.05 * (distortion_range[1] - distortion_range[0]),
                distortion_range[1]
                + 0.05 * (distortion_range[1] - distortion_range[0]),
            )

        # Formatting of tick labels.
        # For yaxis (i.e. energies): 1 decimal point if deltaE = (max E - min E) > 0.4 eV,
        # 2 if deltaE > 0.1 eV, otherwise 3.
        ax = _format_ticks(
            ax=ax,
            energies_list=list(energies_dict["distortions"].values())
            + [
                energies_dict["Unperturbed"],
            ],
        )

        # reformat 'line' legend handle to include 'im' datapoint handle
        handles, labels = ax.get_legend_handles_labels()
        # get handle and label that corresponds to line, if line present:
        if line:
            line_handle, line_label = [
                (handle, label)
                for handle, label in zip(handles, labels)
                if label == legend_label
            ][0]
            # remove line handle and label from handles and labels
            handles = [handle for handle in handles if handle != line_handle]
            labels = [label for label in labels if label != line_label]
            # add line handle and label to handles and labels, with datapoint handle
            handles = [(im, line_handle)] + handles
            labels = [line_label] + labels

        plt.legend(handles, labels, scatteryoffsets=[0.5], frameon=True).set_zorder(
            100
        )  # make sure it's on top of the other points

        _ = _format_colorbar(
            fig=fig, ax=ax, im=im, metric=metric, vmin=vmin, vmax=vmax, vmedium=vmedium
        )  # Colorbar formatting

    # Save plot?
    if save_plot:
        _save_plot(
            fig=fig,
            defect_name=defect_species,
            output_path=output_path,
            save_format=save_format,
            verbose=verbose,
        )
    return fig


def plot_datasets(
    datasets: list,
    dataset_labels: list,
    defect_species: str,
    include_site_num_in_name: Optional[bool] = False,
    title: Optional[str] = None,
    neighbour_atom: Optional[str] = None,
    num_nearest_neighbours: Optional[int] = None,
    max_energy_above_unperturbed: Optional[float] = 0.5,
    y_label: str = r"Energy (eV)",
    markers: Optional[list] = None,
    linestyles: Optional[list] = None,
    colors: Optional[list] = None,
    markersize: Optional[float] = None,
    linewidth: Optional[float] = None,
    save_plot: Optional[bool] = False,
    output_path: Optional[str] = ".",
    save_format: Optional[str] = "png",
    verbose: Optional[bool] = True,
) -> Figure:
    """
    Generate energy versus bond distortion plots for multiple datasets.

    Args:
        datasets (:obj:`list`):
            List of {distortion: energy} dictionaries to plot (each dictionary
            matching distortion to final energy (eV), as produced by
            `analysis._organize_data()` or `analysis.get_energies()`)
        dataset_labels (:obj:`list`):
            Labels for each dataset plot legend.
        defect_species (:obj:`str`):
            Specific defect name that will appear in plot labels (in LaTeX form)
             and file names (e.g 'vac_1_Cd_0')
        include_site_num_in_name (:obj:`bool`):
            Whether to include the site number (as generated by doped) in the
            defect name. Useful for materials with many symmetry-inequivalent
            sites.
            (Default: False)
        neighbour_atom (:obj:`str`):
            Name(s) of distorted neighbour atoms (e.g. 'Cd')
        title (:obj:`str`, optional):
            Plot title
            (Default: None)
        num_nearest_neighbours (:obj:`int`):
            Number of distorted neighbour atoms (e.g. 2)
        max_energy_above_unperturbed (:obj:`float`):
            Maximum energy (in chosen `units`), relative to the unperturbed
            structure, to show on the plot.
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
            Marker size to use for plots (single value, or list of values for
            each dataset)
            (Default: None)
        linewidth (:obj:`float`):
            Linewidth to use for plots (single value, or list of values for each
            dataset)
            (Default: None)
        save_plot (:obj:`bool`):
            Whether to save the plots.
            (Default: True)
        output_path (:obj:`str`):
            Path to top-level directory containing the defect directory (in which to save the
            plot).
            (Default: ".")
        save_format (:obj:`str`):
            Format to save the plot as.
            (Default: 'png')
        verbose (:obj:`bool`):
            Whether to print information about the plot (warnings and where it's saved).

    Returns:
        :obj:`mpl.figure.Figure`:
            Energy vs distortion plot for multiple datasets,
            as a mpl.figure.Figure object
    """
    # Validate input
    if len(datasets) != len(dataset_labels):
        raise ValueError(
            f"Number of datasets and labels must match! "
            f"You gave me {len(datasets)} datasets and"
            f" {len(dataset_labels)} labels."
        )

    _install_custom_font()
    # Set up figure
    with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
        fig, ax = plt.subplots(1, 1)
        # Line colors
        if not colors:
            colors = _get_line_colors(number_of_colors=len(datasets))  # get list of
            # colors to use for each dataset
        elif len(colors) < len(datasets):
            if verbose:
                warnings.warn(
                    f"Insufficient colors provided for {len(datasets)} datasets. "
                    "Using default colors."
                )
            colors = _get_line_colors(number_of_colors=len(datasets))
        # Title and labels of axis
        if title:
            ax.set_title(title)

    try:
        formatted_defect_name = _format_defect_name(
            defect_species, include_site_num_in_name=include_site_num_in_name
        )
    except Exception:
        formatted_defect_name = "defect"

    ax = _format_axis(
        ax=ax,
        y_label=y_label,
        defect_name=formatted_defect_name,
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

        for key in list(dataset["distortions"].keys()):
            # remove high E points (relative to reference)
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

        # Format distortion keys of the distortions imported from other charge states
        (
            imported_indices,
            keys,
            sorted_distortions,
            sorted_energies,
        ) = _format_datapoints_from_other_chargestates(
            energies_dict=dataset, disp_dict=None
        )
        with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
            if "Rattled" in dataset["distortions"].keys():
                ax.scatter(  # Scatter plot for Rattled (1 datapoint)
                    0.0,
                    dataset["distortions"]["Rattled"],
                    c=colors[dataset_number],
                    s=50,
                    marker=default_style_settings["marker"],
                    label="Rattled",
                )

            if (
                len(sorted_distortions) > 0
                and len([key for key in dataset["distortions"] if key != "Rattled"]) > 0
            ):  # more than just Rattled
                if imported_indices:  # Exclude datapoints from other charge states
                    non_imported_sorted_indices = [
                        i
                        for i in range(len(sorted_distortions))
                        if i not in imported_indices.values()
                    ]
                else:
                    non_imported_sorted_indices = range(len(sorted_distortions))

                if len(non_imported_sorted_indices) > 1:  # more than one point
                    # Plot non-imported distortions
                    ax.plot(  # plot bond distortions
                        [sorted_distortions[i] for i in non_imported_sorted_indices],
                        [sorted_energies[i] for i in non_imported_sorted_indices],
                        c=colors[dataset_number],
                        markersize=default_style_settings["markersize"],
                        marker=default_style_settings["marker"],
                        linestyle=default_style_settings["linestyle"],
                        label=dataset_labels[dataset_number],
                        linewidth=default_style_settings["linewidth"],
                    )

            if imported_indices:
                other_charges = len(
                    set(
                        list(dataset["distortions"].keys())[i].split("_")[-1]
                        for i in imported_indices
                    )
                )  # number of other charge states whose distortions have been imported
                for i, j in zip(imported_indices, range(other_charges)):
                    other_charge_state = int(
                        list(dataset["distortions"].keys())[i].split("_")[-1]
                    )
                    ax.scatter(  # distortions from other charge states
                        np.array(keys)[i],
                        list(dataset["distortions"].values())[i],
                        c=colors[dataset_number],
                        edgecolors="k",
                        ls="-",
                        s=50,
                        zorder=10,  # make sure it's on top of the other lines
                        marker=["s", "v", "<", ">", "^", "p", "X"][
                            j
                        ],  # different markers for different charge states
                        alpha=1,
                        label=f"From {'+' if other_charge_state > 0 else ''}{other_charge_state} "
                        f"charge state",
                    )

    datasets[0][
        "Unperturbed"
    ] = 0.0  # unperturbed energy of first dataset (our reference energy)

    # Plot Unperturbed point for every dataset, relative to the unperturbed energy of first dataset
    for key, value in unperturbed_energies.items():
        if (
            abs(value) > 0.1
        ):  # Only plot if different energy from the reference Unperturbed
            print(
                f"Energies for unperturbed structures obtained with different methods "
                f"({dataset_labels[key]}) differ by {value:.2f}. If testing different "
                "magnetic states (FM, AFM) this is normal, otherwise you may want to check this!"
            )
            with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
                ax.plot(
                    0,
                    datasets[key]["Unperturbed"],
                    ls="None",
                    marker="d",
                    markersize=9,
                    c=colors[key],
                )

    with plt.style.context(f"{MODULE_DIR}/shakenbreak.mplstyle"):
        ax.plot(  # plot our reference energy
            0,
            datasets[0]["Unperturbed"],
            ls="None",
            marker="d",
            markersize=9,
            c=colors[0],
            label="Unperturbed",
        )

    # distortion_range is sorted_distortions range, including 0 if above/below this range
    distortion_range = (min(sorted_distortions + (0,)), max(sorted_distortions + (0,)))
    # set xlim to distortion_range + 5% (matplotlib default padding)
    if distortion_range[1] - distortion_range[0] > 0:
        ax.set_xlim(
            distortion_range[0] - 0.05 * (distortion_range[1] - distortion_range[0]),
            distortion_range[1] + 0.05 * (distortion_range[1] - distortion_range[0]),
        )

    # Format tick labels:
    # For yaxis, 1 decimal point if energy difference between max E and min E
    # > 0.4 eV, 3 if E < 0.1 eV, 2 otherwise
    ax = _format_ticks(
        ax=ax,
        energies_list=list(datasets[0]["distortions"].values())
        + [
            datasets[0]["Unperturbed"],
        ],
    )
    # If several datasets, check min & max energy are included
    if len(datasets) > 1:
        min_energy = min(
            [min(list(dataset["distortions"].values())) for dataset in datasets]
        )
        max_energy = max(
            [max(list(dataset["distortions"].values())) for dataset in datasets]
        )
        ax.set_ylim(
            min_energy - 0.1 * (max_energy - min_energy),
            max_energy + 0.1 * (max_energy - min_energy),
        )

    ax.legend(frameon=True).set_zorder(
        100
    )  # show legend on top of all other datapoints

    if save_plot:  # Save plot?
        _save_plot(
            fig=fig,
            defect_name=defect_species,
            output_path=output_path,
            save_format=save_format,
            verbose=verbose,
        )
    return fig
