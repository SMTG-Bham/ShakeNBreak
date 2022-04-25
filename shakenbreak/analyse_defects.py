"""
Module containing functions to analyse rattled and bond-distorted defect structure relaxations
@author: Irea Mosquera
"""

import json
import os
import sys
from copy import deepcopy
from typing import Optional, Union
import warnings

import pandas as pd
import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

crystalNN = CrystalNN(
    distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False, search_cutoff=5
)

# format warnings output:
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{os.path.split(filename)[-1]}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line

# using stackoverflow.com/questions/15411967/
# how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def isipython():
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False  # Probably standard Python interpreter


if isipython():
    from IPython.display import display


class HiddenPrints:  # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


###################################################################################################
# Helper functions


def open_file(path: str) -> list:
    """Open file and return list of file lines as strings"""
    if os.path.isfile(path):
        with open(path) as ff:
            read_file = ff.read()
            distortion_list = read_file.splitlines()
        return distortion_list
    else:
        print(f"Path {path} does not exist")
        return []


def organize_data(distortion_list: list) -> dict:
    """
    Create a dictionary mapping distortion factors to final energies.

    Args:
        distortion_list (:obj:`list`):
            List of lines in bond distortion output summary file, obtained using
            `BDM_parsing_script.sh` (which specifies bond distortions and corresponding energies).

    Returns:
        Sorted dictionary of bond distortions and corresponding final energies.
    """
    # TODO: Update docstrings here when we implement CLI parsing functions to generate this
    #  output file
    defect_energies_dict = {"distortions": {}}
    for i in range(len(distortion_list) // 2):
        i *= 2
        if "rattle" in distortion_list[i]:
            key = "rattled"
            defect_energies_dict["distortions"][key] = float(distortion_list[i + 1])
        else:
            if "Unperturbed" in distortion_list[i]:
                defect_energies_dict["Unperturbed"] = float(distortion_list[i + 1])
            else:
                key = distortion_list[i].split("_Bond")[0].split("%")[0]
                key = float(key.split("_")[-1]) / 100  # from % to decimal
                defect_energies_dict["distortions"][key] = float(distortion_list[i + 1])

    # Order dict items by key (e.g. from -0.6 to 0 to +0.6)
    sorted_energies_dict = {
        "distortions": dict(sorted(defect_energies_dict["distortions"].items()))
    }
    if "Unperturbed" in defect_energies_dict:
        sorted_energies_dict["Unperturbed"] = defect_energies_dict["Unperturbed"]
    return sorted_energies_dict


def get_gs_distortion(defect_energies_dict: dict):
    """
    Calculate energy difference between `Unperturbed` structure and lowest energy distortion.
    Returns the energy (in eV) and bond distortion of the ground-state relative to `Unperturbed`.
    If `Unperturbed` not present, returns (None, ground-state bond distortion).

    Args:
        defect_energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy, as produced by `organize_data()`.

    Returns:
        (Energy difference, ground state bond distortion)
    """
    lowest_E_distortion = min(
        defect_energies_dict["distortions"].values()
    )  # lowest energy obtained with bond distortions
    if "Unperturbed" in defect_energies_dict:
        if len(defect_energies_dict["distortions"]) == 1:
            energy_diff = (
                defect_energies_dict["distortions"]["rattled"]
                - defect_energies_dict["Unperturbed"]
            )
            if energy_diff < 0:
                gs_distortion = "rattled"  # just rattle (no bond distortion)
            else:
                gs_distortion = "Unperturbed"
        else:
            energy_diff = lowest_E_distortion - defect_energies_dict["Unperturbed"]
            if (
                lowest_E_distortion < defect_energies_dict["Unperturbed"]
            ):  # if energy lower than Unperturbed
                gs_distortion = list(defect_energies_dict["distortions"].keys())[
                    list(defect_energies_dict["distortions"].values()).index(
                        lowest_E_distortion
                    )
                ]  # bond distortion that led to ground-state
            else:
                gs_distortion = "Unperturbed"
    else:
        energy_diff = None
        gs_distortion = list(defect_energies_dict["distortions"].keys())[
            list(defect_energies_dict["distortions"].values()).index(
                lowest_E_distortion
            )
        ]

    return energy_diff, gs_distortion


def sort_data(energies_file: str):
    """
    Organize bond distortion results in a dictionary, calculate energy of ground-state defect
    structure relative to `Unperturbed` structure (in eV) and its corresponding bond distortion,
    and return all three as a tuple. If `Unperturbed` not present, returns (defect_energies_dict,
    None, ground-state distortion).

    Args:
        energies_file (:obj:`str`):
            Path to txt file with bond distortions and final energies (in eV), obtained using
            `BDM_parsing_script.sh`.

    Returns:
        defect_energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy, as produced by `organize_data()`
        energy_diff (:obj:`float`):
            Energy difference between minimum energy structure and `Unperturbed` (in eV).
            None if `Unperturbed` not present.
        gs_distortion (:obj:`float`):
            Distortion corresponding to the minimum energy structure
    """
    defect_energies_dict = organize_data(open_file(energies_file))
    energy_diff, gs_distortion = get_gs_distortion(defect_energies_dict)
    defect_name = energies_file.split("/")[-1].split(".txt")[0]
    if energy_diff and energy_diff < -0.1:
        print(
            f"{defect_name}: Energy difference between minimum, found with {gs_distortion} bond "
            f"distortion, and unperturbed: {energy_diff:+.2f} eV.\n"
        )
    elif energy_diff is None:
        print(
            f"{defect_name}: Unperturbed energy not found in {energies_file}. Lowest energy "
            f"structure found with {gs_distortion} bond distortion.\n"
        )
    return defect_energies_dict, energy_diff, gs_distortion


###################################################################################################


def grab_contcar(file_path: str) -> Structure:
    """
    Read VASP structure from `file_path` and convert to `pymatgen` Structure object.

    Args:
        file_path (:obj:`str`):
            Path to VASP `CONTCAR` file

    Returns:
        `pymatgen` Structure object
    """
    abs_path_formatted = file_path.replace("\\", "/")  # for Windows compatibility
    if not os.path.isfile(abs_path_formatted):
        warnings.warn(
            f"{abs_path_formatted} file doesn't exist, storing as 'Not converged'. Check path & "
            f"relaxation"
        )
        struct = "Not converged"
    else:
        try:
            struct = Structure.from_file(abs_path_formatted)
        except:
            warnings.warn(
                f"Problem obtaining structure from: {abs_path_formatted}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            struct = "Not converged"
    return struct


def analyse_defect_site(
    structure: Structure,
    name: str = "Unnamed Defect",
    site_num: Optional[int] = None,
    vac_site: Optional[list] = None,
):
    """
    Analyse coordination environment and bond distances to nearest neighbours of defect site.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            `pymatgen` Structure object to analyse
        name (:obj:`str`):
            Defect name for printing. (Default: "Unnamed Defect")
        site_num (:obj:`int`):
            Defect site index in the structure, starting from 1 (VASP rather than python
            indexing). (Default: None)
        vac_site (:obj:`list`):
            For vacancies, the fractional coordinates of the vacant lattice site.
            (Default: None)

    Returns:
        Tuple of coordination analysis and bond length DataFrames, respectively.
    """
    # get defect site
    struct = deepcopy(structure)
    if site_num:
        isite = site_num - 1  # python/pymatgen indexing (starts counting from zero!)
    elif vac_site:
        struct.append(
            "V", vac_site
        )  # Have to add a fake element for coordination analysis
        isite = (
            len(struct.sites) - 1
        )  # python/pymatgen indexing (starts counting from zero!)
    else:
        raise ValueError("Either site_num or vac_site must be specified")

    print("==> ", name + " structural analysis ", " <==")
    print("Analysing site", struct[isite].specie, struct[isite].frac_coords)
    coordination = crystalNN.get_local_order_parameters(struct, isite)
    if coordination is not None:
        coord_list = []
        for coord, value in coordination.items():
            coordination_dict = {"Coordination": coord, "Factor": value}
            coord_list.append(coordination_dict)
        print(
            "Local order parameters (i.e. resemblance to given structural motif, via CrystalNN):"
        )
        if isipython():
            display(pd.DataFrame(coord_list))  # display in Jupyter notebook
    # Bond Lengths:
    bond_lengths = []
    for i in crystalNN.get_nn_info(struct, isite):
        bond_lengths.append(
            {
                "Element": i["site"].specie.as_dict()["element"],
                "Distance": f"{i['site'].distance(struct[isite]):.3f}",
            }
        )
    bond_length_df = pd.DataFrame(bond_lengths)
    print("\nBond-lengths (in \u212B) to nearest neighbours: ")
    if isipython():
        display(bond_length_df)
        print()  # spacing
    if coordination is not None:
        return pd.DataFrame(coord_list), bond_length_df
    else:
        return None, bond_length_df


def analyse_structure(
    defect_species: str,
    structure: Structure,
    output_path: str,
):
    """
    Analyse the local distortion of the input defect structure. Requires access to the
    distortion_metadata.json file generated with ShakeNBreak to read info about defect site.
    If lacking this, can alternatively use `analyse_defect_site`.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure to analyse
        output_path (:obj:`str`):
            Path to directory containing `distortion_metadata.json`

    Returns:
        Tuple of coordination analysis and bond length DataFrames, respectively.
    """
    defect_name_without_charge = defect_species.rsplit("_", 1)[0]

    # Read site from distortion_metadata.json
    with open(f"{output_path}/distortion_metadata.json", "r") as json_file:
        distortion_metadata = json.load(json_file)

    defect_site = distortion_metadata["defects"][defect_name_without_charge].get(
        "defect_site_index"
    )  # VASP indexing (starts counting from 1)
    if defect_site is None:  # for vacancies, get fractional coordinates
        defect_frac_coords = distortion_metadata["defects"][defect_name_without_charge][
            "unique_site"
        ]
        return analyse_defect_site(
            structure, name=defect_species, vac_site=defect_frac_coords
        )
    return analyse_defect_site(structure, name=defect_species, site_num=defect_site)


# TODO: Refactor `get_structures` to read the distortions present from the subfolders,
#  rather than requiring it to be specified in the function argument.
def get_structures(
    defect_species: str,
    output_path: str = "./",
    distortion_increment: Optional[float] = None,
    bond_distortions: Optional[list] = None,
    distortion_type="BDM",
) -> dict:
    """
    Import all structures found with rattling & bond distortions, and store them in a dictionary
    matching the bond distortion to the final structure. By default, will read the
    `distortion_metadata.json` file (generated with ShakeNBreak) if present in the current
    directory (and `distortion_increment` and `bond_distortions` not specified.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        output_path (:obj:`str`):
            Path to top-level directory containing `defect_species` subdirectories. (Default is
            current directory)
        distortion_increment (:obj:`float`):
            Bond distortion increment used. Assumes range of +/-60% (otherwise use
            `bond_distortions`). (Default:None)
        bond_distortions (:obj:`list`):
            List of distortions applied to nearest neighbours, instead of the default set
            (e.g. [-0.5, 0.5]). (Default: None)
        distortion_type (:obj:`str`):
            Type of distortion method used.
            Either 'BDM' (bond distortion method (standard)) or 'champion'. The option 'champion'
            is used when relaxing a defect from the relaxed structure(s) found for other charge
            states of that defect – in which case only the unperturbed and rattled configurations of
            the relaxed other-charge defect structure(s) are calculated.
            (Default: 'BDM')

    Returns:
        Dictionary of bond distortions and corresponding final structures.
    """
    defect_structures_dict = {}
    if distortion_increment is None and bond_distortions is None:
        try:  # Read distortion parameters from distortion_metadata.json
            with open(f"{output_path}/distortion_metadata.json") as json_file:
                distortion_parameters = json.load(json_file)["distortion_parameters"]
                bond_distortions = distortion_parameters["bond_distortions"]
                bond_distortions = [i * 100 for i in bond_distortions]
        except:
            raise FileNotFoundError(
                f"No `distortion_metadata.json` file found in {output_path}. Please specify "
                f"`distortion_increment` or `bond_distortions`."
            )
    else:  # if user specifies values
        if bond_distortions:
            bond_distortions = [i * 100 for i in bond_distortions]
        else:
            bond_distortions = np.arange(
                -60, 60.1, distortion_increment * 100
            )  # if user didn't specify bond_distortions, assume default range

    rattle_dir_path = (
        output_path
        + "/"
        + defect_species
        + "/"
        + distortion_type
        + "/"
        + defect_species
        + "_"
        + "only_rattled"
    )
    if os.path.isdir(
        rattle_dir_path
    ):  # check if rattle folder exists (if so, it means only rattling was applied with no bond
        # distortions), hence parse rattled & Unperturbed structures, not distortions)
        try:
            path = rattle_dir_path + "/vasp_gam/CONTCAR"
            defect_structures_dict["rattle"] = grab_contcar(path)
        except:
            warnings.warn(
                f"Unable to parse CONTCAR at {path}, storing as 'Not converged'"
            )
            defect_structures_dict["rattle"] = "Not converged"
    else:
        for i in bond_distortions:
            key = round(
                i / 100, 3
            )  # Dictionary key in the same format as the {distortions: final energies} dictionary
            i = f"{i+0:.1f}"  # 1 decimal place
            path = (
                output_path
                + "/"
                + defect_species
                + "/"
                + distortion_type
                + "/"
                + defect_species
                + "_"
                + str(i)
                + "%_Bond_Distortion/vasp_gam/CONTCAR"
            )
            try:
                defect_structures_dict[key] = grab_contcar(path)
            except FileNotFoundError or IndexError or ValueError:
                warnings.warn(
                    f"Unable to parse structure at {path}, storing as 'Not converged'"
                )
                defect_structures_dict[key] = "Not converged"
            except:
                warnings.warn(
                    f"Problem parsing structure at {path}, storing as 'Not converged'"
                )
                defect_structures_dict[key] = "Not converged"
    try:
        unperturbed_path = (
            output_path
            + "/"
            + defect_species
            + "/"
            + distortion_type
            + "/"
            + defect_species
            + "_"
            + "Unperturbed_Defect"
            + "/vasp_gam/CONTCAR"
        )
        defect_structures_dict["Unperturbed"] = grab_contcar(unperturbed_path)
    except FileNotFoundError:
        warnings.warn(
            f"Unable to parse structure at {unperturbed_path}, storing as 'Not converged'"
        )
        defect_structures_dict["Unperturbed"] = "Not converged"
    return defect_structures_dict


def get_energies(
    defect_species: str,
    output_path: str,
    distortion_type: str = "BDM",
    units: str = "eV",
) -> dict:
    """
    Parse final energies for each bond distortion and store them in a dictionary matching the
    bond distortion to the final energy in eV.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        output_path (:obj:`str`):
            Path to top-level directory containing `defect_species` subdirectories.
        distortion_increment (:obj:`float`):
            Bond distortion increment. Recommended values: 0.1-0.3 (Default: 0.1)
        distortion_type (:obj:`str`) :
            Type of distortion method used.
            Either 'BDM' (bond distortion method (standard)) or 'champion'. The option 'champion'
            is used when relaxing a defect from the relaxed structure(s) found for other charge
            states of that defect – in which case only the unperturbed and rattled configurations of
            the relaxed other-charge defect structure(s) are calculated.
            (Default: 'BDM')
        units (:obj:`str`):
            Energy units for outputs (either 'eV' or 'meV'). (Default: "eV")

    Returns:
        Dictionary matching bond distortions to final energies in eV.
    """
    energy_file_path = (
        f"{output_path}/{defect_species}/{distortion_type}/{defect_species}.txt"
    )
    defect_energies_dict, _e_diff, gs_distortion = sort_data(energy_file_path)
    if "Unperturbed" in defect_energies_dict:
        for distortion, energy in defect_energies_dict["distortions"].items():
            defect_energies_dict["distortions"][distortion] = (
                energy - defect_energies_dict["Unperturbed"]
            )
        defect_energies_dict["Unperturbed"] = 0.0
    else:
        warnings.warn(
            "Unperturbed defect energy not found in energies file. Energies will be "
            "given relative to the lowest energy defect structure found."
        )
        lowest_E_distortion = defect_energies_dict["distortions"][gs_distortion]
        for distortion, energy in defect_energies_dict["distortions"].items():
            defect_energies_dict["distortions"][distortion] = (
                energy - lowest_E_distortion
            )
    if units == "meV":
        defect_energies_dict["distortions"] = {
            k: v * 1000 for k, v in defect_energies_dict["distortions"].items()
        }

    return defect_energies_dict


def calculate_struct_comparison(
    defect_structures_dict: dict,
    metric: str = "max_dist",
    ref_structure: Union[str, float, Structure] = "Unperturbed",
    stol: float = 0.5,
) -> Optional[dict]:
    """
    Calculate either the summed atomic displacement normalised to the average free length per
    atom := ( V / Nsites ) ** (1/3), with metric = "disp", or the maximum distance between
    matched atoms, with metric = "max_dist", (default) between each distorted structure in
    `defect_struct_dict`, and either 'Unperturbed' or a specified structure (`ref_structure`).

    Args:
        defect_structures_dict (:obj:`dict`):
            Dictionary of bond distortions and corresponding (final) structures (as pymatgen
            Structure objects).
        metric (:obj:`str`):
            Structure comparison metric to use. Either summed atomic displacement normalised to
            the average free length per atom ('disp') or the maximum distance between matched
            atoms ('max_dist', default).
            (Default: "max_dist")
        ref_structure (:obj:`str` or :obj:`float` or :obj:`Structure`):
            Structure to use as a reference for comparison (to compute atomic displacements).
            Either as a key from `defect_structures_dict` or a pymatgen Structure object (to
            compare with a specific external structure).
            (Default: "Unperturbed")
        stol (:obj:`float`):
            Site tolerance used for structural comparison (via `pymatgen`'s `StructureMatcher`),
            as a fraction of the average free length per atom := ( V / Nsites ) ** (1/3). If
            output contains too many 'NaN' values, this likely needs to be increased.
            (Default: 0.5)

    Returns:
        disp_dict (:obj:`dict`, optional):
            Dictionary matching bond distortions to structure comparison metric (disp or
            max_dist).
    """
    if isinstance(ref_structure, str) or isinstance(ref_structure, float):
        if isinstance(ref_structure, str):
            ref_name = ref_structure
        else:
            ref_name = f"{ref_structure:.1%} bond distorted structure"
        try:
            ref_structure = defect_structures_dict[ref_structure]
        except KeyError:
            raise KeyError(
                f"Reference structure key '{ref_structure}' not found in defect_structures_dict."
            )
        if ref_structure == "Not converged":
            raise ValueError(
                f"Specified reference structure ({ref_name}) is not converged and cannot be used "
                f"for structural comparison. Check structures or specify a different reference "
                f"structure (ref_structure)."
            )
    elif isinstance(ref_structure, Structure):
        ref_name = f"specified ref_structure ({ref_structure.composition})"
    else:
        raise TypeError(
            f"ref_structure must be either a key from defect_structures_dict or a pymatgen "
            f"Structure object. Got {type(ref_structure)} instead."
        )
    print(f"Comparing structures to {ref_name}...")

    disp_dict = {}
    sm = StructureMatcher(
        ltol=0.3, stol=stol, angle_tol=5, primitive_cell=False, scale=True
    )
    for distortion in list(defect_structures_dict.keys()):
        if defect_structures_dict[distortion] != "Not converged":
            try:
                rms_disp, max_dist = sm.get_rms_dist(
                    ref_structure, defect_structures_dict[distortion]
                )
                if metric == "disp":
                    disp_dict[distortion] = rms_disp * np.sqrt(len(ref_structure))
                elif metric == "max_dist":
                    disp_dict[distortion] = max_dist
                else:
                    raise ValueError(
                        f"Invalid metric '{metric}'. Must be one of 'disp' or 'max_dist'."
                    )
            except TypeError:
                disp_dict[
                    distortion
                ] = None  # algorithm couldn't match lattices. Set comparison metric to None
                warnings.warn(
                    f"pymatgen StructureMatcher could not match lattices between {ref_name} "
                    f"and {distortion} structures."
                )
        else:
            disp_dict[distortion] = "Not converged"  # Structure not converged

    return disp_dict


# TODO: Add check if too many 'NaN' values in disp_dict, if so, try with higher stol


def compare_structures(
    defect_structures_dict: dict,
    defect_energies_dict: dict,
    ref_structure: Union[str, float, Structure] = "Unperturbed",
    stol: float = 0.5,
    units: str = "eV",
) -> pd.DataFrame:
    """
    Compare final bond-distorted structures with either 'Unperturbed' or a specified structure
    (`ref_structure`), and calculate the summed atomic displacement normalised to the average
    free length per atom := ( V / Nsites ) ** (1/3), and maximum distance between matched atomic
    sites.

    Args:
        defect_structures_dict (:obj:`dict`):
            Dictionary mapping bond distortion to (relaxed) structure
        defect_energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by `organize_data()`.
        ref_structure (:obj:`str` or :obj:`float` or :obj:`Structure`):
            Structure to use as a reference for comparison (to compute atomic displacements).
            Either as a key from `defect_structures_dict` or a pymatgen Structure object (to
            compare with a specific external structure).
            (Default: "Unperturbed")
        stol (:obj:`float`):
            Site tolerance used for structural comparison (via `pymatgen`'s `StructureMatcher`),
            as a fraction of the average free length per atom := ( V / Nsites ) ** (1/3). If
            structure comparison output contains too many 'NaN' values, this likely needs to be
            increased.
            (Default: 0.5)
        units (:obj:`str`):
            Energy units label for outputs (either 'eV' or 'meV'). Should be the same as the
            units in `defect_energies_dict`, as this does not modify the supplied values.
            (Default: "eV")

    Returns:
        DataFrame containing structural comparison results (summed normalised atomic displacement
        and maximum distance between matched atomic sites), and relative energies.
    """
    df_list = []
    disp_dict = calculate_struct_comparison(
        defect_structures_dict, metric="disp", ref_structure=ref_structure, stol=stol
    )
    with HiddenPrints():  # only print "Comparing to..." once
        max_dist_dict = calculate_struct_comparison(
            defect_structures_dict,
            metric="max_dist",
            ref_structure=ref_structure,
            stol=stol,
        )

    for distortion in defect_energies_dict["distortions"]:
        try:
            rel_energy = defect_energies_dict["distortions"][distortion]
        except KeyError:  # if relaxation didn't converge for this bond distortion, store it
            # as NotANumber
            rel_energy = float("NaN")
        df_list.append(
            [
                distortion,
                round(disp_dict[distortion], 3) + 0
                if isinstance(disp_dict[distortion], float)
                else None,
                round(max_dist_dict[distortion], 3) + 0
                if isinstance(max_dist_dict[distortion], float)
                else None,
                round(rel_energy, 2) + 0,
            ]
        )

    if "Unperturbed" in defect_energies_dict:
        df_list.append(
            [
                "Unperturbed",
                round(disp_dict["Unperturbed"], 3) + 0
                if isinstance(disp_dict["Unperturbed"], float)
                else None,
                round(max_dist_dict["Unperturbed"], 3) + 0
                if isinstance(max_dist_dict["Unperturbed"], float)
                else None,
                round(defect_energies_dict["Unperturbed"], 2) + 0,
            ]
        )

    struct_comparison_df = pd.DataFrame(
        df_list,
        columns=[
            "Bond Distortion",
            "\u03A3{Normalised Displacement}",  # Sigma
            "Max Distance (\u212B)",  # Angstrom
            f"\u0394 Energy ({units})",  # Delta
        ],
    )
    if isipython():
        display(struct_comparison_df)
    return struct_comparison_df
