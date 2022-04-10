"""
Module containing functions to analyse rattled and bond-distorted defect structure relaxations
@author: Irea Mosquera
"""

import json
import os
from copy import deepcopy
from typing import Optional

import pandas as pd
from IPython.display import display
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

crystalNN = CrystalNN(
    distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False, search_cutoff=5
)


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
    energies_dict = {"distortions": {}}
    for i in range(len(distortion_list) // 2):
        i *= 2
        if "rattle" in distortion_list[i]:
            key = "rattled"
            energies_dict["distortions"][key] = float(distortion_list[i + 1])
        else:
            if "Unperturbed" in distortion_list[i]:
                energies_dict["Unperturbed"] = float(distortion_list[i + 1])
            else:
                key = distortion_list[i].split("_BDM")[0].split("%")[0]
                key = float(key.split("_")[-1]) / 100  # from % to decimal
                if key == -0.0:
                    key = 0.0
                energies_dict["distortions"][key] = float(distortion_list[i + 1])
    sorted_dict = {"distortions": {}, "Unperturbed": energies_dict["Unperturbed"]}
    for key in sorted(
        energies_dict["distortions"].keys()
    ):  # Order dict items by key (e.g. from -0.6 to 0 to +0.6)
        sorted_dict["distortions"][key] = energies_dict["distortions"][key]
    return sorted_dict


def get_gs_distortion(energies_dict: dict):
    """
    Calculate energy difference between `Unperturbed` structure and lowest energy distortion.
    Returns the energy (in eV) and bond distortion of the ground-state relative to `Unperturbed`.

    Args:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy, as produced by `organize_data()`.

    Returns:
        (Energy difference, ground state bond distortion)
    """
    if len(energies_dict["distortions"]) == 1:
        energy_diff = (
            energies_dict["distortions"]["rattled"] - energies_dict["Unperturbed"]
        )
        if energy_diff < 0:
            gs_distortion = "rattled"  # just rattle (no bond distortion)
        else:
            gs_distortion = "Unperturbed"
    else:
        lowest_E_RBDM = min(
            energies_dict["distortions"].values()
        )  # lowest energy obtained with RBDM
        energy_diff = lowest_E_RBDM - energies_dict["Unperturbed"]
        if (
            lowest_E_RBDM < energies_dict["Unperturbed"]
        ):  # if energy lower than Unperturbed
            gs_distortion = list(energies_dict["distortions"].keys())[
                list(energies_dict["distortions"].values()).index(lowest_E_RBDM)
            ]  # BDM distortion that led to ground-state
        else:
            gs_distortion = "Unperturbed"

    return energy_diff, gs_distortion


def sort_data(energies_file: str):
    """
    Organize bond distortion results in a dictionary, calculate energy of ground-state defect
    structure relative to `Unperturbed` structure (in eV) and its corresponding bond distortion,
    and return all three as a tuple.

    Args:
        energies_file (:obj:`str`):
            Path to txt file with bond distortions and final energies (in eV), obtained using
            `BDM_parsing_script.sh`.

    Returns:
        energies_dict (:obj:`dict`):
            Dictionary matching distortion to final energy, as produced by `organize_data()`
        energy_diff (:obj:`float`):
            Energy difference between minimum energy structure and `Unperturbed` (in eV)
        gs_distortion (:obj:`float`):
            Distortion corresponding to the minimum energy structure
    """
    energies_dict = organize_data(open_file(energies_file))
    energy_diff, gs_distortion = get_gs_distortion(energies_dict)
    if energy_diff < -0.1:
        defect_name = energies_file.split("/")[-1].split(".txt")[0]
        print(
            f"{defect_name}: E diff. between minimum found with {gs_distortion} RBDM and "
            f"unperturbed: {energy_diff:+.2f} eV.\n"
        )
    return energies_dict, energy_diff, gs_distortion


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
        print(f"{abs_path_formatted} file doesn't exist. Check path & relaxation")
    try:
        struct = Structure.from_file(abs_path_formatted)
    except (FileNotFoundError or IndexError or ValueError):
        struct = "Not converged"
        print(f"Path to structure {abs_path_formatted}")
    except:
        print(f"Problem obtaining structure from: {abs_path_formatted}.")
        struct = "Not converged"
    return struct


def analyse_defect_site(
    name: str,
    structure: Structure,
    site_num: Optional[int] = None,
    vac_site: Optional[list] = None,
):
    """
    Analyse coordination environment and bond distances to nearest neighbours of defect site.

    Args:
        name (:obj:`str`):
            Defect name
        structure (:obj:`~pymatgen.core.structure.Structure`):
            `pymatgen` Structure object to analyse
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
    display(bond_length_df)
    print()
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
    distortion_metadata.json file generated with defect-finder to read info about defect site.
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

    defect_site = distortion_metadata["defects"][defect_name_without_charge][
        "defect_index"
    ]  # VASP indexing (starts counting from 1)
    if not defect_site:  # for vacancies, get fractional coordinates
        defect_frac_coords = distortion_metadata["defects"][defect_name_without_charge][
            "unique_site"
        ]
        return analyse_defect_site(
            defect_species, structure, vac_site=defect_frac_coords
        )
    return analyse_defect_site(defect_species, structure, site_num=defect_site)


def compare_structures(
    defect_dict: dict,
    defect_energies: dict,
    compare_to: str = "Unperturbed",
    ref_structure: Optional[Structure] = None,
    stol: float = 0.5,
    units: str = "eV",
) -> pd.DataFrame:
    """
    Compare final bond-distorted structures with either 'Unperturbed' or a specified structure (
    `ref_structure`), and calculate the root-mean-squared-displacement (RMS disp.) and maximum
    distance between matched atomic sites.

    Args:
        defect_dict (:obj:`dict`):
            Dictionary mapping bond distortion to (relaxed) structure
        defect_energies (:obj:`dict`):
            Dictionary matching distortion to final energy (eV), as produced by `organize_data()`.
        compare_to (:obj:`str`):
            Name of reference structure used for comparison (recommended to compared with relaxed
            'Unperturbed' defect structure).
            (Default: "Unperturbed")
        ref_structure:
            Structure used as reference structure for comparison. This allows the user to compare
            final bond-distorted structures with a specific external structure not obtained using
            `defect-finder`.
            (Default: None)
        stol (:obj:`float`):
            Site tolerance used for structural comparison (via `pymatgen`'s `StructureMatcher`).
            (Default: 0.5 Angstrom)
        units (:obj:`str`):
            Energy units for outputs (either 'eV' or 'meV'). (Default: "eV")

    Returns:
        DataFrame containing structural comparison results (
        RMS displacement and maximum distance between matched atomic sites), and relative energies.
    """
    print(f"Comparing structures to {compare_to}...")

    rms_list = []
    if (
        ref_structure
    ):  # if we give an external structure (not obtained with `defect-finder`)
        norm_struct = ref_structure
    else:  # else we take reference structure from defect dictionary
        norm_struct = defect_dict[compare_to]
    assert norm_struct

    distortion_list = list(defect_energies["distortions"].keys())
    distortion_list.append("Unperturbed")
    for distortion in distortion_list:
        if distortion == "Unperturbed":
            rel_energy = defect_energies[distortion]
        else:
            try:
                rel_energy = defect_energies["distortions"][distortion]
            except KeyError:  # if relaxation didn't converge for this BDM distortion, store it
                # as NotANumber
                rel_energy = float("nan")
        struct = defect_dict[distortion]
        if (
            struct
            and struct != "Not converged"
            and norm_struct
            and norm_struct != "Not converged"
        ):
            new_sm = StructureMatcher(
                ltol=0.3, stol=stol, angle_tol=5, primitive_cell=False, scale=True
            )  # higher stol for calculating rms
            try:
                rms_displacement = round(new_sm.get_rms_dist(norm_struct, struct)[0], 3)
                rms_dist_sites = round(
                    new_sm.get_rms_dist(norm_struct, struct)[1], 3
                )  # select rms displacement normalized by (Vol / nsites) ** (1/3)
            except TypeError:  # lattices didn't match
                rms_displacement = None
                rms_dist_sites = None
            rms_list.append(
                [distortion, rms_displacement, rms_dist_sites, round(rel_energy, 2)]
            )
    display(
        pd.DataFrame(
            rms_list, columns=["BDM Dist.", "rms", "max. dist (A)", f"Rel. E ({units})"]
        )
    )
    return pd.DataFrame(
        rms_list, columns=["BDM Dist.", "rms", "max. dist (A)", f"Rel. E ({units})"]
    )


############################################################################


def get_structures(
    defect_species: str,
    output_path: str,
    distortion_increment: float = 0.1,
    bond_distortions: Optional[list] = None,
    distortion_type="BDM",
) -> dict:
    """
    Import all structures found with rattling & bond distortions, and store them in a dictionary
    matching the bond distortion to the final structure.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        output_path (:obj:`str`):
            Path to top-level directory containing `defect_species` subdirectories.
        distortion_increment (:obj:`float`):
            Bond distortion increment. Recommended values: 0.1-0.3 (Default: 0.1)
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
    try:
        # Read distortion parameters from distortion_metadata.json
        with open(f"{output_path}/distortion_metadata.json") as json_file:
            distortion_parameters = json.load(json_file)["distortion_parameters"]
            bond_distortions = distortion_parameters["bond_distortions"]
            bond_distortions = [i * 100 for i in bond_distortions]
    except:  # if there's not a distortion metadata file
        if bond_distortions:
            bond_distortions = [i * 100 for i in bond_distortions]
        else:
            bond_distortions = range(
                -60, 69, distortion_increment * 100
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
            print(f"Unable to parse CONTCAR at {path}")
            defect_structures_dict["rattle"] = "Not converged"
    else:
        for i in bond_distortions:
            key = (
                i / 100
            )  # Dictionary key in the same format as the {distortions: final energies} dictionary
            i = f"{i:.1f}"  # 1 decimal place
            if i == "0.0":
                i = "-0.0"  # this is the format used in defect folder names
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
                + "%_BDM_Distortion/vasp_gam/CONTCAR"
            )
            try:
                defect_structures_dict[key] = grab_contcar(path)
            except FileNotFoundError or IndexError or ValueError:
                print("Error grabbing structure.")
                print("Your defect path is: ", path)
                defect_structures_dict[key] = "Not converged"
            except:
                print("Problem in get_structures")
                print("Your defect path is: ", path)
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
        print("Unable to parse CONTCAR at", unperturbed_path)
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
    energy_file_path = f"{output_path}/{defect_species}/{distortion_type}/{defect_species}.txt"
    energies_dict = sort_data(energy_file_path)[0]  # TODO: Add try except warning here
    for distortion, energy in energies_dict["distortions"].items():
        energies_dict["distortions"][distortion] = energy - energies_dict["Unperturbed"]
    energies_dict["Unperturbed"] = 0.0
    if units == "meV":
        energies_dict["distortions"] = {k: v * 1000 for k, v in energies_dict[
            "distortions"].items()}

    return energies_dict


def calculate_struct_comparison(
    defect_struct_dict: dict,
    metric: str = "max_dist",
) -> Optional[dict]:
    """
    Calculate either the root-mean-squared displacement (RMS disp.)(with metric = "rms") or the
    maximum distance between matched atoms (with metric = "max_dist", default) between each
    distorted structure in `defect_struct_dict`, and the Unperturbed structure.

    Args:
        defect_struct_dict (:obj:`dict`):
            Dictionary of bond distortions and corresponding (final) structures (as pymatgen
            Structure objects).
        metric (:obj:`str`):
            Structure comparison metric to use. Either root-mean-squared displacement ('rms') or
            the maximum distance between matched atoms ('max_dist', default).
            (Default: "max_dist")

    Returns:
        rms_dict (:obj:`dict`, optional):
            Dictionary matching bond distortions to structure comparison metric (rms or
            max_dist). Will return None if the comparison metric couldn't be calculated for more
            than 5 distortions.
    """
    rms_dict = {}
    metric_dict = {"rms": 0, "max_dist": 1}
    sm = StructureMatcher(
        ltol=0.3, stol=0.5, angle_tol=5, primitive_cell=False, scale=True
    )
    for distortion in list(defect_struct_dict.keys()):
        if defect_struct_dict[distortion] != "Not converged":
            try:
                rms_dict[distortion] = sm.get_rms_dist(
                    defect_struct_dict["Unperturbed"], defect_struct_dict[distortion]
                )[metric_dict[metric]]
            except TypeError:
                rms_dict[
                    distortion
                ] = None  # algorithm couldn't match lattices. Set comparison metric to None
        else:
            rms_dict[distortion] = "Not converged"  # Structure not converged

    if (
        sum(value is None for value in rms_dict.values()) > 5
    ):  # If metric couldn't be calculated for more than 5 distortions, then return None
        return None
    return rms_dict
# TODO: Why cutoff of 5 here? If the user uses a coarser mesh, or only some of the calculations
#  converge, is there anything wrong with printing the comparison metric info for just the small
#  set that did finish ok?
