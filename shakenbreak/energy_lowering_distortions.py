""" 
Functions to apply energy lowering distortions found for a certain defect species (charge state)
to other charge states of that defect.
"""
import copy
import os
import shutil
import warnings
from typing import Optional
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
aaa = AseAtomsAdaptor()

import ase

from shakenbreak.analysis import (
    _sort_data,
    _get_distortion_filename,
    get_structures,
    get_energies,
    calculate_struct_comparison,
    compare_structures,
)
from shakenbreak.io import read_vasp_structure, read_espresso_structure


def _format_distortion_directory_name(
    distorted_distortion: str,
    distorted_charge: int,
    defect_species: str,
    output_path: str,
) -> str:
    """Format name of distortion directory"""
    if (
        isinstance(distorted_distortion, str)
        and "_from_" not in distorted_distortion
    ):
        distorted_dir = (
            f"{output_path}/{defect_species}/Bond_Distortion_"
            f"{distorted_distortion}_from_{distorted_charge}"
        )
    elif (
        isinstance(distorted_distortion, str)
        and "_from_" in distorted_distortion
    ):
        distorted_dir = (
            f"{output_path}/{defect_species}/Bond_Distortion_"
            f"{distorted_distortion}"
        )
    else:
        distorted_dir = (
            f"{output_path}/{defect_species}/Bond_Distortion_"
            f"{round(distorted_distortion * 100, 1)+0}%_from_"
            f"{distorted_charge}"
        )
    return distorted_dir


def read_defects_directories(output_path: str = "./") -> dict:
    """
    Reads all defect folders in the `output_path` directory and stores defect names and charge
    states in a dictionary.

    Args:
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations.
            (Default is current directory = "./")

    Returns:
        Dictionary with defect names and charge states.
    """
    list_subdirectories = [  # Get only subdirectories in the current directory
        i
        for i in next(os.walk(output_path))[1]
        if ("as_" in i) or ("vac_" in i) or ("Int_" in i) or ("sub_" in i)
    ]  # matching doped/PyCDT/pymatgen defect names
    list_name_charge = [
        i.rsplit("_", 1) for i in list_subdirectories
    ]  # split by last "_" (separate defect name from charge state)
    defect_charges_dict = {}
    for i in list_name_charge:
        if i[0] in defect_charges_dict:
            if i[1] not in defect_charges_dict[i[0]]:  # if charge not in value
                defect_charges_dict[i[0]].append(int(i[1]))
        else:
            defect_charges_dict[i[0]] = [int(i[1])]
    return defect_charges_dict

# TODO: Update get_energy_lowering_distortions() to optionally also store non-spontaneous
#  _metastable_ energy-lowering distortions, as these can become ground-state distortions for
#  other charge states
def get_energy_lowering_distortions(
    defect_charges_dict: Optional[dict] = None,
    output_path: str = ".",
    code: str = "VASP",
    structure_filename: str = "CONTCAR",
    min_e_diff: float = 0.05,
    stol: float = 0.5,
    min_dist: float = 0.2,
    verbose: bool = True,
    write_input_files: bool = False,
) -> dict:
    """
    Convenience function to identify defect species undergoing energy-lowering distortions.
    Useful for then testing these distorted structures for the other charge states of that defect.
    Considers all identified energy-lowering distortions for each defect in each charge state,
    and screens out duplicate distorted structures found for multiple charge states.

    Args:
        defect_charges_dict (:obj:`dict`, optional):
            Dictionary matching defect name(s) to list(s) of their charge states. (e.g {
            "Int_Sb_1":[0,+1,+2]} etc). If not specified, all defects present in `output_path`
            will be parsed.
            (Default: None)
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (need CONTCAR files for
            structure matching) and distortion_metadata.json. (Default is current directory = "./")
        code (:obj:`str`, optional):
            Code used for the geometry relaxations.
            (Default: VASP)
        structure_filename (:obj:`str`, optional):
            Name of the file containing the structure.
            (Default: CONTCAR)
        min_e_diff (:obj: `float`):
            Minimum energy difference (in eV) between the ground-state defect structure,
            relative to the `Unperturbed` structure, to consider it as having found a new
            energy-lowering distortion. Default is 0.05 eV.
        stol (:obj:`float`):
            Site-matching tolerance for structure matching. Site tolerance. Defined as the
            fraction of the average free length per atom := ( V / Nsites ) ** (1/3).
            (Default: 0.5)
        min_dist (:obj:`float`):
            Minimum atomic displacement threshold between structures, in order to consider them
            not matching (in Å, default = 0.2 Å).
        verbose (:obj:`bool`):
            Whether to print verbose information about energy lowering distortions, if found.
            (Default: True)
        write_input_files (:obj:`bool`):
            Whether to write input files for the identified distortions
            (Default: False)
    Returns:
        low_energy_defects (:obj:`dict`):
            Dictionary of defects for which bond distortion found an energy-lowering distortion
            (which is missed with normal unperturbed relaxation), of the form {defect: [list of
            distortion dictionaries (with corresponding charge states, energy lowering,
            distortion factors, structures and charge states for which these structures weren't
            found)]}.
    """
    if not os.path.isdir(output_path):  # check if output_path exists
        raise FileNotFoundError(f"Path {output_path} does not exist!")

    low_energy_defects = (
        {}
    )  # dict of defects undergoing energy-lowering distortions, relative to unperturbed structure

    if not defect_charges_dict:
        defect_charges_dict = read_defects_directories(output_path=output_path)
    defect_pruning_dict = copy.deepcopy(
        defect_charges_dict
    )  # defects and charge states to analyse
    
    # later comparison and pruning against other charge states
    for defect in defect_charges_dict:
        print(f"\n{defect}")
        defect_pruning_dict[defect] = []
        for charge in defect_charges_dict[defect]:
            defect_pruning_dict[defect].append(charge)
            defect_species = f"{defect}_{charge}"
            energies_file = f"{output_path}/{defect_species}/{defect_species}.txt"
            energies_dict, energy_diff, gs_distortion = _sort_data(
                energies_file, verbose=verbose
            )

            if energies_dict is None:
                print(
                    f"No data parsed for {defect_species}. This species will be skipped and "
                    f"will not be included in the low_energy_defects charge state lists (and so "
                    f"energy lowering distortions found for other charge states will not be "
                    f"applied for this species)."
                )
                defect_pruning_dict[defect].remove(charge)

            elif (
                energy_diff and float(energy_diff) < -min_e_diff
            ):  # if a significant energy drop occurred, then store this distorted defect
                bond_distortion = _get_distortion_filename(gs_distortion)
                # format distortion label to the one used in file name
                # (e.g. from 0.1 to Bond_Distortion_10.0%)
                file_path = f"{output_path}/{defect_species}/{bond_distortion}/CONTCAR"
                with warnings.catch_warnings(record=True) as w:
                    gs_struct = read_vasp_structure(
                        file_path
                    )  # get the final structure of the
                    # energy lowering distortion
                    if any(warning.category == UserWarning for warning in w):
                        # problem parsing structure, user will have received appropriate
                        # warning from read_vasp_structure()
                        print(
                            f"Problem parsing final, low-energy structure for {gs_distortion} "
                            f"bond distortion of {defect_species} at {file_path}. This species "
                            f"will be skipped and will not be included in low_energy_defects ("
                            f"check relaxation calculation and folder)."
                        )
                        defect_pruning_dict[defect].remove(charge)
                        continue

                if (
                    defect in low_energy_defects
                ):  # Check if the lower-energy distorted structure was already found with bond
                    # distortions for a different charge state of this defect
                    comparison_dicts_dict = {}  # index: comparison_dict
                    for i in range(
                        len(low_energy_defects[defect])
                    ):  # use _initial_ list count
                        # rather than iterating directly over list, as this will result in unwanted
                        # repetition because we append to this list if new structure found
                        struct_comparison_dict = calculate_struct_comparison(
                            {"Ground State": gs_struct},
                            metric="disp",
                            ref_structure=low_energy_defects[defect][i]["structures"][
                                0
                            ],  # just select the first structure in
                            # each list as these structures have already been found to match
                            stol=stol,
                            min_dist=min_dist,
                        )
                        comparison_dicts_dict[i] = struct_comparison_dict

                    matching_distortion_dict = {
                        index: struct_comparison_dict
                        for index, struct_comparison_dict in comparison_dicts_dict.items()
                        if struct_comparison_dict["Ground State"] == 0
                    }

                    if (
                        len(matching_distortion_dict) > 0
                    ):  # if it matches _any_ other distortion
                        index = list(matching_distortion_dict.keys())[
                            0
                        ]  # should only be one
                        print(
                            f"Low-energy distorted structure for {defect_species} already found"
                            f" with charge states {low_energy_defects[defect][index]['charges']}, "
                            f"storing together."
                        )
                        low_energy_defects[defect][index]["charges"].append(charge)
                        low_energy_defects[defect][index]["structures"].append(
                            gs_struct
                        )
                        low_energy_defects[defect][index]["energy_diffs"].append(
                            energy_diff
                        )
                        low_energy_defects[defect][index]["bond_distortions"].append(
                            gs_distortion
                        )

                    else:  # only add to list if it doesn't match _any_ of the other distortions
                        # if the structure was not previously found, then add it to the list of
                        # distortions for this defect
                        print(
                            f"New (according to structure matching) low-energy distorted "
                            f"structure found for {defect_species}, adding to low_energy_defects["
                            f"'{defect}'] list."
                        )
                        low_energy_defects[defect].append(
                            {
                                "charges": [charge],
                                "structures": [gs_struct],
                                "energy_diffs": [energy_diff],
                                "bond_distortions": [gs_distortion],
                                "excluded_charges": set(),
                            }
                        )

                elif defect not in low_energy_defects:  # if defect not in dict, add it
                    print(
                        f"Energy lowering distortion found for {defect} with charge {charge}. "
                        f"Adding to low_energy_defects dictionary."
                    )
                    low_energy_defects[defect] = [
                        {
                            "charges": [charge],
                            "structures": [gs_struct],
                            "energy_diffs": [energy_diff],
                            "bond_distortions": [gs_distortion],
                            "excluded_charges": set(),
                        }
                    ]

            else:
                print(
                    f"No energy lowering distortion with energy difference greater than "
                    f"min_e_diff = {min_e_diff:.2f} eV found for {defect} with charge {charge}."
                )

    # Screen through defects to check if any lower-energy distorted structures were already
    # found with/without bond distortions for other charge states (i.e. found but higher energy,
    # found but also with unperturbed, found but with energy lowering less than min_e_diff etc)
    print("\nComparing and pruning defect structures across charge states...")
    for defect, distortion_list in low_energy_defects.items():
        for distortion_dict in distortion_list:
            for charge in list(
                set(defect_pruning_dict[defect]) - set(distortion_dict["charges"])
            ):
                # charges in defect_pruning_dict that aren't already in this distortion entry
                defect_species = f"{defect}_{charge}"
                comparison_results = compare_struct_to_distortions(
                    distortion_dict["structures"][0],
                    defect_species,
                    output_path,
                    code=code,
                    structure_filename=structure_filename,
                    stol=stol,
                    min_dist=min_dist,
                )
                if comparison_results[0]:
                    # structure found in distortion tests for this charge state. Add it to the
                    # list to avoid redundant work
                    print(
                        f"Ground-state structure found for {defect} with charges "
                        f"{distortion_dict['charges']} has been also previously been found for "
                        f"charge state {charge} (according to structure matching). Adding this "
                        f"charge to the corresponding entry in low_energy_defects[{defect}]."
                    )
                    distortion_dict["charges"].append(charge)
                    distortion_dict["structures"].append(comparison_results[1])
                    distortion_dict["energy_diffs"].append(comparison_results[2])
                    distortion_dict["bond_distortions"].append(comparison_results[3])
                elif comparison_results[0] is False:
                    distortion_dict["excluded_charges"].add(charge)
                elif comparison_results[0] is None:
                    print(
                        f"Problem parsing structures for {defect_species}. This species will be "
                        f"skipped and will not be included in low_energy_defects (check relaxation "
                        f"folders with CONTCARs are present)."
                    )
    
    # Write input files for the identified distortions
    if write_input_files:
        write_distorted_inputs(
            low_energy_defects=low_energy_defects,
            output_path=output_path,
            code=code,
        )
    return low_energy_defects


def compare_struct_to_distortions(
    distorted_struct: Structure,
    defect_species: str,
    output_path: str = ".",
    code: str="VASP",
    structure_filename: str="CONTCAR",
    stol: float = 0.5,
    min_dist: float = 0.2,
) -> tuple:
    """
    Compares the ground-state structure found for a certain defect charge state with all
    relaxed bond-distorted structures for `defect_species`, to avoid redundant work (testing
    this distorted structure for other charge states when it has already been found for them).

    Args:
        distorted_struct (:obj:`~pymatgen.core.structure.Structure`):
            Structure of ground-state distorted defect
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to calculate structure
            comparisons – needs VASP CONTCAR files). (Default is current directory = "./")
        code (:obj:`str`, optional):
            Code used for the geometry relaxations.
            (Default: VASP)
        structure_filename (:obj:`str`, optional):
            Name of the file containing the structure.
            (Default: CONTCAR)
        stol (:obj:`float`):
             Site-matching tolerance for structure matching. Site tolerance. Defined as the
             fraction of the average free length per atom := ( V / Nsites ) ** (1/3).
             (Default: 0.5)
        min_dist (:obj:`float`):
            Minimum atomic displacement threshold between structures, in order to consider them
            not matching (in Å, default = 0.2 Å).

    Returns:
        (True/False/None, matching structure, energy difference of the matching structure compared
        to its unperturbed reference, bond distortion of the matching structure). True if a match
        is found between the input structure and the relaxed bond-distorted structures for
        `defect_species`, False if no match, None if no converged structures found for
        defect_species.
    """
    try:
        defect_structures_dict = get_structures(
            defect_species=defect_species, 
            output_path=output_path,
            code=code,
            structure_filename=structure_filename,
        )
    except FileNotFoundError:  # catch exception raised by `get_structures`` if `defect_species`
        # folder does not exist
        # print(
        #     f"No structures found for {defect_species}. Returning None. Check that the "
        #     f"relaxation folders for {defect_species} are present in {output_path}."
        # )
        return None, None, None, None
    defect_energies_dict = get_energies(
        defect_species=defect_species, output_path=output_path, verbose=False
    )

    struct_comparison_df = compare_structures(
        defect_structures_dict=defect_structures_dict,
        defect_energies_dict=defect_energies_dict,
        ref_structure=distorted_struct,
        stol=stol,
        min_dist=min_dist,
        display_df=False,
    )
    if struct_comparison_df is None:  # no converged structures found for defect_species
        return None, None, None, None

    matching_sub_df = struct_comparison_df[
        struct_comparison_df["Σ{Displacements} (Å)"] == 0
    ]

    if not matching_sub_df.empty:  # if there are any matches
        unperturbed_df = matching_sub_df[
            matching_sub_df["Bond Distortion"]
            == "Unperturbed"  # if present, otherwise empty
        ]
        rattled_df = matching_sub_df[
            matching_sub_df["Bond Distortion"]
            == "Rattled"  # if present, otherwise empty
        ]
        sorted_distorted_df = matching_sub_df[
            matching_sub_df["Bond Distortion"].apply(lambda x: isinstance(x, float))
        ].sort_values(by="Bond Distortion", key=abs)  # if present, otherwise empty

        string_vals_sorted_distorted_df = matching_sub_df[
            matching_sub_df["Bond Distortion"].apply(lambda x: isinstance(x, str))
        ]
        imported_sorted_distorted_df = string_vals_sorted_distorted_df[
            string_vals_sorted_distorted_df["Bond Distortion"].apply(lambda x: "_from_" in x)
        ]

        if not imported_sorted_distorted_df.empty:
            # convert "X%_from_Y" strings to floats and then sort
            # needs to be done this way because 'key' in pd.sort_values() needs to be vectorised...
            s = imported_sorted_distorted_df['Bond Distortion'].str.slice(0, 3)
            s = s.astype(float)
            imported_sorted_distorted_df = imported_sorted_distorted_df.loc[
                s.sort_values(key=lambda x: abs(x)).index]

        # first unperturbed, then rattled, then distortions sorted by initial distortion magnitude
        # from low to high (if present)
        sorted_matching_df = pd.concat(
            [unperturbed_df, rattled_df, sorted_distorted_df, imported_sorted_distorted_df]
        )

        if sorted_matching_df.empty:  # TODO: Add test for this
            raise KeyError(f"Unrecognized label in parsed structures:"
                           f" {matching_sub_df['Bond Distortion'].values}")

        struc_key = sorted_matching_df["Bond Distortion"].iloc[
            0
        ]  # first matching structure
        if struc_key == "Unperturbed":
            return (  # T/F, matching structure, energy_diff, distortion factor
                True,
                defect_structures_dict[struc_key],
                defect_energies_dict[struc_key],
                struc_key,
            )
        else:
            return (  # T/F, matching structure, energy_diff, distortion factor
                True,
                defect_structures_dict[struc_key],
                defect_energies_dict["distortions"][struc_key],
                struc_key,
            )

    else:
        return (
            False,
            None,
            None,
            None,
        )  # T/F, matching structure, energy_diff, distortion factor


def write_distorted_inputs(
    low_energy_defects: dict, 
    output_path: str = ".",
    code: str = "VASP",   
    input_filename: str = None, 
) -> None:
    """
    Create folders with VASP input files for testing the low-energy distorted defect structures
    found for other charge states of that defect, as identified with
    `get_energy_lowering_distortions()`.

    Args:
        low_energy_defects (:obj:`dict`):
             Dictionary of defects for which bond distortion found an energy-lowering distortion
             which is missed with normal unperturbed relaxation), generated by
             `get_energy_lowering_distortions()`. Has the form {defect: [list of distortion
             dictionaries (with corresponding charge states, energy lowering, distortion factors,
             structures and charge states for which these structures weren't found)]}.
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to write input files for
            distorted defect structures to test). (Default is current directory = "./")
        code (:obj:`str`):
            Code used for the geometry relaxations. The supported codes include "VASP", 
            "CP2K", "espresso", "CASTEP" and "FHI-aims".
            (Default: "VASP")
        input_filename (:obj:`str`):
            Name of the code input file if different from `ShakeNBreak` default. Only applies
            to CP2K, Quantum Espresso, CASTEP and FHI-aims. If not specified, `ShakeNBreak` default
            name is assumed, that is: for Quantum Espresso: "espresso.pwi", CP2K: "cp2k_input.inp",
            CASTEP: "castep.param", FHI-aims: "control.in"
            (Default: None)
    Returns:
        None
    """
    for defect, distortion_list in low_energy_defects.items():
        for distortion_dict in distortion_list:
            for charge in distortion_dict[
                "excluded_charges"
            ]:  # charges for which this distortion wasn't found
                defect_species = f"{defect}_{charge}"
                distorted_charge = distortion_dict["charges"][
                    0
                ]  # first charge state for which this distortion was found
                distorted_structure = distortion_dict["structures"][
                    0
                ]  # first structure for which this distortion was found
                distorted_distortion = distortion_dict["bond_distortions"][
                    0
                ]  # first bond distortion for which this distortion was found

                distorted_dir = _format_distortion_directory_name(
                    distorted_distortion=distorted_distortion,
                    distorted_charge=distorted_charge,
                    output_path=output_path,
                    defect_species=defect_species,
                ) # Format distoriton directory name

                if os.path.exists(distorted_dir):
                    print(
                        f"As {distorted_dir} already exists, it's assumed this structure "
                        f"has already been tested. Skipping..."
                    )
                    continue

                print(f"Writing low-energy distorted structure to {distorted_dir}")

                if not os.path.exists(f"{output_path}/{defect_species}"):
                    print(
                        f"Directory {output_path}/{defect_species} not found, creating..."
                    )
                    os.mkdir(f"{output_path}/{defect_species}")
                os.mkdir(distorted_dir)
                
                # copy input files from Unperturbed directory
                if code == "VASP":
                    _copy_vasp_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                    )
                elif code == "espresso":
                    _copy_espresso_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )
                elif code == "CP2K":
                    _copy_cp2k_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )
                elif code == "CASTEP":
                    _copy_castep_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )
                elif code == "FHI-aims":
                    _copy_fhi_aims_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )


def _copy_vasp_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
) -> None:
    """
    Copy VASP input files from an existing Distortion directory
    to a new directory.
    """
    distorted_structure.to(fmt="poscar", filename=f"{distorted_dir}/POSCAR")

    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/INCAR"):
        for i in ["INCAR", "KPOINTS", "POTCAR"]:
            shutil.copyfile(
                f"{output_path}/{defect_species}/Unperturbed/{i}",
                f"{distorted_dir}/{i}",
            ) # copy input files from Unperturbed directory
    else:
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/INCAR"
            ):
                subfolders_with_input_files.append(subfolder)
        if len(subfolders_with_input_files) > 0:
            for i in ["INCAR", "KPOINTS", "POTCAR"]:
                shutil.copyfile(
                    f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}/"
                    f"{i}",
                    f"{distorted_dir}/{i}",
                )
        else:
            print(
                f"No subfolders with VASP input files found in "
                f"{output_path}/{defect_species}, so just writing distorted POSCAR "
                f"file to {distorted_dir} directory."
            )
    

def _copy_espresso_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "espresso.pwi",
) -> None:
    """
    Copy Quantum Espresso input files from an existing Distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "espresso.pwi"
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        # Parse input parameters from file and update structural info with 
        # new distorted structure
        # ase/pymatgen dont support this
        with open(f"{output_path}/{defect_species}/Unperturbed/{input_filename}") as f:
            params=f.read() # Read input parameters
        # Write distorted structure in QE format, to then update input file 
        atoms = aaa.get_atoms(distorted_structure)
        ase.io.write(filename=f"{distorted_dir}/{input_filename}", images=atoms, format="espresso-in")
        with open(f"{distorted_dir}/{input_filename}") as f:
            new_struct=f.read() 
        params = params.replace(
            params[params.find("ATOMIC_POSITIONS"):], 
            new_struct[new_struct.find("ATOMIC_POSITIONS"):], 
            1
        ) # Replace ionic positions
        with open(f"{distorted_dir}/{input_filename}", "w") as f:
            f.write(params)
    else:
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            with open(f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}/"
                      "{input_filename}")as f:
                params=f.read() # Read input parameters
            # Write distorted structure in QE format, to then update input file
            atoms = aaa.get_atoms(distorted_structure)
            ase.io.write(filename=f"{distorted_dir}/{input_filename}", images=atoms, format="espresso-in")
            with open(f"{distorted_dir}/{input_filename}") as f:
                new_struct=f.read() 
            params = params.replace(
                params[params.find("ATOMIC_POSITIONS"):], 
                new_struct[new_struct.find("ATOMIC_POSITIONS"):], 
                1
            ) # Replace lines with the ionic positions
            with open(f"{distorted_dir}/{input_filename}", "w") as f:
                f.write(params)
        else: # only write input structure
            print(
                f"No subfolders with Quantum Espresso input file (`{input_filename}`) found in "
                f"{output_path}/{defect_species}, so just writing distorted structure "
                f"file to {distorted_dir} directory."
            )
            atoms = aaa.get_atoms(distorted_structure)
            ase.io.write(filename=f"{distorted_dir}/{input_filename}", images=atoms, format="espresso-in")


def _copy_cp2k_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "cp2k_input.inp",
) -> None:
    """
    Copy CP2K input files from an existing Distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "cp2k_input.inp"
    distorted_structure.to('cif', f"{distorted_dir}/structure.cif")
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        shutil.copyfile(
            f"{output_path}/{defect_species}/Unperturbed/{input_filename}",
            f"{distorted_dir}/{input_filename}",
        )
    else: # Check of input file present in the other distortion subfolders
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            shutil.copyfile(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}/{input_filename}",
                f"{distorted_dir}/{input_filename}",
            )
            
        else: # only write input structure
            print(
                f"No subfolders with CP2K input file (`cp2k_input.inp`) found in "
                f"{output_path}/{defect_species}, so just writing distorted structure "
                f"file to {distorted_dir} directory (in CIF format)."
            )
            

def _copy_castep_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "castep.param",
) -> None:
    """
    Copy CASTEP input files from an existing Distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "castep.param"
    atoms = aaa.get_atoms(distorted_structure)
    ase.io.write(
                filename=f"{distorted_dir}/castep.cell", 
                images=atoms, 
                format="castep-cell"
                ) # Write structure
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        shutil.copyfile(
            f"{output_path}/{defect_species}/Unperturbed/{input_filename}",
            f"{distorted_dir}/{input_filename}",
        )
    else: # Check of input file present in the other distortion subfolders
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            shutil.copyfile(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}/{input_filename}",
                f"{distorted_dir}/{input_filename}",
            )
            
        else: # only write input structure
            print(
                f"No subfolders with CASTEP input file (`{input_filename}`) found in "
                f"{output_path}/{defect_species}, so just writing distorted structure "
                f"file to {distorted_dir} directory (in CASTEP `.cell` format)."
            )

 
def _copy_fhi_aims_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str ="control.in",
) -> None:
    """
    Copy FHI-aims input files from an existing Distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "control.in"
    atoms = aaa.get_atoms(distorted_structure)
    ase.io.write(
            filename=f"{distorted_dir}/geometry.in", 
            images=atoms, format="aims",
            ) # write input structure file
    
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        shutil.copyfile(
            f"{output_path}/{defect_species}/Unperturbed/{input_filename}",
            f"{distorted_dir}/{input_filename}",
        )
    else: # Check of input file present in the other distortion subfolders
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            shutil.copyfile(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}/{input_filename}",
                f"{distorted_dir}/{input_filename}",
            )
            
        else: # only write input structure
            print(
                f"No subfolders with FHI-aims input file (`{input_filename}`) found in "
                f"{output_path}/{defect_species}, so just writing distorted structure "
                f"file to {distorted_dir} directory (in FHI-aims `geometry.in` format)."
            )           
       

def write_groundstate_structure(
    output_path: str = ".",
    groundstate_filename: str = "groundstate_POSCAR",
    structure_filename: str = "CONTCAR",
) -> None:
    """
    Writes the groundstate structure of each defect to the corresponding 
    defect folder, with an optional name (default "groundstate_POSCAR"), to then run 
    continuation calculations.
    Args:
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (need CONTCAR files for
            structure matching) and distortion_metadata.json. 
            (Default: current directory = "./")
        groundstate_filename (:obj:`str`):
            Name of the file to write the groundstate structure to. 
            (Default: "groundstate_POSCAR")
        structure_filename (:obj:`str`):
            Name of the file to read the structure from.
            (Default: "CONTCAR")
    Returns:
        None
    """
    defect_charges_dict = read_defects_directories(output_path=output_path)
    for defect_species in defect_charges_dict:
        for charge in defect_charges_dict[defect_species]:
            energies_file = f"{output_path}/{defect_species}_{charge}/{defect_species}_{charge}.txt"
            _, _, gs_distortion = _sort_data(energies_file=energies_file, verbose=False)
            bond_distortion = _get_distortion_filename(gs_distortion)
            shutil.copyfile(
                f"{output_path}/{defect_species}/{bond_distortion}/{structure_filename}",
                f"{output_path}/{defect_species}/{groundstate_filename}",
            )