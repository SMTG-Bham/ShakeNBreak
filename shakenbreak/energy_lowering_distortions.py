"""
Module to apply energy lowering distortions found for a certain defect
species (charge state) to other charge states of that defect.
"""
import copy
import os
import shutil
import warnings
from typing import Optional
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase.io import write as ase_write

from shakenbreak import io, analysis

aaa = AseAtomsAdaptor()


def _format_distortion_directory_name(
    distorted_distortion: str,
    distorted_charge: int,
    defect_species: str,
    output_path: str,
) -> str:
    """Format name of distortion directory."""
    if isinstance(distorted_distortion, str) and "_from_" not in distorted_distortion:
        distorted_dir = (
            f"{output_path}/{defect_species}/Bond_Distortion_"
            f"{distorted_distortion}_from_{distorted_charge}"
        )
    elif isinstance(distorted_distortion, str) and "_from_" in distorted_distortion:
        distorted_dir = (
            f"{output_path}/{defect_species}/Bond_Distortion_" f"{distorted_distortion}"
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
    Reads all defect folders in the `output_path` directory and stores defect
    names and charge states in a dictionary.

    Args:
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations.
            (Default is current directory = "./")

    Returns:
        :obj:`dict`:
            Dictionary mapping defect names to a list of its charge states.
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
        try:
            if i[0] in defect_charges_dict:
                if i[1] not in defect_charges_dict[i[0]]:  # if charge not in value
                    defect_charges_dict[i[0]].append(int(i[1]))
            else:
                defect_charges_dict[i[0]] = [int(i[1])]
        except ValueError:
            print(
                f"{i[0]}_{i[1]} not recognised as a valid defect name (should "
                f"end with charge e.g. 'vac_1_Cd_-2'), skipping..."
            )
    return defect_charges_dict


def _compare_distortion(
    defect: str,
    defect_species: str,
    charge: int,
    energy_diff: float,
    gs_distortion: float,
    gs_struct: Structure,
    low_energy_defects: dict,
    stol: float = 0.5,
    min_dist: float = 0.2,
) -> dict:
    """
    Compare the ground state distortion (`gs_distortion`) to the other
    favourable distortions stored in `low_energy_defects`. If different,
    add distortion as separate entry to `low_energy_defects`.
    If same, store together with the other similar distortions.

    Args:
        defect (:obj:`str`):
            Name of the defect, without charge state
        defect_species (:obj:`str`):
            Name of the defect, with charge state
        charge (:obj:`int`):
            Defect charge state
        energy_diff (:obj:`float`):
            Energy difference between the distortion and the associated
            Unperturbed structure.
        gs_distortion (:obj:`float`):
            Distortion factor leading to the ground state configuration.
        gs_struct (:obj:`Structure`):
            pymatgen Structure object of the ground state configuration.
        low_energy_defects (:obj:`dict):
            Dictionary storing all unique, energy-lowering distortions.
        stol (:obj:`float`):
            Site-matching tolerance for structure matching. Site
            tolerance defined as the fraction of the average free length
            per atom := ( V / Nsites ) ** (1/3).
            (Default: 0.5)
        min_dist (:obj:`float`):
            Minimum atomic displacement threshold between structures, in
            order to consider them not matching (in Å, default = 0.2 Å).

    Returns:
        :obj:`dict`
    """
    comparison_dicts_dict = {}  # index: comparison_dict
    for i in range(len(low_energy_defects[defect])):  # use _initial_ list count
        # rather than iterating directly over list, as this will
        # result in unwanted repetition because we append to
        # this list if new structure found
        struct_comparison_dict = analysis.calculate_struct_comparison(
            {"Ground State": gs_struct},
            metric="disp",
            ref_structure=low_energy_defects[defect][i]["structures"][
                0
            ],  # just select the first structure in
            # each list as these structures have already been
            # found to match
            stol=stol,
            min_dist=min_dist,
        )
        comparison_dicts_dict[i] = struct_comparison_dict

    matching_distortion_dict = {
        index: struct_comparison_dict
        for index, struct_comparison_dict in comparison_dicts_dict.items()
        if struct_comparison_dict["Ground State"] == 0
    }

    if len(matching_distortion_dict) > 0:  # if it matches _any_ other distortion
        index = list(matching_distortion_dict.keys())[0]  # should only be one
        print(
            f"Low-energy distorted structure for {defect_species} "
            f"already found with charge states "
            f"{low_energy_defects[defect][index]['charges']}, "
            f"storing together."
        )
        # Store together the info of all distortions leading to the same structure
        for property, value in zip(
            ["charges", "structures", "energy_diffs", "bond_distortions"],
            [charge, gs_struct, energy_diff, gs_distortion],
        ):
            low_energy_defects[defect][index][property].append(value)

    else:  # only add to list if it doesn't match _any_ of the
        # other distortions and the structure was not previously
        # found, then add it to the list of distortions for this
        # defect
        print(
            f"New (according to structure matching) low-energy "
            f"distorted  structure found for {defect_species}, "
            f"adding to low_energy_defects['{defect}'] list."
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
    return low_energy_defects


def _prune_dict_across_charges(
    low_energy_defects: dict,
    defect_pruning_dict: dict,
    code: str = "VASP",
    structure_filename: str = "CONTCAR",
    output_path: str = ".",
    stol: float = 0.5,
    min_dist: float = 0.2,
) -> dict:
    """
    Screen through defects to check if any lower-energy distorted structures
    were already found with/without bond distortions for other charge states
    (i.e. found but higher energy, found but also with unperturbed, found
    but with energy lowering less than min_e_diff etc)

    Args:
        low_energy_defects (dict):
            Dictionary storing all unique, energy-lowering distortions.
        defect_pruning_dict (dict):
            Dictionary with defects and charge states to analyse.
        output_path (:obj:`str`)::
            Path to directory with your distorted defect calculations
            (need CONTCAR files for structure matching) and
            distortion_metadata.json.
            (Default is current directory = "./")
        code (:obj:`str`, optional):
            Code used for the geometry relaxations. Supported code names are:
            "vasp", "espresso", "cp2k" and "fhi-aims" (case insensitive).
            (Default: "vasp")
        structure_filename (:obj:`str`, optional):
            Name of the file containing the structure.
            (Default: CONTCAR)
        stol (:obj:`float`):
            Site-matching tolerance for structure matching. Site
            tolerance defined as the fraction of the average free length
            per atom := ( V / Nsites ) ** (1/3).
            (Default: 0.5)
        min_dist (:obj:`float`):
            Minimum atomic displacement threshold between structures, in
            order to consider them not matching (in Å, default = 0.2 Å).

    Returns:
        :obj:`dict`
    """
    for defect, distortion_list in low_energy_defects.items():
        for distortion_dict in distortion_list:
            for charge in list(
                set(defect_pruning_dict[defect]) - set(distortion_dict["charges"])
            ):
                # charges in defect_pruning_dict that aren't already in this
                # distortion entry
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
                    # structure found in distortion tests for this charge state.
                    # Add it to the list to avoid redundant work
                    print(
                        f"Ground-state structure found for {defect} with charges "
                        f"{distortion_dict['charges']} has been also previously "
                        f"been found for charge state {charge} (according to "
                        f"structure matching). Adding this charge to the "
                        f"corresponding entry in low_energy_defects[{defect}]."
                    )
                    for property, value in zip(
                        ["charges", "structures", "energy_diffs", "bond_distortions"],
                        [
                            charge,
                            comparison_results[1],
                            comparison_results[2],
                            comparison_results[3],
                        ],
                    ):
                        distortion_dict[property].append(copy.deepcopy(value))
                elif comparison_results[0] is False:
                    distortion_dict["excluded_charges"].add(charge)
                elif comparison_results[0] is None:
                    print(
                        f"Problem parsing structures for {defect_species}. "
                        f"This species will be skipped and will not be included "
                        f"in low_energy_defects (check relaxation "
                        f"folders with CONTCARs are present)."
                    )
    return low_energy_defects


def get_energy_lowering_distortions(
    defect_charges_dict: Optional[dict] = None,
    output_path: str = ".",
    code: str = "vasp",
    structure_filename: str = "CONTCAR",
    min_e_diff: float = 0.05,
    stol: float = 0.5,
    min_dist: float = 0.2,
    verbose: bool = True,
    write_input_files: bool = False,
    metastable: bool = False,
) -> dict:
    """
    Convenience function to identify defect species undergoing
    energy-lowering distortions. Useful for then testing these distorted
    structures for the other charge states of that defect. Considers all
    identified energy-lowering distortions for each defect in each charge
    state, and screens out duplicate distorted structures found for
    multiple charge states.

    Args:
        defect_charges_dict (:obj:`dict`, optional):
            Dictionary matching defect name(s) to list(s) of their
            charge states. (e.g {"Int_Sb_1":[0,+1,+2]} etc). If not
            specified, all defects present in `output_path` will be
            parsed.
            (Default: None)
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations
            (need code output/structure files for structure matching) and
            distortion_metadata.json.
            (Default is current directory = "./")
        code (:obj:`str`, optional):
            Code used for the geometry relaxations. Supported code names are:
            "vasp", "espresso", "cp2k" and "fhi-aims" (case insensitive).
            (Default: "vasp")
        structure_filename (:obj:`str`, optional):
            Name of the file containing the structure.
            (Default: CONTCAR)
        min_e_diff (:obj: `float`):
            Minimum energy difference (in eV) between the ground-state
            defect structure, relative to the `Unperturbed` structure,
            to consider it as having found a new energy-lowering
            distortion. Default is 0.05 eV.
        stol (:obj:`float`):
            Site-matching tolerance for structure matching. Site
            tolerance defined as the fraction of the average free length
            per atom := ( V / Nsites ) ** (1/3).
            (Default: 0.5)
        min_dist (:obj:`float`):
            Minimum atomic displacement threshold between structures, in
            order to consider them not matching (in Å, default = 0.2 Å).
        verbose (:obj:`bool`):
            Whether to print verbose information about energy lowering
            distortions, if found.
            (Default: True)
        write_input_files (:obj:`bool`):
            Whether to write input files for the identified distortions
            (Default: False)
        metastable (:obj:`bool`):
            Whether to also store non-spontaneous _metastable_
            energy-lowering distortions, as these can become ground-state
            distortions for other charge states.
            (Default: False)

    Returns:
        :obj:`dict`:
            Dictionary of defects for which bond distortion found an
            energy-lowering distortion (which is missed with normal
            unperturbed relaxation), of the form {defect: [list of
            distortion dictionaries (with corresponding charge states,
            energy lowering, distortion factors, structures and charge
            states for which these structures weren't found)]}.
    """
    if not os.path.isdir(output_path):  # check if output_path exists
        raise FileNotFoundError(f"Path {output_path} does not exist!")

    low_energy_defects = {}  # Dict of defects undergoing energy-lowering distortions,
    # relative to unperturbed structure
    # Maps each defect_name to a tuple of favourable distortions:
    # low_energy_defects[defect] = (
    #     {
    #         "charges": [charge],
    #         "structures": [gs_struct],
    #         "energy_diffs": [energy_diff],
    #         "bond_distortions": [gs_distortion],
    #         "excluded_charges": set(),
    #     },  # Dict with info for one of the favourable distortions
    #    ...
    # )

    if not defect_charges_dict:
        # defect_charges_dict maps defect_name to list of charge states
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
            energies_file = f"{output_path}/{defect_species}/{defect_species}.yaml"
            energies_dict, energy_diff, gs_distortion = analysis._sort_data(
                energies_file, verbose=verbose
            )

            # Defect without data
            if energies_dict is None:
                print(
                    f"No data parsed for {defect_species}. This species will be "
                    f"skipped and will not be included in the low_energy_defects "
                    f"charge state lists (and so energy lowering distortions "
                    f"found for other charge states will not be applied for "
                    f"this species)."
                )
                defect_pruning_dict[defect].remove(charge)

            # Parse only ground state distortion for each charge state
            elif (
                energy_diff and float(energy_diff) < -min_e_diff and not metastable
            ):  # if a significant energy drop occurred, then store this distorted defect
                bond_distortion = analysis._get_distortion_filename(gs_distortion)
                # format distortion label to the one used in file name
                # (e.g. from 0.1 to Bond_Distortion_10.0%)
                with warnings.catch_warnings(record=True) as w:
                    gs_struct = io.parse_structure(
                        code=code,
                        structure_path=f"{output_path}/{defect_species}/{bond_distortion}",
                        structure_filename=structure_filename,
                    )  # get the final structure of the
                    # energy lowering distortion
                    if any(warning.category == UserWarning for warning in w):
                        # problem parsing structure, user will have received appropriate
                        # warning from io.read_vasp_structure()
                        print(
                            f"Problem parsing final, low-energy structure for "
                            f"{gs_distortion} bond distortion of {defect_species} "
                            f"at {output_path}/{defect_species}/{bond_distortion}/"
                            f"{structure_filename}. This species will be skipped and "
                            f"will not be included in low_energy_defects (check"
                            f"relaxation calculation and folder)."
                        )
                        defect_pruning_dict[defect].remove(charge)
                        continue

                if (
                    defect in low_energy_defects
                ):  # Check if the lower-energy distorted structure was already
                    # found with bond distortions for a different charge state
                    # of this defect
                    low_energy_defects = _compare_distortion(
                        defect=defect,
                        defect_species=defect_species,
                        charge=charge,
                        energy_diff=energy_diff,
                        gs_distortion=gs_distortion,
                        gs_struct=gs_struct,
                        low_energy_defects=low_energy_defects,
                        stol=stol,
                        min_dist=min_dist,
                    )

                elif defect not in low_energy_defects:
                    # if defect not in dict, add it
                    print(
                        f"Energy lowering distortion found for {defect} with "
                        f"charge {charge}. Adding to low_energy_defects dictionary."
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

            # Parse all energy-lowering distortions (ground-state and metastable)
            elif metastable and energy_diff and float(energy_diff) < -min_e_diff:
                fav_energies_dict = {  # favourable distortions
                    "distortions": {
                        key: round(value - energies_dict["Unperturbed"], 2)
                        for key, value in energies_dict["distortions"].items()
                        if value - energies_dict["Unperturbed"] < -min_e_diff
                    }
                }
                # Get unique distortions
                # Discard the ones with similar energies to other distortions
                unique_distortions = {  # energy_diff: unique_distortion
                    value: key
                    for key, value in fav_energies_dict["distortions"].items()
                    if value in set(fav_energies_dict["distortions"].values())
                }
                for energy_diff, distortion in unique_distortions.items():
                    # for each unique distortion, get the corresponding
                    # structure
                    bond_distortion = analysis._get_distortion_filename(distortion)
                    with warnings.catch_warnings(record=True) as w:
                        struct = io.parse_structure(
                            code=code,
                            structure_path=f"{output_path}/{defect_species}/{bond_distortion}",
                            structure_filename=structure_filename,
                        )
                        if any(warning.category == UserWarning for warning in w):
                            # problem parsing structure, user will have received appropriate
                            # warning from io.read_vasp_structure()
                            print(
                                f"Problem parsing final, low-energy structure for "
                                f"{gs_distortion} bond distortion of {defect_species} "
                                f"at {output_path}/{defect_species}/{bond_distortion}/"
                                f"{structure_filename}. This species will be skipped and "
                                f"will not be included in low_energy_defects (check"
                                f"relaxation calculation and folder)."
                            )
                            defect_pruning_dict[defect].remove(charge)
                            continue

                        if (
                            defect in low_energy_defects
                        ):  # Check if the lower-energy distorted structure was already
                            # found with bond distortions for a different charge state
                            # of this defect
                            low_energy_defects = _compare_distortion(
                                defect=defect,
                                defect_species=defect_species,
                                charge=charge,
                                energy_diff=energy_diff,
                                gs_distortion=distortion,
                                gs_struct=struct,
                                low_energy_defects=low_energy_defects,
                                stol=stol,
                                min_dist=min_dist,
                            )

                        elif defect not in low_energy_defects:
                            # if defect not in dict, add it
                            print(
                                f"Energy lowering distortion found for {defect} with "
                                f"charge {charge}. Adding to low_energy_defects dictionary."
                            )
                            low_energy_defects[defect] = [
                                {
                                    "charges": [charge],
                                    "structures": [struct],
                                    "energy_diffs": [energy_diff],
                                    "bond_distortions": [distortion],
                                    "excluded_charges": set(),
                                }
                            ]

            # warning if all rattled distortions are higher energy than unperturbed
            elif gs_distortion == "Unperturbed" and all(
                [
                    value - energies_dict["Unperturbed"] > 0.1
                    for value in energies_dict["distortions"].values()
                ]
            ):
                warnings.warn(
                    f"All distortions for {defect} with charge {charge} are >0.1 eV higher energy "
                    f"than unperturbed, indicating problems with the relaxations. You should "
                    f"firstly check the calculations finished ok for this defect species, "
                    f"and if so, you likely need to adjust the `std_dev` rattling parameter ("
                    f"can occur for hard/ionic/close-packed materials); see "
                    f"https://shakenbreak.readthedocs.io/en/latest/Tips.html#hard-ionic-materials."
                )

            else:
                print(
                    f"No energy lowering distortion with energy difference greater "
                    f"than  min_e_diff = {min_e_diff:.2f} eV found for {defect} "
                    f"with charge {charge}."
                )

    # Screen through defects to check if any lower-energy distorted structures
    # were already found with/without bond distortions for other charge states
    # (i.e. found but higher energy, found but also with unperturbed, found
    # but with energy lowering less than min_e_diff etc)
    print("\nComparing and pruning defect structures across charge states...")
    low_energy_defects = _prune_dict_across_charges(
        low_energy_defects=low_energy_defects,
        defect_pruning_dict=defect_pruning_dict,
        code=code,
        structure_filename=structure_filename,
        output_path=output_path,
        stol=stol,
        min_dist=min_dist,
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
    code: str = "vasp",
    structure_filename: str = "CONTCAR",
    stol: float = 0.5,
    min_dist: float = 0.2,
) -> tuple:
    """
    Compares the ground-state structure found for a certain defect charge
    state with all relaxed bond-distorted structures for `defect_species`,
    to avoid redundant work (testing this distorted structure for other
    charge states when it has already been found for them).

    Args:
        distorted_struct (:obj:`~pymatgen.core.structure.Structure`):
            Structure of ground-state distorted defect
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to
            calculate structure comparisons – needs code output/structure
            files to parse the structures).
            (Default is current directory = "./")
        code (:obj:`str`, optional):
            Code used for the geometry relaxations. Options include:
            "vasp", "cp2k", "espresso", "castep", "fhi-aims" (case insensitive).
            (Default: "vasp")
        structure_filename (:obj:`str`, optional):
            Name of the file containing the structure.
            (Default: CONTCAR)
        stol (:obj:`float`):
            Site-matching tolerance for structure matching. Site
            tolerance defined as thefraction of the average free length
            per atom := ( V / Nsites ) ** (1/3).
            (Default: 0.5)
        min_dist (:obj:`float`):
            Minimum atomic displacement threshold between structures, in
            orderto consider them not matching (in Å, default = 0.2 Å).

    Returns:
        :obj:`tuple`:
            (True/False/None, matching structure, energy difference of the
            matching structure compared to its unperturbed reference, bond
            distortion of the matching structure). True if a match is found
            between the input structure and the relaxed bond-distorted
            structures for `defect_species`, False if no match, None if no
            converged structures found for defect_species.
    """
    try:
        defect_structures_dict = analysis.get_structures(
            defect_species=defect_species,
            output_path=output_path,
            code=code,
            structure_filename=structure_filename,
        )
    except FileNotFoundError:  # catch exception raised by `analysis.get_structures()`
        return None, None, None, None
    defect_energies_dict = analysis.get_energies(
        defect_species=defect_species, output_path=output_path, verbose=False
    )

    # Compare distorted_struct to all structures of defect_species
    struct_comparison_df = analysis.compare_structures(
        defect_structures_dict=defect_structures_dict,
        defect_energies_dict=defect_energies_dict,
        ref_structure=distorted_struct,
        stol=stol,
        min_dist=min_dist,
        display_df=False,
    )
    if struct_comparison_df is None:  # no converged structures found for
        # defect_species
        return None, None, None, None

    matching_sub_df = struct_comparison_df[
        struct_comparison_df["Σ{Displacements} (Å)"] == 0
    ]  # Get matches (sum of atomic disp between structures would be 0)

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
            matching_sub_df["Bond Distortion"].apply(
                lambda x: isinstance(x, float)
            )  # if present, otherwise empty
        ].sort_values(
            by="Bond Distortion", key=abs
        )  # sort values by distortion magnitude

        string_vals_sorted_distorted_df = matching_sub_df[
            matching_sub_df["Bond Distortion"].apply(lambda x: isinstance(x, str))
        ]
        imported_sorted_distorted_df = string_vals_sorted_distorted_df[
            string_vals_sorted_distorted_df["Bond Distortion"].apply(
                lambda x: "_from_" in x
            )
        ]

        if not imported_sorted_distorted_df.empty:
            # convert "X%_from_Y" strings to floats and then sort
            # needs to be done this way because 'key' in pd.sort_values()
            # needs to be vectorised...
            s = imported_sorted_distorted_df["Bond Distortion"].str.slice(0, 3)
            s = s.astype(float)
            imported_sorted_distorted_df = imported_sorted_distorted_df.loc[
                s.sort_values(key=lambda x: abs(x)).index
            ]

        # first unperturbed, then rattled, then distortions sorted by
        # initial distortion magnitude from low to high (if present)
        sorted_matching_df = pd.concat(
            [
                unperturbed_df,
                rattled_df,
                sorted_distorted_df,
                imported_sorted_distorted_df,
            ]
        )

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

    else:  # no matches
        return (
            False,
            None,
            None,
            None,
        )  # T/F, matching structure, energy_diff, distortion factor


def write_distorted_inputs(
    low_energy_defects: dict,
    output_path: str = ".",
    code: str = "vasp",
    input_filename: str = None,
) -> None:
    """
    Create folders with relaxation input files for testing the low-energy
    distorted defect structures found for other charge states of that
    defect, as identified with `get_energy_lowering_distortions()`.

    Args:
        low_energy_defects (:obj:`dict`):
            Dictionary of defects for which bond distortion found an
            energy-lowering distortion which is missed with normal
            unperturbed relaxation), generated by
            `get_energy_lowering_distortions()`. Has the form
            {defect: [list of distortion dictionaries (with
            corresponding charge states, energy lowering, distortion
            factors, structures and charge states for which these
            structures weren't found)]}.
        output_path (:obj:`str`):
            Path to directory with your distorted defect calculations
            (to write input files for distorted defect structures to
            test). (Default is current directory = "./")
        code (:obj:`str`):
            Code used for the geometry relaxations. The supported codes
            include "vasp", "cp2k", "espresso", "castep" and "fhi-aims"
            (case insensitive).
            (Default: "vasp")
        input_filename (:obj:`str`):
            Name of the code input file if different from `ShakeNBreak`
            default. Only applies to CP2K, Quantum Espresso, CASTEP and
            FHI-aims. If not specified, `ShakeNBreak` default name is
            assumed, that is: for Quantum Espresso: "espresso.pwi",
            CP2K: "cp2k_input.inp", CASTEP: "castep.param",
            FHI-aims: "control.in"
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
                )  # Format distoriton directory name

                if os.path.exists(distorted_dir):
                    print(
                        f"As {distorted_dir} already exists, it's assumed this "
                        f"structure has already been tested. Skipping..."
                    )
                    continue

                print(f"Writing low-energy distorted structure to {distorted_dir}")

                if not os.path.exists(f"{output_path}/{defect_species}"):
                    print(
                        f"Directory {output_path}/{defect_species} not found, "
                        f"creating..."
                    )
                    os.mkdir(f"{output_path}/{defect_species}")
                os.mkdir(distorted_dir)

                # copy input files from Unperturbed directory
                if code.lower() == "vasp":
                    _copy_vasp_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                    )
                elif code.lower() == "espresso":
                    _copy_espresso_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )
                elif code.lower() == "cp2k":
                    _copy_cp2k_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )
                elif code.lower() == "castep":
                    _copy_castep_files(
                        distorted_structure,
                        distorted_dir,
                        output_path,
                        defect_species,
                        input_filename,
                    )
                elif code.lower() == "fhi-aims":
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
    Copy VASP input files from an existing distortion directory
    to a new directory.
    """
    distorted_structure.to(fmt="poscar", filename=f"{distorted_dir}/POSCAR")

    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/INCAR"):
        for i in ["INCAR", "KPOINTS", "POTCAR"]:
            shutil.copyfile(
                f"{output_path}/{defect_species}/Unperturbed/{i}",
                f"{distorted_dir}/{i}",
            )  # copy input files from Unperturbed directory
    else:
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(f"{output_path}/{defect_species}/{subfolder}/INCAR"):
                subfolders_with_input_files.append(subfolder)
        if len(subfolders_with_input_files) > 0:
            for i in ["INCAR", "KPOINTS", "POTCAR"]:
                shutil.copyfile(
                    f"{output_path}/{defect_species}/"
                    f"{subfolders_with_input_files[0]}/"
                    f"{i}",
                    f"{distorted_dir}/{i}",
                )
        else:
            print(
                f"No subfolders with VASP input files found in "
                f"{output_path}/{defect_species}, so just writing distorted "
                f"POSCAR file to {distorted_dir} directory."
            )


def _copy_espresso_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "espresso.pwi",
) -> None:
    """
    Copy Quantum Espresso input files from an existing distortion
    directory to a new directory.
    """
    if not input_filename:
        input_filename = "espresso.pwi"
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        # Parse input parameters from file and update structural info with
        # new distorted structure
        # ase/pymatgen dont support this
        with open(f"{output_path}/{defect_species}/Unperturbed/{input_filename}") as f:
            params = f.read()  # Read input parameters
        # Write distorted structure in QE format, to then update input file
        atoms = aaa.get_atoms(distorted_structure)
        ase_write(
            filename=f"{distorted_dir}/{input_filename}",
            images=atoms,
            format="espresso-in",
        )
        with open(f"{distorted_dir}/{input_filename}") as f:
            new_struct = f.read()
        params = params.replace(
            params[params.find("ATOMIC_POSITIONS"):],
            new_struct[new_struct.find("ATOMIC_POSITIONS"):],
            1,
        )  # Replace ionic positions
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
            with open(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}/"
                f"{input_filename}"
            ) as f:
                params = f.read()  # Read input parameters
            # Write distorted structure in QE format, to then update input file
            atoms = aaa.get_atoms(distorted_structure)
            ase_write(
                filename=f"{distorted_dir}/{input_filename}",
                images=atoms,
                format="espresso-in",
            )
            with open(f"{distorted_dir}/{input_filename}") as f:
                new_struct = f.read()
            params = params.replace(
                params[params.find("ATOMIC_POSITIONS"):],
                new_struct[new_struct.find("ATOMIC_POSITIONS"):],
                1,
            )  # Replace lines with the ionic positions
            with open(f"{distorted_dir}/{input_filename}", "w") as f:
                f.write(params)
        else:  # only write input structure
            print(
                f"No subfolders with Quantum Espresso input file (`{input_filename}`) "
                f"found in {output_path}/{defect_species}, so just writing "
                f"distorted structure file to {distorted_dir} directory."
            )
            atoms = aaa.get_atoms(distorted_structure)
            ase_write(
                filename=f"{distorted_dir}/{input_filename}",
                images=atoms,
                format="espresso-in",
            )


def _copy_cp2k_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "cp2k_input.inp",
) -> None:
    """
    Copy CP2K input files from an existing distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "cp2k_input.inp"
    distorted_structure.to("cif", f"{distorted_dir}/structure.cif")
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        shutil.copyfile(
            f"{output_path}/{defect_species}/Unperturbed/{input_filename}",
            f"{distorted_dir}/{input_filename}",
        )
    else:  # Check of input file present in the other distortion subfolders
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            shutil.copyfile(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}"
                f"/{input_filename}",
                f"{distorted_dir}/{input_filename}",
            )

        else:  # only write input structure
            print(
                f"No subfolders with CP2K input file (`cp2k_input.inp`) "
                f"found in {output_path}/{defect_species}, so just writing "
                f"distorted structure file to {distorted_dir} directory "
                f"(in CIF format)."
            )


def _copy_castep_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "castep.param",
) -> None:
    """
    Copy CASTEP input files from an existing distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "castep.param"
    atoms = aaa.get_atoms(distorted_structure)
    ase_write(
        filename=f"{distorted_dir}/castep.cell", images=atoms, format="castep-cell"
    )  # Write structure
    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        shutil.copyfile(
            f"{output_path}/{defect_species}/Unperturbed/{input_filename}",
            f"{distorted_dir}/{input_filename}",
        )
    else:  # Check of input file present in the other distortion subfolders
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            shutil.copyfile(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}"
                f"/{input_filename}",
                f"{distorted_dir}/{input_filename}",
            )

        else:  # only write input structure
            print(
                f"No subfolders with CASTEP input file (`{input_filename}`) "
                f"found in {output_path}/{defect_species}, so just writing "
                f"distorted structure file to {distorted_dir} directory (in "
                f"CASTEP `.cell` format)."
            )


def _copy_fhi_aims_files(
    distorted_structure: Structure,
    distorted_dir: str,
    output_path: str,
    defect_species: str,
    input_filename: str = "control.in",
) -> None:
    """
    Copy FHI-aims input files from an existing distortion directory
    to a new directory.
    """
    if not input_filename:
        input_filename = "control.in"
    atoms = aaa.get_atoms(distorted_structure)
    ase_write(
        filename=f"{distorted_dir}/geometry.in",
        images=atoms,
        format="aims",
    )  # write input structure file

    if os.path.exists(f"{output_path}/{defect_species}/Unperturbed/{input_filename}"):
        shutil.copyfile(
            f"{output_path}/{defect_species}/Unperturbed/{input_filename}",
            f"{distorted_dir}/{input_filename}",
        )
    else:  # Check of input file present in the other distortion subfolders
        subfolders_with_input_files = []
        for subfolder in os.listdir(f"{output_path}/{defect_species}"):
            if os.path.exists(
                f"{output_path}/{defect_species}/{subfolder}/{input_filename}"
            ):
                subfolders_with_input_files.append(subfolder)
                break
        if len(subfolders_with_input_files) > 0:
            shutil.copyfile(
                f"{output_path}/{defect_species}/{subfolders_with_input_files[0]}"
                f"/{input_filename}",
                f"{distorted_dir}/{input_filename}",
            )

        else:  # only write input structure
            print(
                f"No subfolders with FHI-aims input file (`{input_filename}`) "
                f"found in {output_path}/{defect_species}, so just writing "
                f"distorted structure file to {distorted_dir} directory (in "
                f"FHI-aims `geometry.in` format)."
            )


def write_groundstate_structure(
    all: bool = True,
    output_path: str = ".",
    groundstate_folder: str = None,
    groundstate_filename: str = "groundstate_POSCAR",
    structure_filename: str = "CONTCAR",
    verbose: bool = False,
) -> None:
    """
    Writes the groundstate structure of each defect (if `all=True`, default)
    to the corresponding defect folder, with an optional name
    (default "groundstate_POSCAR"), to then run continuation calculations.

    Args:
        all (:obj: `bool`):
            Write groundstate structures for all defect folders in the
            (top-level) directory, specified by `output_path`. If False,
            `output_path` should be a single defect folder, for which the
            groundstate structure will be written.
        output_path (:obj:`str`):
            Path to top-level directory with your distorted defect
            calculation folders (if `all=True`, else path to single defect
            folder)(need CONTCAR files for structure matching) and
            distortion_metadata.json.
            (Default: current directory = "./")
        groundstate_folder (:obj:`str`):
            Name of the directory to write the groundstate structure to.
            (Default: None (ground state structure is written to the root
            defect directory))
        groundstate_filename (:obj:`str`):
            Name of the file to write the groundstate structure to.
            (Default: "groundstate_POSCAR")
        structure_filename (:obj:`str`):
            Name of the file to read the structure from.
            (Default: "CONTCAR")
        verbose (:obj:`bool`):
            Whether to print additional information about the generated folders.

    Returns:
        None
    """

    def _write_single_groundstate(
        output_path,
        species,
        groundstate_folder,
        groundstate_filename,
        structure_filename,
        verbose,
    ):
        energies_file = f"{output_path}/{species}/{species}.yaml"
        # Get ground state distortion
        _, _, gs_distortion = analysis._sort_data(
            energies_file=energies_file, verbose=False
        )
        bond_distortion = analysis._get_distortion_filename(gs_distortion)

        # Origin path
        origin_path = f"{output_path}/{species}/{bond_distortion}/{structure_filename}"
        if not os.path.exists(origin_path):
            raise FileNotFoundError(
                f"The structure file {structure_filename} is not present"
                f" in the directory {output_path}/{species}/{bond_distortion}"
            )

        # Destination path
        if groundstate_folder:
            if not os.path.exists(f"{output_path}/{species}/{groundstate_folder}"):
                os.mkdir(f"{output_path}/{species}/{groundstate_folder}")
            destination_path = os.path.join(
                f"{output_path}/{species}/",
                f"{groundstate_folder}/{groundstate_filename}",
            )
            if verbose:
                print(
                    f"{species}: Gound state structure (found with "
                    f"{gs_distortion} distortion) saved to {destination_path}"
                )
        else:
            destination_path = f"{output_path}/{species}/{groundstate_filename}"

        shutil.copyfile(
            origin_path,
            destination_path,
        )

    if all:
        defect_charges_dict = read_defects_directories(output_path=output_path)
        for defect, charges in defect_charges_dict.items():
            for charge in charges:
                _write_single_groundstate(
                    output_path=output_path,
                    species=f"{defect}_{charge}",
                    groundstate_folder=groundstate_folder,
                    groundstate_filename=groundstate_filename,
                    structure_filename=structure_filename,
                    verbose=verbose,
                )
    else:
        species = output_path.split("/")[-1]
        output_path = output_path.rsplit("/", 1)[0]

        _write_single_groundstate(
            output_path=output_path,
            species=species,
            groundstate_folder=groundstate_folder,
            groundstate_filename=groundstate_filename,
            structure_filename=structure_filename,
            verbose=verbose,
        )
