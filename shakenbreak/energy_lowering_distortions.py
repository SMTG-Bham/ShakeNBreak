""" 
Functions to apply energy lowering distortions found for a certain defect species (charge state)
to other charge states of that defect.
In progress
"""
# TODO: Incorporate this in a different module / submodule for structure distortion, and merge
#  with champion_defects_rerun
import copy
import warnings
import pandas as pd

from pymatgen.core.structure import Structure

from shakenbreak.analysis import (
    _sort_data,
    grab_contcar,
    get_structures,
    get_energies,
    calculate_struct_comparison,
    compare_structures,
)


# TODO: Update get_deep_distortions() to optionally also store non-spontaneous _metastable_
#  energy-lowering distortions, as these can become ground-state distortions for other charge
#  states
def get_deep_distortions(
    defect_charges_dict: dict,
    output_path: str = "./",
    min_e_diff: float = 0.05,
    stol: float = 0.5,
    min_dist: float = 0.2,
    verbose: bool = True
) -> dict:
    """Convenience function to identify defect species undergoing energy-lowering distortions.
     Useful for then testing this distorted structure for the other charge states of that defect.
     Considers all identified energy-lowering distortions for each defect in each charge state,
     and screens out duplicate distorted structures for different charge states.

     Args:
         defect_charges_dict (:obj:`dict`):
             Dictionary matching defect name(s) to list(s) of their charge states. (e.g {
             "Int_Sb_1":[0,+1,+2]} etc)
         output_path (:obj:`str`):
             Path to directory with your distorted defect calculations (need CONTCAR files for
             structure matching) and distortion_metadata.json. (Default is current directory = "./")
         min_e_diff (:obj: `float`):
             Minimum energy difference (in eV) between the `champion` test relaxation and the
             previous lowest energy relaxation, to consider it as having found a new energy-lowering
             distortion. Default is 0.05 eV.
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

    Returns:
         low_energy_defects (:obj:`dict`):
             Dictionary of defects for which bond distortion found a 'deep' distortion (missed with
             normal unperturbed relaxation), of the form {defect: [list of distortion
             dictionaries (with corresponding charge states, energy lowering, distortion factors,
             structures and charge states for which these structures weren't found)]}.
    """
    low_energy_defects = (
        {}
    )  # dict of defects undergoing deep energy-lowering distortions

    defect_pruning_dict = copy.deepcopy(defect_charges_dict)  # defects and charge states to do
    # later comparison and pruning against other charge states
    for defect in defect_charges_dict:
        print(f"\n{defect}")
        defect_pruning_dict[defect] = []
        for charge in defect_charges_dict[defect]:
            defect_pruning_dict[defect].append(charge)
            defect_species = f"{defect}_{charge}"
            energies_file = f"{output_path}/{defect_species}/{defect_species}.txt"
            energies_dict, energy_diff, gs_distortion = _sort_data(energies_file, verbose=verbose)

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
                if gs_distortion != "rattled":
                    bond_distortion = (
                        f"{round(gs_distortion * 100, 1)+0}"  # change distortion
                    )
                    # format to the one used in file name (e.g. from 0.1 to 10.0)
                else:
                    bond_distortion = (
                        "only_rattled"  # file naming format used for rattle
                    )
                file_path = f"{output_path}/{defect_species}/Bond_Distortion" \
                            f"_{bond_distortion}%/CONTCAR"
                with warnings.catch_warnings(record=True) as w:
                    gs_struct = grab_contcar(
                        file_path
                    )  # get the final structure of the
                    # energy lowering distortion
                    if any([warning.category == UserWarning for warning in w]):
                        # TODO: Pylint use a generator instead here
                        # problem parsing structure, user will have received appropriate
                        # warning from grab_contcar()
                        print(
                            f"Problem parsing final, low-energy structure for {bond_distortion}% "
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
                    for i in range(len(low_energy_defects[defect])):  # use _initial_ list count
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
                        if struct_comparison_dict["Ground State"] == 0:  # match
                            print(
                                f"Low-energy distorted structure for {defect_species} already found"
                                f" with charge states {low_energy_defects[defect]['charges']}, "
                                f"storing together."
                            )
                            low_energy_defects[defect][i]["charges"].append(charge)
                            low_energy_defects[defect][i]["structures"].append(gs_struct)
                            low_energy_defects[defect][i]["energy_diffs"].append(energy_diff)
                            low_energy_defects[defect][i]["bond_distortions"].append(gs_distortion)
                        else:
                            # if the structure was not previously found, then add it to the list of
                            # distortions for this defect
                            print(
                                f"New (according to structure matching) low-energy distorted "
                                f"structure for {defect_species} found with charge state {charge}, "
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
                            continue

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

    # Screen through defects to check that if any lower-energy distorted structures were already
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

    return low_energy_defects


def compare_struct_to_distortions(
    distorted_struct: Structure,
    defect_species: str,
    output_path: str = "./",
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
    defect_structures_dict = get_structures(defect_species, output_path)
    defect_energies_dict = get_energies(defect_species, output_path, verbose=False)

    struct_comparison_df = compare_structures(
        defect_structures_dict,
        defect_energies_dict,
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
    unperturbed_df = matching_sub_df[
        matching_sub_df["Bond Distortion"]
        == "Unperturbed"  # if present, otherwise empty
    ]
    rattled_df = matching_sub_df[
        matching_sub_df["Bond Distortion"] == "rattled"  # if present, otherwise empty
    ]
    sorted_distorted_df = matching_sub_df[
        ~matching_sub_df["Bond Distortion"].isin(
            ["Unperturbed", "rattled"]
        )  # tilde means 'not'
    ].sort_values(by="Bond Distortion", key=abs)

    # first unperturbed, then rattled, then distortions sorted by initial distortion magnitude
    # from low to high (if present)
    sorted_matching_df = pd.concat([unperturbed_df, rattled_df, sorted_distorted_df])

    if not sorted_matching_df.empty:  # if there are any matches
        struc_key = sorted_matching_df["Bond Distortion"].iloc[
            0
        ]  # first matching structure
        if struc_key == "Unperturbed":
            return (
                True,
                defect_structures_dict[struc_key],
                defect_energies_dict[struc_key],
                struc_key,
            )
        else:
            return (
                True,
                defect_structures_dict[struc_key],
                defect_energies_dict["distortions"][struc_key],
                struc_key,
            )

    else:
        return False, None, None, None
    # return True/False, matching structure, energy_diff, distortion factor


def import_deep_distortion_by_type(
    defect_dict: dict,
    low_energy_defects: dict,  # TODO: Refactor to just use this, as low_energy_defects now also
    # contains the tested charge states for which this distortion wasn't found (which also
    # means if a certain charge state wasn't parsed correctly then it won't be included (as
    # if it were, this could add unnecessary duplicate calculations for cases where it was
    # found for that charge state, but just wasn't parsed right)(i.e. only include charge
    # states for which we've actually tested with structure matching)
) -> dict:
    """
    Import the ground-state energy-lowering distortion found for certain defect charge states, in
    order to test these structures for other charge states of those defects.

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the `doped.pycdt.core.defectsmaker.ChargedDefectsStructures()`
            format.
        low_energy_defects (:obj:`dict`):
             Dictionary of defects for which bond distortion found a 'deep' energy-lowering
             distortion (missed with normal unperturbed relaxation).

    Returns:
        Dictionary of deep-distorted defects with initial structures matching that of the
        distorted ground-state for a different charge state of the same defect.
    """
    deep_distortion_dict = {}
    for k, v in defect_dict.items():  # TODO: I think should be refactored to use
        # defect_charges_dict rather than defect_dict here, as simpler and we're only using
        # defect_dict for names and charges
        if (
            v["name"] in low_energy_defects.keys()
        ):  # if defect underwent a deep distortion
            defect_subdict = copy.deepcopy(v)  # copy the defect dict and use this
            defect_name = defect_subdict["name"]
            print(defect_name)
            defect_subdict["supercell"]["structure"] = low_energy_defects[defect_name][
                "structure"
            ]  # structure of energy-lowering distortion
            print(
                f"Using the distortion found for charge state(s) "
                f"{low_energy_defects[defect_name]['charges']} with bond distortion "
                f"{low_energy_defects[defect_name]['bond_distortion']}"
            )
            # remove the charge state of the energy-lowering distortion
            if len(low_energy_defects[defect_subdict["name"]]["charges"]) > 1:
                print(
                    "Initial charge states of defect:",
                    defect_subdict["charges"],
                    "Removing the ones where the distortion was found...",
                )
                for charge in low_energy_defects[defect_name]["charges"]:
                    defect_subdict["charges"].remove(charge)
                print("Trying distortion for charge states:", defect_subdict["charges"])
            else:
                defect_subdict["charges"].remove(
                    low_energy_defects[defect_name]["charges"][0]
                )
            if defect_subdict[
                "charges"
            ]:  # if list of charges to try deep distortion not empty, then add defect to the dict
                deep_distortion_dict[k] = defect_subdict
    return deep_distortion_dict
