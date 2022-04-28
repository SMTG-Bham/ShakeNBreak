"""
Functions to apply and test the energy-lowering distortion found for a certain charge state of a
defect to the other charge states.
"""

import os
from shakenbreak.analysis import sort_data, grab_contcar


def read_defects_directories(defect_path=None) -> dict:
    """
    Reads all defect folders in the current directory and stores defect names and charge states in
    a dictionary.

    Args:
        defect_path (:obj: `str`): Path to the defect folders.

    Returns:
        Dictionary with defect names and charge states.
    """
    if defect_path:
        path = defect_path
    else:
        path = "./defects"
    list_subdirectories = [  # Get only subdirectories in the current directory
        i
        for i in next(os.walk(path))[1]
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


def compare_champion_to_distortions(
    defect_species, base_path, min_e_diff=0.05
) -> tuple:
    """
    Check if an energy-lowering distortion was found when relaxing from the (relaxed) ground-state
    structure of another charge state of that defect, by comparing the energy difference between
    this calculation and the previous lowest energy relaxation.

    Args:
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0').
        base_path (:obj: `str`):
            Path to the defect folders (within which the `defect_name` folder is located).
        min_e_diff (:obj: `float`):
            Minimum energy difference (in eV) between the `champion` test relaxation and the
            previous lowest energy relaxation, to consider it as having found a new energy-lowering
            distortion.

    Returns:
        (True/False, energy_diff) where True if a lower-energy result was found (at least
        `energy_difference` eV lower than the previous lowest energy relaxation),
        where `energy_diff` is the energy difference between this lowest-energy relaxation and
        the Unperturbed result. If False, `energy_diff` is the energy difference between the
        `champion` test relaxation and the previous lowest energy relaxation.
    """
    distorted_energies_dict, distorted_energy_drop, distorted_gs_dist = sort_data(
        f"{base_path}/{defect_species}/{defect_species}.txt"
    )
    champ_energies_dict, champ_energy_drop, champ_gs_dist = sort_data(
        f"{base_path}/{defect_species}/champion_{defect_species}.txt"
    )

    # Check what distortion lead to the lowest E structure: Unperturbed, bond-distorted or just
    # rattled?
    if isinstance(distorted_gs_dist, float) or distorted_gs_dist == "rattled":
        min_energy_distorted = distorted_energies_dict["distortions"][
            distorted_gs_dist
        ]
    else:
        min_energy_distorted = distorted_energies_dict["Unperturbed"]

    if isinstance(champ_gs_dist, float) or champ_gs_dist == "rattled":
        min_energy_champ = champ_energies_dict["distortions"][champ_gs_dist]
    else:
        min_energy_champ = champ_energies_dict["Unperturbed"]

    energy_diff = (
        min_energy_champ - min_energy_distorted
    )
    #  if lower E structure found by importing the gs of another charge state
    if energy_diff < -min_e_diff:
        energy_diff_vs_unperturbed = (
                min_energy_champ - distorted_energies_dict["Unperturbed"]
        )  # for later comparison, set the energy difference relative to Unperturbed structure (not
        # to the minimum energy found with bond distortions)
        print(f"Lower energy structure found for the 'champion' relaxation with {defect_species}, "
              f"with an energy {energy_diff:.2f} eV lower than the previous lowest energy from "
              f"distortions, with an energy {energy_diff_vs_unperturbed:.2f} eV lower than "
              f"relaxation from the Unperturbed structure.")
        return True, energy_diff_vs_unperturbed
    return False, energy_diff


def get_champion_defects(defect_charges_dict, base_path, energy_difference=0.05) -> dict:
    """
    Get defect names and charge states for which bond distortion found a ground-state structure
    that is missed by Unperturbed relaxation.

    Args:
        defect_charges_dict (:obj:`dict`):
             Dictionary matching defect name(s) to list(s) of their charge states. (e.g {
             "Int_Sb_1":[0,+1,+2]} etc)
        base_path (:obj: `str`):
            Path to the defect folders (within which the `defect_species` folders (e.g. `vac_1_Cd_0`)
            are located).
        energy_difference (:obj: `float`):
            Minimum energy difference (in eV) between the lowest-energy relaxation and the
            Unperturbed relaxation, to consider it as having found a new energy-lowering
            distortion (that will be further analysed and tested for other charge states of that
            defect if they did not also find this structure).

    Returns:
        Dictionary of the form {defect_species: info_subdict} for the defects for which
        energy-lowering distortions were found (that were missed by Unperturbed relaxation),
        where `info_subdict` gives the relaxed Unperturbed and ground-state distorted structures,
        and the energy difference between them.
    """
    all_defects_structs = {}
    for defect in defect_charges_dict:
        for charge in defect_charges_dict[defect]:
            defect_name = f"{defect}_{charge}"

            new_struct_found = (
                False  # check where energy-lowering distortions were found by
            )
            # importing the gs of another charge state
            # check if the gs found for another charge state was tried for the defect (this means
            # initial bond distortion tests didn't find an energy-lowering distortion)

            if os.path.isfile(
                f"{base_path}/{defect_name}/champion_{defect_name}.txt"
            ):  # if champion folder exists
                distortion_type = "champion"
                energies_file = (
                    f"{base_path}/{defect_name}/{distortion_type}_{defect_name}.txt"
                )
                energies_dict, E_diff, gs_distortion = sort_data(
                    energies_file
                )  # between imported structure and imported structure + bond distortions
                new_struct_found, energy_diff = compare_champion_to_distortions(
                    defect_name, base_path, energy_difference
                )  # check if lower energy structure was found from importing the ground-state
                # found for another charge state
                if not new_struct_found:
                    # if we didn't find an energy-lowering distortion when relaxing from the (
                    # relaxed) ground-state structure of another charge state of that defect,
                    # check if initial bond distortions did
                    distortion_type = "BDM"
                    energies_file = (
                        f"{base_path}/{defect_name}/{defect_name}.txt"
                    )
                    energies_dict, energy_diff, gs_distortion = sort_data(energies_file)
                    if float(energy_diff) < -energy_difference:
                        new_struct_found = True

            elif os.path.isfile(f"{base_path}/{defect_name}/{defect_name}.txt"):
                distortion_type = "BDM"
                energies_file = (
                    f"{base_path}/{defect_name}/{defect_name}.txt"
                )
                energies_dict, energy_diff, gs_distortion = sort_data(energies_file)
                if float(energy_diff) < -energy_difference:
                    new_struct_found = True

            # if a significant energy drop occurred then further analyse this defect
            if new_struct_found:
                # transform to format used in file names
                if isinstance(gs_distortion, float):
                    gs_distortion = f"Bond_Distortion_{100*gs_distortion:.1f}%"
                if gs_distortion == "rattled":
                    gs_distortion = "only_rattled"
                # Grab CONTCAR and transform to Structure object
                try:
                    if distortion_type != 'BDM':
                        contcar_path = f"{base_path}/{defect_name}/{distortion_type}_" 
                    else:
                        contcar_path = f"{base_path}/{defect_name}/"
                    unperturbed_contcar = (
                        f"{contcar_path}Unperturbed/CONTCAR"
                    )
                    distorted_contcar = (
                        f"{contcar_path}{gs_distortion}/CONTCAR"
                    )
                    if (
                        distortion_type == "champion"
                    ):  # then get the Unperturbed from BDM (not the imported structure found for other
                        # charge state)
                        unperturbed_contcar = (
                            f"{base_path}/{defect_name}/"
                            f"Unperturbed/CONTCAR"
                        )

                    # transform CONTCAR to Structure object and store in dictionary
                    struct_unperturbed = grab_contcar(
                        unperturbed_contcar
                    )  # transform to pmg Structure format
                    struct_distorted = grab_contcar(distorted_contcar)
                    defect_dict = {
                        "Unperturbed": struct_unperturbed,
                        "Distorted": struct_distorted,
                        "E_drop": energy_diff,
                    }
                    all_defects_structs[defect_name] = defect_dict
                except:
                    print(
                        f"Problem parsing files: {unperturbed_contcar} and {distorted_contcar}"
                    )

    return all_defects_structs
