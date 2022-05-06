""" 
Functions to apply energy lowering distortions found for a certain defect species (charge state)
to other charge states of that defect.
In progress
"""
# TODO: Incorporate this in a different module / submodule for structure distortion
# TODO: Flesh out function arguments and docstrings
import copy

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

from shakenbreak.analysis import _sort_data, grab_contcar, get_structures, get_energies, \
    compare_structures


# TODO: Update get_deep_distortions() to optionally also store non-spontaneous metastable
#  energy-lowering distortions, as these can become ground-state distortions for other charge
#  states
def get_deep_distortions(
    defect_charges_dict: dict,
    base_path: str = "./",
    min_e_diff: float = 0.05,
    stol=0.2,
) -> dict:
    """Convenience function to identify defect species undergoing energy-lowering distortions.
     Useful for then testing this distorted structure for the other charge states of that defect.

     Args:
         defect_charges_dict (:obj:`dict`):
             Dictionary matching defect name(s) to list(s) of their charge states. (e.g {
             "Int_Sb_1":[0,+1,+2]} etc)
         base_path (:obj:`str`):
             Path to directory with your distorted defect calculations (need CONTCAR files for
             structure matching) and distortion_metadata.txt. (Default: "./")
         min_e_diff (:obj: `float`):
             Minimum energy difference (in eV) between the `champion` test relaxation and the
             previous lowest energy relaxation, to consider it as having found a new energy-lowering
             distortion. Default is 0.05 eV.
         stol (:obj:`float`):
             Site-matching tolerance for structure matching. Site tolerance. Defined as the
             fraction of the average free length per atom := ( V / Nsites ) ** (1/3).
             (Default: 0.2)

    Returns:
         low_energy_defects (:obj:`dict`):
             Dictionary of defects for which bond distortion found a 'deep' distortion (missed with
             normal unperturbed relaxation). The charge state with the distortion associated to
             the greatest energy drop is selected.
    """
    low_energy_defects = (
        {}
    )  # dict of defects undergoing deep energy-lowering distortions
    for defect in defect_charges_dict:
        print(f"\n{defect}")
        for charge in defect_charges_dict[defect]:
            defect_name = f"{defect}_{charge}"
            energies_file = (
                f"{base_path}/{defect_name}/{defect_name}.txt"
            )
            energies_dict, energy_diff, gs_distortion = sort_data(energies_file)

            if energy_diff and float(energy_diff) < -min_e_diff:  # if a significant energy drop
                # occurred, then store this distorted defect
                print(f"Deep distortion found for {defect_name}")
                if gs_distortion != "rattled":
                    bond_distortion = (
                        f"{round(gs_distortion * 100, 1)+0}"  # change distortion
                    )
                    # format to the one used in file name (e.g. from 0.1 to 10.0)
                else:
                    bond_distortion = (
                        "only_rattled"  # file naming format used for rattle
                    )
                try:
                    file_path = (
                        f"{base_path}/{defect_name}/Bond_Distortion_{bond_distortion}%/CONTCAR"
                    )
                    gs_struct = grab_contcar(
                        file_path
                    )  # get the final structure of the energy lowering distortion
                    if gs_struct == "Not converged":
                        print(
                            f"Problem parsing final, low-energy structure for {bond_distortion} of"
                            f" {defect_name} – Unconverged (check relaxation calculation)"
                        )
                except FileNotFoundError:
                    print(
                        f"NO CONTCAR found for low-energy distortion: {bond_distortion} of"
                        f" {defect_name} – check folder"
                    )
                    break
                if (
                    defect in low_energy_defects
                ):  # check if defect already in stored dict (i.e. a different charge state of
                    # this defect gave a distorted lower energy structure)
                    gs_struct_in_dict = low_energy_defects[defect]["structure"]

                    if (
                        energy_diff < low_energy_defects[defect]["energy_diff"]
                    ):  # if energy drop for the current charge state is greater (more negative)
                        # than the already-stored distorted low-energy structure) then replace
                        # the stored structure with the current structure
                        print(
                            f"Energy lowering distortion found for {defect} with charge "
                            f"{charge}, with a greater energy drop ({energy_diff:.3f} eV) than the "
                            f"previously identified distorting charge state(s) ("
                            f"{low_energy_defects[defect]['charges']}, with largest energy drop "
                            f"of {low_energy_defects[defect]['energy_diff']:.3f} eV). Updating "
                            f"low_energy_defects dictionary with this."
                        )
                        low_energy_defects[defect].update(
                            {
                                "structure": gs_struct,
                                "bond_distortion": gs_distortion,
                                "energy_diff": energy_diff,
                                "charges": [charge],
                            }
                        )

                elif defect not in low_energy_defects:  # if defect not in dict, add it
                    print(
                        f"Energy lowering distortion found for {defect} with charge {charge}. "
                        f"Adding to low_energy_defects dictionary."
                    )
                    low_energy_defects[defect] = {
                        "charges": [charge],
                        "structure": gs_struct,
                        "energy_diff": energy_diff,
                        "bond_distortion": gs_distortion,
                    }

            else:
                print(
                    f"No energy lowering distortion with energy difference greater than "
                    f"min_e_diff = {min_e_diff:.2f} eV found for {defect} with charge {charge}."
                )

        # Check that the lower-energy distorted structure wasn't already found with bond
        # distortions for the other charge states
        if (
            defect in low_energy_defects
        ):  # if an energy lowering distortion was found for this
            # defect
            for charge in defect_charges_dict[
                defect
            ]:  # for all charge states of the defect
                if (
                    charge not in low_energy_defects[defect]["charges"]
                ):  # if lower-energy
                    # distorted structure wasn't already found for that charge state
                    defect_name = f"{defect}_{charge}"
                    gs_struct_in_dict = low_energy_defects[defect]["structure"]
                    if compare_gs_struct_to_distorted_structs(
                        gs_struct_in_dict, defect_name, base_path, stol=stol
                    ):
                        # structure found in distortion tests for this charge state. Add it to the
                        # list to avoid redundant work
                        low_energy_defects[defect]["charges"].append(charge)
            # print("Ground-state structure found for {defect}_{low_energy_defects[defect][
            # 'charges'][0]} has been also found for the charge states: {low_energy_defects[
            # defect]['charges']}")
    return low_energy_defects


def compare_gs_struct_to_distorted_structs(  # TODO: Can we just use 'compare_structures' from
    # 'defects_analysis' module here?
    gs_struct: Structure,
    defect_species: str,
    base_path: str = "./",
    stol: float = 0.2,
) -> bool:
    """
    Compares the ground-state structure found for a certain defect charge state with all
    relaxed bond-distorted structures for `defect_species`, to avoid redundant work (testing
    this distorted structure for other charge states when it has already been found for them).

    Args:
        gs_struct (:obj:`~pymatgen.core.structure.Structure`):
            Structure of ground-state distorted defect
        defect_species (:obj:`str`):
            Defect name including charge (e.g. 'vac_1_Cd_0')
        base_path (:obj:`str`):
            Path to directory with your distorted defect calculations (to calculate structure
            comparisons – needs VASP CONTCAR files)

    Returns:
        True if a match is found between the input structure and the relaxed bond-distorted
        structures for `defect_species`, False otherwise.
    """
    sm = StructureMatcher(ltol=0.2, stol=stol)
    defect_structures = get_structures(defect_species, base_path)
    for key, structure in defect_structures.items():
        if gs_struct == "Not converged":
            print("Input gs_struct not converged")
            break
        elif structure != "Not converged":
            try:
                if sm.fit(gs_struct, structure):
                    return True  # as soon as a structure match is found, return True
            except AttributeError:
                print("Error matching structures")
        else:
            print(f"{key} structure not converged")
    return False  # structure match not found for this charge state


def import_deep_distortion_by_type(
    defect_dict: dict,
    low_energy_defects: dict,
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
    for k, v in defect_dict.items():
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
