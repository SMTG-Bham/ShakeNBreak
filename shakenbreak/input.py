"""
Module containing functions to generate rattled and bond-distorted structures,
as well as input files to run Gamma point relaxations with `VASP`, `CP2K`,
`Quantum-Espresso`, `FHI-aims` and `CASTEP`.
"""
import os
import copy
import json
import warnings
import datetime
from typing import Optional, Tuple
import functools
import numpy as np
from monty.serialization import loadfn

import ase
from ase.calculators.espresso import Espresso
from ase.calculators.castep import Castep
from ase.calculators.aims import Aims

from pymatgen.core.structure import Structure, Composition, Element
from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.io.vasp.sets import BadInputSetWarning
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cp2k.inputs import Cp2kInput

from shakenbreak import analysis, distortions, vasp, io

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


warnings.filterwarnings(
    "ignore", category=UnknownPotcarWarning
)  # Ignore pymatgen POTCAR warnings
warnings.filterwarnings("ignore", message=".*Ignoring unknown variable type.*")


def _warning_on_one_line(
    message, category, filename, lineno, file=None, line=None
) -> str:
    """Output warning messages on one line."""
    # To set this as warnings.formatwarning, we need to be able to take in `file`
    # and `line`, but don't want to print them, so unused arguments here
    return f"{os.path.split(filename)[-1]}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = _warning_on_one_line


# Helper functions
def _bold_print(string: str) -> None:
    """Prints the input string in bold."""
    print("\033[1m" + string + "\033[0m")


def _create_folder(folder_name: str) -> None:
    """Creates a folder at `./folder_name` if it doesn't already exist."""
    path = os.getcwd()
    if not os.path.isdir(path + "/" + folder_name):
        try:
            os.makedirs(path + "/" + folder_name, exist_ok=True)
        except OSError:
            print(f"Creation of the directory {path} failed")


def _write_distortion_metadata(
    new_metadata: dict,
    filename: str = "distortion_metadata.json",
    output_path: str = ".",
) -> None:
    """
    Write metadata to file. If the file already exists, it will be
    renamed to distortion_metadata_<datetime>.json and updated with new metadata.

    Args:
        new_metadata (:obj:`dict`):
            Distortion metadata containing distortion parameters used, as well as information
            about the defects and their charge states modelled.
        filename (:obj:`str`, optional):
            Filename to save metadata. Defaults to "distortion_metadata.json".
        output_path (:obj:`str`):
             Path to directory in which to write distortion_metadata.json file.
             (Default is current directory = "./")

    Returns:
        None
    """
    filepath = os.path.join(output_path, filename)
    if os.path.exists(filepath):
        current_datetime = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M"
        )  # keep copy of old metadata file
        os.rename(
            filepath,
            os.path.join(
                output_path, f"distortion_metadata" f"_{current_datetime}.json"
            ),
        )
        print(
            f"There is a previous version of {filename}. Will rename old metadata to "
            f"distortion_metadata_{current_datetime}.json"
        )
        try:
            print(f"Combining old and new metadata in {filename}.")
            with open(
                os.path.join(
                    output_path, f"distortion_metadata_{current_datetime}.json"
                ),
                "r",
            ) as old_metadata_file:
                old_metadata = json.load(old_metadata_file)
            # Combine old and new metadata dictionaries
            for defect in old_metadata["defects"]:
                if (
                    defect in new_metadata["defects"]
                ):  # if defect in both metadata files
                    for charge in new_metadata["defects"][defect]["charges"]:
                        if (
                            charge in old_metadata["defects"][defect]["charges"]
                        ):  # if charge state in both files,
                            # then we update the mesh of distortions
                            # (i.e. [-0.3, 0.3] + [-0.4, -0.2, 0.2, 0.4])
                            if (
                                new_metadata["defects"][defect]["charges"][charge]
                                == old_metadata["defects"][defect]["charges"][charge]
                            ):
                                # make sure there are no inconsistencies (same number of
                                # neighbours distorted and same distorted atoms)
                                new_metadata["defects"][defect]["charges"][charge][
                                    "distortion_parameters"
                                ] = {
                                    "bond_distortions": new_metadata["defects"][defect][
                                        "charges"
                                    ][charge]["distortion_parameters"][
                                        "bond_distortions"
                                    ]
                                    + old_metadata["defects"][defect]["charges"][
                                        charge
                                    ]["distortion_parameters"]["bond_distortions"]
                                }
                            else:  # different number of neighbours distorted in new run
                                warnings.warn(
                                    f"Previous and new metadata show different number of "
                                    f"distorted neighbours for {defect} in charge {charge}. "
                                    f"File {filepath} will only show the new number of distorted "
                                    f"neighbours."
                                )
                                continue
                        else:  # if charge state only in old metadata, add it to file
                            new_metadata["defects"][defect]["charges"][
                                charge
                            ] = old_metadata["defects"][defect]["charges"][charge]
                else:
                    new_metadata["defects"][defect] = old_metadata["defects"][
                        defect
                    ]  # else add new entry
        except KeyError:
            warnings.warn(
                f"There was a problem when combining old and new metadata files! Will only write "
                f"new metadata to {filepath}."
            )
    with open(filepath, "w") as new_metadata_file:
        new_metadata_file.write(json.dumps(new_metadata, indent=4))


def _create_vasp_input(
    defect_name: str,
    distorted_defect_dict: dict,
    incar_settings: dict,
    potcar_settings: Optional[dict] = None,
    output_path: str = ".",
) -> None:
    """
    Creates folders for storing VASP ShakeNBreak files.

    Args:
        defect_name (:obj:`str`):
            Folder name
        distorted_defect_dict (:obj:`dict`):
            Dictionary with the distorted structures of charged defect
        incar_settings (:obj:`dict`):
            Dictionary of user VASP INCAR settings, to overwrite/update the
            `doped` defaults
        potcar_settings (:obj:`dict`):
            Dictionary of user VASP POTCAR settings, to overwrite/update the
            `doped` defaults. Using `pymatgen` syntax (e.g. {'POTCAR':
            {'Fe': 'Fe_pv', 'O': 'O'}}).
        output_path (:obj:`str`):
            Path to directory in which to write distorted defect structures and
            calculation inputs.
            (Default is current directory = "./")

    Returns:
        None
    """
    # create folder for defect
    _create_folder(os.path.join(output_path, defect_name))
    for (
        distortion,
        single_defect_dict,
    ) in (
        distorted_defect_dict.items()
    ):  # for each distortion, create sub-subfolder folder
        potcar_settings_copy = copy.deepcopy(
            potcar_settings
        )  # files empties `potcar_settings dict` (via pop()), so make a
        # deepcopy each time
        vasp.write_vasp_gam_files(
            single_defect_dict=single_defect_dict,
            input_dir=f"{output_path}/{defect_name}/{distortion}",
            incar_settings=incar_settings,
            potcar_settings=potcar_settings_copy,
        )


def _get_bulk_comp(defect_dict) -> Composition:
    """
    Convenience function to determine the chemical composition of the bulk
    structure for a given defect. Useful for auto-determing oxidation states.

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the
            `doped.pycdt.core.defectsmaker.ChargedDefectsStructures()` format.

    Returns:
        Pymatgen Composition object for the bulk structure of the defect.
    """
    defect_struc = defect_dict["supercell"]["structure"].copy()
    defect_sites = defect_struc.sites
    defect_type = defect_dict["defect_type"]

    if defect_type == "vacancy":
        bulk_sites = defect_sites + [defect_dict["unique_site"]]

    elif defect_type == "interstitial":
        bulk_sites = defect_sites
        bulk_sites.remove(defect_dict["unique_site"])

    elif defect_type in ["substitution", "antisite"]:
        bulk_sites = defect_sites
        bulk_sites.remove(defect_dict["bulk_supercell_site"])
        bulk_sites += [defect_dict["unique_site"]]

    return Structure.from_sites(bulk_sites).composition


def _most_common_oxi(element) -> int:
    """
    Convenience function to get the most common oxidation state of an element, using pymatgen's
    elemental data.

    Args:
        element (:obj:`str`):
            Element symbol.

    Returns:
        Most common oxidation state of the element.
    """
    comp_obj = Composition("O")
    comp_obj.add_charges_from_oxi_state_guesses()
    element_obj = Element(element)
    oxi_probabilities = [
        (k, v) for k, v in comp_obj.oxi_prob.items() if k.element == element_obj
    ]
    most_common = max(oxi_probabilities, key=lambda x: x[1])[0]
    return most_common.oxi_state


def _calc_number_electrons(
    defect_dict: dict,
    oxidation_states: dict,
    verbose: bool = False,
) -> int:
    """
    Calculates the number of extra/missing electrons of the defect species
    (in `defect_dict`) based on `oxidation_states`.

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the
            `doped.pycdt.core.defectsmaker.ChargedDefectsStructures()` format.
        oxidation_states (:obj:`dict`):
            Dictionary with oxidation states of the atoms in the material (e.g.
            {"Cd": +2, "Te": -2}).
        verbose (:obj:`bool`):
            If True, prints the number of extra/missing electrons for the defect
            species.

    Returns:
        :obj:`int`:
            Extra/missing charge (negative of the number of extra/missing electrons).
    """
    oxidation_states["Vac"] = 0  # A vacancy has an oxidation state of zero

    # Determine number of extra/missing electrons based on defect type and
    # oxidation states
    if defect_dict["defect_type"] == "vacancy":
        site_specie = str(defect_dict["site_specie"])
        substituting_specie = "Vac"

    elif defect_dict["defect_type"] == "interstitial":
        substituting_specie = str(defect_dict["site_specie"])
        # Consider interstitials as substituting a vacant (zero oxidation-state
        # position)
        site_specie = "Vac"

    elif defect_dict["defect_type"] == "antisite":
        site_specie = str(defect_dict["site_specie"])
        substituting_specie = defect_dict["substituting_specie"]

    elif defect_dict["defect_type"] == "substitution":
        site_specie = str(defect_dict["site_specie"])
        substituting_specie = defect_dict["substitution_specie"]

    else:
        raise ValueError(
            "`defect_dict` has an invalid `defect_type`:"
            + f"{defect_dict['defect_type']}"
        )

    num_electrons = (
        oxidation_states[substituting_specie] - oxidation_states[site_specie]
    )

    if verbose:
        print(
            f"Number of extra/missing electrons of defect {defect_dict['name']}: "
            f"{int(num_electrons)} -> \u0394q = {int(-num_electrons)}"
        )

    return int(-num_electrons)


def _calc_number_neighbours(num_electrons: int) -> int:
    """
    Calculate the number of neighbours to distort based off the number of
    extra/missing electrons. An octet rule approach is used; if the electron
    count change is less than or equal to 4, we distort that number of
    neighbours, if it is greater than 4, then we distort (8 -(electron
    change) neighbours.

    Args:
        num_electrons (:obj:`int`): Number of extra/missing electrons for the
            defect species.

    Returns:
        :obj:`int`:
            Number of neighbours to distort
    """
    if abs(num_electrons) > 4:
        num_neighbours = abs(8 - abs(num_electrons))
    else:
        num_neighbours = abs(num_electrons)
    return abs(num_neighbours)


# Main functions


def _apply_rattle_bond_distortions(
    defect_dict: dict,
    num_nearest_neighbours: int,
    distortion_factor: float,
    local_rattle: bool = False,
    stdev: float = 0.25,
    d_min: Optional[float] = None,
    active_atoms: Optional[list] = None,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """
    Applies rattle and bond distortions to the unperturbed defect structure
    (in `defect_dict`) by calling `distortion.distort` with either:
            - fractional coordinates (for vacancies) or
            - defect site index (other defect types).

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the format of
            `doped.vasp_input.prepare_vasp_defect_dict`
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to.
        distortion_factor (:obj:`float`):
            The distortion factor to apply to the bond distance between
            the defect and nearest neighbours. Typical choice is between
            0.4 (-60%) and 1.4 (+60%).
        local_rattle (:obj:`bool`):
            Whether to apply random displacements that tail off as we move
            away from the defect site. If False, all supercell sites are
            rattled with the same amplitude.
            (Default: False)
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution
            from which atomic displacement distances are drawn.
            (Default: 0.25)
        d_min (:obj:`float`):
            Minimum interatomic distance (in Angstroms) in the rattled
            structure. Monte Carlo rattle moves that put atoms at
            distances less than this will be heavily penalised.
            Default is to set this to 80% of the nearest neighbour
            distance in the defect supercell (ignoring interstitials).
        active_atoms (:obj:`list`, optional):
            List (or array) of which atomic indices should undergo Monte Carlo
            rattling. Default is to apply rattle to all atoms except the
            defect and the bond-distorted neighbours.
        distorted_element (:obj:`str`, optional):
            Neighbouring element to distort. If None, the closest neighbours
            to the defect will be chosen.
            (Default: None)
        verbose (:obj:`bool`):
            Whether to print distortion information.
            (Default: False)
        **kwargs:
            Additional keyword arguments to pass to `hiphive`'s
            `mc_rattle` function. These include:
            - d_min (:obj:`float`):
                Minimum interatomic distance (in Angstroms). Monte Carlo rattle
                moves that put atoms at distances less than this will be heavily
                penalised.
                (Default: 2.25)
            - max_disp (:obj:`float`):
                Maximum atomic displacement (in Angstroms) during Monte Carlo
                rattling. Rarely occurs and is used primarily as a safety net.
                (Default: 2.0)
            - max_attempts (:obj:`int`):
                Limit for how many attempted rattle moves are allowed a single atom.
            - active_atoms (:obj:`list`):
                List of the atomic indices which should undergo Monte
                Carlo rattling. By default, all atoms are rattled.
                (Default: None)
            - seed (:obj:`int`):
                Seed for setting up NumPy random state from which random
                numbers are generated.

    Returns:
        :obj:`dict`:
            Dictionary with distorted defect structure and the distortion
            parameters.
    """
    # Apply bond distortions to defect neighbours:
    if (
        defect_dict["defect_type"] == "vacancy"
    ):  # for vacancies, we need to use fractional coordinates
        # (no atom site in structure!)
        frac_coords = defect_dict["bulk_supercell_site"].frac_coords
        defect_site_index = None
        bond_distorted_defect = distortions.distort(
            structure=defect_dict["supercell"]["structure"],
            num_nearest_neighbours=num_nearest_neighbours,
            distortion_factor=distortion_factor,
            frac_coords=frac_coords,
            distorted_element=distorted_element,
            verbose=verbose,
        )
    else:
        defect_site_index = len(
            defect_dict["supercell"]["structure"]
        )  # defect atom comes last in structure, using doped or pymatgen
        frac_coords = None  # only for vacancies
        bond_distorted_defect = distortions.distort(
            structure=defect_dict["supercell"]["structure"],
            num_nearest_neighbours=num_nearest_neighbours,
            distortion_factor=distortion_factor,
            site_index=defect_site_index,
            distorted_element=distorted_element,
            verbose=verbose,
        )

    # Apply rattle to the bond distorted structure
    if not d_min:
        defect_supercell = defect_dict["supercell"]["structure"]
        sorted_distances = np.sort(defect_supercell.distance_matrix.flatten())
        d_min = (
            0.8 * sorted_distances[len(defect_supercell) + 20]
        )  # ignoring interstitials by
        # ignoring the first 10 non-zero bond lengths (double counted in the distance matrix)
        if d_min < 1:
            warnings.warn(
                f"Automatic bond-length detection gave a bulk bond length of "
                f"{(1/0.8)*d_min} \u212B, which is almost certainly too small. "
                f"Reverting to 2.25 \u212B. If this is too large, set `d_min` manually"
            )
            d_min = 2.25

    if active_atoms is None:
        distorted_atom_indices = [
            i[0] for i in bond_distorted_defect["distorted_atoms"]
        ] + [
            bond_distorted_defect.get(
                "defect_site_index"
            )  # only adds defect site if not vacancy
        ]  # Note this is VASP indexing here
        distorted_atom_indices = [
            i - 1 for i in distorted_atom_indices if i is not None
        ]  # remove
        # 'None' if defect is vacancy, and convert to python indexing
        rattling_atom_indices = np.arange(0, len(defect_dict["supercell"]["structure"]))
        idx = np.in1d(
            rattling_atom_indices, distorted_atom_indices
        )  # returns True for matching indices
        active_atoms = rattling_atom_indices[~idx]  # remove matching indices

    try:
        if local_rattle:
            bond_distorted_defect["distorted_structure"] = distortions.local_mc_rattle(
                structure=bond_distorted_defect["distorted_structure"],
                frac_coords=frac_coords,
                site_index=defect_site_index,
                stdev=stdev,
                d_min=d_min,
                active_atoms=active_atoms,
                **kwargs,
            )
        else:
            bond_distorted_defect["distorted_structure"] = distortions.rattle(
                structure=bond_distorted_defect["distorted_structure"],
                stdev=stdev,
                d_min=d_min,
                active_atoms=active_atoms,
                **kwargs,
            )
    except Exception as ex:
        if "attempts" in str(ex):
            distorted_defect_struc = bond_distorted_defect["distorted_structure"]
            sorted_distances = np.sort(distorted_defect_struc.distance_matrix.flatten())
            reduced_d_min = sorted_distances[len(distorted_defect_struc)] + (1 * stdev)
            if local_rattle:
                bond_distorted_defect[
                    "distorted_structure"
                ] = distortions.local_mc_rattle(
                    structure=bond_distorted_defect["distorted_structure"],
                    frac_coords=frac_coords,
                    site_index=defect_site_index,
                    stdev=stdev,
                    d_min=reduced_d_min,  # min distance in supercell plus 1 stdev
                    active_atoms=active_atoms,
                    max_attempts=7000,  # default is 5000
                    **kwargs,
                )
            else:
                bond_distorted_defect["distorted_structure"] = distortions.rattle(
                    structure=bond_distorted_defect["distorted_structure"],
                    stdev=stdev,
                    d_min=reduced_d_min,  # min distance in supercell plus 1 stdev
                    active_atoms=active_atoms,
                    max_attempts=7000,  # default is 5000
                    **kwargs,
                )
            if verbose:
                warnings.warn(
                    f"Initial rattle with d_min {d_min:.2f} \u212B failed (some bond lengths "
                    f"significantly smaller than this present), setting d_min to "
                    f"{reduced_d_min:.2f} \u212B for this defect."
                )
        else:
            raise ex

    return bond_distorted_defect


def apply_snb_distortions(
    defect_dict: dict,
    num_nearest_neighbours: int,
    bond_distortions: list,
    local_rattle: bool = False,
    stdev: float = 0.25,
    d_min: Optional[float] = None,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """
    Applies rattle and bond distortions to `num_nearest_neighbours` of the
    unperturbed defect structure (in `defect_dict`).

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the format of
            `doped.vasp_input.prepare_vasp_defect_dict`
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to
        bond_distortions (:obj:`list`):
            List of specific distortions to apply to defect nearest neighbours.
            (e.g. [-0.5, 0.5])
        local_rattle (:obj:`bool`):
            Whether to apply random displacements that tail off as we move
            away from the defect site. If False, all supercell sites are
            rattled with the same amplitude.
            (Default: False)
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution
            from which atomic displacement distances are drawn.
            (Default: 0.25)
        d_min (:obj:`float`, optional):
            Minimum interatomic distance (in Angstroms) in the rattled
            structure. Monte Carlo rattle moves that put atoms at distances
            less than this will be heavily penalised. Default is to set this
            to 80% of the nearest neighbour distance in the bulk supercell.
        distorted_element (:obj:`str`, optional):
            Neighbouring element to distort. If None, the closest neighbours
            to the defect will be chosen.
            (Default: None)
        verbose (:obj:`bool`):
            Whether to print distortion information.
            (Default: False)
        **kwargs:
            Additional keyword arguments to pass to `hiphive`'s
            `mc_rattle` function. These include:
            - d_min (:obj:`float`):
                Minimum interatomic distance (in Angstroms). Monte Carlo rattle
                moves that put atoms at distances less than this will be heavily
                penalised.
                (Default: 2.25)
            - max_disp (:obj:`float`):
                Maximum atomic displacement (in Angstroms) during Monte Carlo
                rattling. Rarely occurs and is used primarily as a safety net.
                (Default: 2.0)
            - max_attempts (:obj:`int`):
                Limit for how many attempted rattle moves are allowed a single atom.
            - active_atoms (:obj:`list`):
                List of the atomic indices which should undergo Monte
                Carlo rattling. By default, all atoms are rattled.
                (Default: None)
            - seed (:obj:`int`):
                Seed for setting up NumPy random state from which random
                numbers are generated.

    Returns:
        :obj:`dict`:
            Dictionary with distorted defect structure and the distortion
            parameters.
    """
    distorted_defect_dict = {
        "Unperturbed": defect_dict,
        "distortions": {},
        "distortion_parameters": {},
    }

    if num_nearest_neighbours != 0:
        for distortion in bond_distortions:
            distortion = (
                round(distortion, ndigits=3) + 0
            )  # ensure positive zero (not "-0.0%")
            if verbose:
                print(f"--Distortion {distortion:.1%}")
            distortion_factor = 1 + distortion
            bond_distorted_defect = _apply_rattle_bond_distortions(
                defect_dict=defect_dict,
                num_nearest_neighbours=num_nearest_neighbours,
                distortion_factor=distortion_factor,
                local_rattle=local_rattle,
                stdev=stdev,
                d_min=d_min,
                distorted_element=distorted_element,
                verbose=verbose,
                **kwargs,
            )
            distorted_defect_dict["distortions"][
                analysis._get_distortion_filename(distortion)
            ] = bond_distorted_defect["distorted_structure"]
            distorted_defect_dict["distortion_parameters"] = {
                "unique_site": defect_dict["bulk_supercell_site"].frac_coords,
                "num_distorted_neighbours": num_nearest_neighbours,
                "distorted_atoms": bond_distorted_defect["distorted_atoms"],
            }
            if bond_distorted_defect.get(
                "defect_site_index"
            ):  # only add site index if vacancy
                distorted_defect_dict["distortion_parameters"][
                    "defect_site_index"
                ] = bond_distorted_defect["defect_site_index"]

    elif (
        num_nearest_neighbours == 0
    ):  # when no extra/missing electrons, just rattle the structure.
        # Likely to be a shallow defect.
        if defect_dict["defect_type"] == "vacancy":
            defect_site_index = None
            frac_coords = defect_dict["bulk_supercell_site"].frac_coords
        else:
            frac_coords = None  # only for vacancies!
            defect_site_index = len(
                defect_dict["supercell"]["structure"]
            )  # defect atom comes last in structure
        if not d_min:
            defect_supercell = defect_dict["supercell"]["structure"]
            sorted_distances = np.sort(defect_supercell.distance_matrix.flatten())
            d_min = (
                0.8 * sorted_distances[len(defect_supercell) + 20]
            )  # ignoring interstitials by
            # ignoring the first 10 non-zero bond lengths (double counted in
            # the distance matrix)
            if d_min < 1:
                warnings.warn(
                    f"Automatic bond-length detection gave a bulk bond length of "
                    f"{(1 / 0.8) * d_min} \u212B, which is almost certainly too small. "
                    f"Reverting to 2.25 \u212B. If this is too large, set `d_min` "
                    f"manually"
                )
                d_min = 2.25
        if local_rattle:
            perturbed_structure = distortions.local_mc_rattle(
                defect_dict["supercell"]["structure"],
                site_index=defect_site_index,
                frac_coords=frac_coords,
                stdev=stdev,
                d_min=d_min,
                **kwargs,
            )
        else:
            perturbed_structure = distortions.rattle(
                defect_dict["supercell"]["structure"],
                stdev=stdev,
                d_min=d_min,
                **kwargs,
            )
        distorted_defect_dict["distortions"]["Rattled"] = perturbed_structure
        distorted_defect_dict["distortion_parameters"] = {
            "unique_site": defect_dict["bulk_supercell_site"].frac_coords,
            "num_distorted_neighbours": num_nearest_neighbours,
            "distorted_atoms": None,
        }
        if defect_site_index:  # only add site index if vacancy
            distorted_defect_dict["distortion_parameters"][
                "defect_site_index"
            ] = defect_site_index
    return distorted_defect_dict


class Distortions:
    """
    Class to apply rattle and bond distortion to all defects in `defects_dict`
    (in `doped` `ChargedDefectsStructures()` format).
    """

    def __init__(
        self,
        defects_dict: dict,
        oxidation_states: Optional[dict] = None,
        dict_number_electrons_user: Optional[dict] = None,
        distortion_increment: float = 0.1,
        bond_distortions: Optional[list] = None,
        local_rattle: bool = True,
        stdev: float = 0.25,
        distorted_elements: Optional[dict] = None,
        **kwargs,  # for mc rattle
    ):
        """
        Args:
            defects_dict (:obj:`dict`):
                Dictionary of defects as generated with `doped`
                `ChargedDefectsStructures()`
            oxidation_states (:obj:`dict`):
                Dictionary of oxidation states for species in your material,
                used to determine the number of defect neighbours to distort
                (e.g {"Cd": +2, "Te": -2}). If none is provided, the oxidation
                states will be guessed based on the bulk composition and most
                common oxidation states of any extrinsic species.
            dict_number_electrons_user (:obj:`dict`):
                Optional argument to set the number of extra/missing charge
                (negative of electron count change) for the input defects
                in their neutral state, as a dictionary with format
                {'defect_name': charge_change} where charge_change is the
                negative of the number of extra/missing electrons.
                (Default: None)
            distortion_increment (:obj:`float`):
                Bond distortion increment. Distortion factors will range from
                0 to +/-0.6, in increments of `distortion_increment`.
                Recommended values: 0.1-0.3
                (Default: 0.1)
            bond_distortions (:obj:`list`):
                List of bond distortions to apply to nearest neighbours,
                instead of the default set (e.g. [-0.5, 0.5]).
                (Default: None)
            local_rattle (:obj:`bool`):
                Whether to apply random displacements that tail off as we move
                away from the defect site. Recommended as it is often faster than
                the full rattle (requires less ionic relaxation steps). If False,
                all supercell sites are rattled with the same amplitude (full ratlle).
                (Default: True)
            stdev (:obj:`float`):
                Standard deviation (in Angstroms) of the Gaussian distribution
                from which random atomic displacement distances are drawn during
                rattling. Recommended values: 0.25, or 0.15 for strongly-bound
                /ionic materials.
                (Default: 0.25)
            distorted_elements (:obj:`dict`):
                Optional argument to specify the neighbouring elements to
                distort for each defect, in the form of a dictionary with
                format {'defect_name': ['element1', 'element2', ...]}
                (e.g {'vac_1_Cd': ['Te']}). If None, the closest neighbours to
                the defect are chosen.
                (Default: None)
            **kwargs:
                Additional keyword arguments to pass to `hiphive`'s
                `mc_rattle` function. These include:
                - d_min (:obj:`float`):
                    Minimum interatomic distance (in Angstroms). Monte Carlo rattle
                    moves that put atoms at distances less than this will be heavily
                    penalised.
                    (Default: 2.25)
                - max_disp (:obj:`float`):
                    Maximum atomic displacement (in Angstroms) during Monte Carlo
                    rattling. Rarely occurs and is used primarily as a safety net.
                    (Default: 2.0)
                - max_attempts (:obj:`int`):
                    Limit for how many attempted rattle moves are allowed a single atom.
                - active_atoms (:obj:`list`):
                    List of the atomic indices which should undergo Monte
                    Carlo rattling. By default, all atoms are rattled.
                    (Default: None)
                - seed (:obj:`int`):
                    Seed for setting up NumPy random state from which random
                    numbers are generated.

        """
        self.defects_dict = defects_dict
        self.oxidation_states = oxidation_states
        self.distorted_elements = distorted_elements
        self.dict_number_electrons_user = dict_number_electrons_user
        self.stdev = stdev
        self.local_rattle = local_rattle

        if oxidation_states is None:
            if "bulk" in self.defects_dict:
                bulk_comp = self.defects_dict["bulk"]["supercell"][
                    "structure"
                ].composition
                self.oxidation_states = bulk_comp.oxi_state_guesses()[0]

            else:  # determine bulk composition from first defect in dict
                defect_subdict = list(self.defects_dict.values())[0][0]
                bulk_comp = _get_bulk_comp(defect_subdict)
                self.oxidation_states = bulk_comp.oxi_state_guesses()[0]

            if "substitutions" in self.defects_dict:
                for substitution in self.defects_dict["substitutions"]:
                    if (
                        substitution["defect_type"] == "substitution"
                        and substitution["bulk_supercell_site"].specie.symbol
                        not in self.oxidation_states
                    ):
                        # substituting species not in bulk composition
                        substitution_specie = substitution["substitution_specie"]
                        likely_substitution_oxi = _most_common_oxi(substitution_specie)
                        self.oxidation_states[
                            substitution_specie
                        ] = likely_substitution_oxi

            print(
                f"Oxidation states were not explicitly set, thus have been "
                f"guessed as {self.oxidation_states}. If this is unreasonable "
                f"you should manually set oxidation_states"
            )

        if bond_distortions:
            self.distortion_increment = None  # user specified
            #  bond_distortions, so no increment
            self.bond_distortions = list(
                np.around(bond_distortions, 3)
            )  # round to 3 decimal places
        else:
            # If the user does not specify bond_distortions, use
            # distortion_increment:
            self.distortion_increment = distortion_increment
            self.bond_distortions = list(
                np.flip(
                    np.around(
                        np.arange(0, 0.601, self.distortion_increment), decimals=3
                    )
                )
                * -1
            )[:-1] + list(
                np.around(np.arange(0, 0.601, self.distortion_increment), decimals=3)
            )

        self._mc_rattle_kwargs = kwargs

        # Create dictionary to keep track of the bond distortions applied
        self.distortion_metadata = {
            "distortion_parameters": {
                "distortion_increment": self.distortion_increment,  # None if
                # user specified bond_distortions
                "bond_distortions": self.bond_distortions,
                "rattle_stdev": self.stdev,
                "local_rattle": self.local_rattle,
            },
            "defects": {},
        }  # dict with distortion parameters, useful for posterior analysis

    def _parse_distorted_element(
        self,
        defect_name,
        distorted_elements: Optional[dict],
    ) -> str:
        """
        Parse the user-defined distorted elements for a given defect
        (if given).

        Args:
            defect_name (:obj:`str`):
                Name of the defect for which to parse the distorted elements.
            distorted_elements (:obj:`dict`):
                Dictionary of distorted elements for each defect, in the form
                of {'defect_name': ['element1', 'element2', ...]}
                (e.g {'vac_1_Cd': ['Te']}).
        """
        # Specific elements to distort
        if distorted_elements:
            try:
                distorted_element = distorted_elements[defect_name]
            except KeyError:
                warnings.warn(
                    "Problem reading the keys in distorted_elements.",
                    "Are they the correct defect names (without charge states)?",
                    "Proceeding without discriminating which neighbour "
                    + "elements to distort.",
                )
                distorted_element = None
        else:
            distorted_element = None
        return distorted_element

    def _parse_number_electrons(
        self,
        defect_name: str,
        oxidation_states: dict,
        dict_number_electrons_user: dict,
        defect: dict,
    ) -> int:
        """
        Parse or calculate the number of extra/missing electrons
        for a neutral defect, and print this information.

        Args:
            defect_name (:obj:`str`):
                Name of the defect for which to parse the distorted elements.
            oxidation_states (:obj:`dict`):
                Dictionary of oxidation states for species in your material,
                used to determine the number of defect neighbours to distort
                (e.g {"Cd": +2, "Te": -2}).
            dict_number_electrons_user (:obj:`dict`):
                Optional argument to set the number of extra/missing charge
                (negative of electron count change) for the input defects,
                as a dictionary with format {'defect_name': charge_change}
                where charge_change is the negative of the number of
                extra/missing electrons.
            defect (:obj:`dict`):
                Defect entry in dictionary of defects as generated with
                `doped` `ChargedDefectsStructures()`.

        Returns:
            :obj:`int`:
                Number of extra/missing electrons for the defect.
        """
        # If the user does not specify the electron count change, we calculate it:
        if dict_number_electrons_user:
            number_electrons = dict_number_electrons_user[defect_name]
        else:
            number_electrons = _calc_number_electrons(defect, oxidation_states)

        _bold_print(f"\nDefect: {defect_name}")
        if number_electrons < 0:
            _bold_print(
                "Number of extra electrons in neutral state: "
                + f"{abs(number_electrons)}"
            )
        elif number_electrons >= 0:
            _bold_print(
                f"Number of missing electrons in neutral state: {number_electrons}"
            )
        return number_electrons

    def _get_number_distorted_neighbours(
        self,
        defect_name: str,
        number_electrons: int,
        charge: int,
    ) -> int:
        """
        Calculate extra/missing electrons accounting for the charge state of
        the defect.
        """
        num_electrons_charged_defect = (
            number_electrons + charge
        )  # negative if extra e-, positive if missing e-
        num_nearest_neighbours = _calc_number_neighbours(
            num_electrons_charged_defect
        )  # Number of distorted neighbours for each charge state
        print(
            f"\nDefect {defect_name} in charge state: {'+' if charge > 0 else ''}{charge}. "
            f"Number of distorted neighbours: {num_nearest_neighbours}"
        )
        return num_nearest_neighbours

    def _print_distortion_info(
        self,
        bond_distortions: list,
        stdev: float,
    ) -> None:
        """Print applied bond distortions and rattle standard deviation."""
        print(
            "Applying ShakeNBreak...",
            "Will apply the following bond distortions:",
            f"{[f'{round(i,3)+0}' for i in bond_distortions]}.",
            f"Then, will rattle with a std dev of {stdev} \u212B \n",
        )

    def _update_distortion_metadata(
        self,
        distortion_metadata: dict,
        defect_name: str,
        charge: int,
        defect_site_index: int,
        num_nearest_neighbours: int,
        distorted_atoms: list,
    ) -> dict:
        """
        Update distortion_metadata with distortion information for each
        charged defect.
        """
        if defect_site_index:
            distortion_metadata["defects"][defect_name][
                "defect_site_index"
            ] = defect_site_index  # store site index of defect if not vacancy
        distortion_metadata["defects"][defect_name]["charges"].update(
            {
                int(charge): {
                    "num_nearest_neighbours": num_nearest_neighbours,
                    "distorted_atoms": distorted_atoms,
                    "distortion_parameters": {
                        "bond_distortions": self.bond_distortions,
                        # store distortions used for each charge state,
                        "rattle_stdev": self.stdev,
                        # in case posterior runs use finer mesh for only
                        # certain defects
                    },
                }
            }
        )
        return distortion_metadata

    def _generate_structure_comment(
        self,
        key_distortion: str,
        charge: int,
        defect_name: str,
    ) -> str:
        """Generate comment for structure files."""
        poscar_comment = (
            str(
                key_distortion.split("_")[-1]
            )  # Get distortion factor (-60.%) or 'Rattled'
            + "__num_neighbours="
            + str(
                self.distortion_metadata["defects"][defect_name]["charges"][charge][
                    "num_nearest_neighbours"
                ]
            )
            + "_"
            + defect_name
        )
        return poscar_comment

    def _setup_distorted_defect_dict(
        self,
        defect: dict,
    ) -> dict:
        """
        Setup `distorted_defect_dict` with info for `defect`.

        Args:
            defect (:obj:`dict`):
                Defect dictionary to generate `distorted_defect_dict` from.

        Returns:
            :obj:`dict`
                Dictionary with information for `defect`.
        """
        distorted_defect_dict = {
            "defect_type": defect["name"],
            "defect_site": defect["unique_site"],
            "defect_supercell_site": defect["bulk_supercell_site"],
            "defect_multiplicity": defect["site_multiplicity"],
            "supercell": defect["supercell"]["size"],
            "charges": {charge: {} for charge in defect["charges"]},
        }  # General info about (neutral) defect
        for key in [
            "substitution_specie",
            "substituting_specie",
        ]:  # substitutions and antisites
            if key in defect:
                distorted_defect_dict[key] = defect[key]
        return distorted_defect_dict

    def write_distortion_metadata(
        self,
        output_path=".",
    ) -> None:
        """
        Write metadata to file. If the file already exists, it will be
        renamed to distortion_metadata_<datetime>.json and updated with
        new metadata.

        Args:
            output_path (:obj:`str`):
                Path to directory where the metadata file will be written.

        Returns:
            None
        """
        _write_distortion_metadata(
            new_metadata=self.distortion_metadata,
            filename="distortion_metadata.json",
            output_path=output_path,
        )

    def apply_distortions(
        self,
        verbose: bool = False,
    ) -> Tuple[dict, dict]:
        """
        Applies rattle and bond distortion to all defects in `defect_dict`
        (in `doped ChargedDefectsStructures()` format).
        Returns a dictionary with the distorted (and undistorted) structures
        for each charge state of each defect.
        If file generation is desired, instead use the methods
        `write_<code>_files()`.

        Args:
            verbose (:obj:`bool`):
                Whether to print distortion information (bond atoms and distances)
                for each charged defect.
                (Default: False)

        Returns:
            :obj:`tuple`:
                Tuple of:
                Dictionary with the distorted and undistorted structures
                for each charge state of each defect, in the format:
                {'defect_name': {
                    'charges': {
                        'charge_state': {
                            'structures': {...},
                        },
                    },
                }
                and dictionary with distortion parameters for each defect.
        """
        self._print_distortion_info(
            bond_distortions=self.bond_distortions, stdev=self.stdev
        )

        distorted_defects_dict = {}  # Store distorted & undistorted structures

        comb_defs = functools.reduce(
            lambda x, y: x + y,
            [self.defects_dict[key] for key in self.defects_dict if key != "bulk"],
        )

        for defect in comb_defs:  # loop for each defect
            defect_name = defect["name"]  # name without charge state
            bulk_supercell_site = defect["bulk_supercell_site"]

            # Parse distortion specifications given by user for neutral
            # defect and use ShakeNBreak defaults if not given
            distorted_element = self._parse_distorted_element(
                defect_name=defect_name,
                distorted_elements=self.distorted_elements,
            )
            number_electrons = self._parse_number_electrons(
                defect_name=defect_name,
                oxidation_states=self.oxidation_states,
                dict_number_electrons_user=self.dict_number_electrons_user,
                defect=defect,
            )

            self.distortion_metadata["defects"][defect_name] = {
                "unique_site": list(bulk_supercell_site.frac_coords),
                "charges": {},
            }

            distorted_defects_dict[defect_name] = self._setup_distorted_defect_dict(
                defect
            )

            for charge in defect["charges"]:  # loop for each charge state of defect
                num_nearest_neighbours = self._get_number_distorted_neighbours(
                    defect_name=defect_name,
                    number_electrons=number_electrons,
                    charge=charge,
                )
                # Generate distorted structures
                defect_distorted_structures = apply_snb_distortions(
                    defect_dict=defect,
                    num_nearest_neighbours=num_nearest_neighbours,
                    bond_distortions=self.bond_distortions,
                    local_rattle=self.local_rattle,
                    stdev=self.stdev,
                    distorted_element=distorted_element,
                    verbose=verbose,
                    **self._mc_rattle_kwargs,
                )

                # Add distorted structures to dictionary
                distorted_defects_dict[defect_name]["charges"][charge]["structures"] = {
                    "Unperturbed": defect_distorted_structures["Unperturbed"][
                        "supercell"
                    ]["structure"],
                    "distortions": {
                        dist: struct
                        for dist, struct in defect_distorted_structures[
                            "distortions"
                        ].items()
                    },
                }

                # Store distortion parameters/info in self.distortion_metadata
                defect_site_index = defect_distorted_structures[
                    "distortion_parameters"
                ].get("defect_site_index")
                self.distortion_metadata = self._update_distortion_metadata(
                    distortion_metadata=self.distortion_metadata,
                    defect_name=defect_name,
                    charge=charge,
                    defect_site_index=defect_site_index,
                    num_nearest_neighbours=num_nearest_neighbours,
                    distorted_atoms=defect_distorted_structures[
                        "distortion_parameters"
                    ]["distorted_atoms"],
                )

        return distorted_defects_dict, self.distortion_metadata

    def write_vasp_files(
        self,
        incar_settings: Optional[dict] = None,
        potcar_settings: Optional[dict] = None,
        output_path: str = ".",
        verbose: bool = False,
    ) -> Tuple[dict, dict]:
        """
        Generates the input files for `vasp_gam` relaxations of all output
        structures.

        Args:

            incar_settings (:obj:`dict`):
                Dictionary of user VASP INCAR settings (e.g.
                {"ENCUT": 300, ...}), to overwrite the `ShakenBreak` defaults
                for those tags. Highly recommended to look at output `INCAR`s,
                or `SnB_input_files/incar.yaml` to see what the default `INCAR`
                settings are.
                (Default: None)
            potcar_settings (:obj:`dict`):
                Dictionary of user VASP POTCAR settings, to overwrite/update
                the `doped` defaults. Using `pymatgen` syntax
                (e.g. {'POTCAR': {'Fe': 'Fe_pv', 'O': 'O'}}). Highly
                recommended to look at output `POTCAR`s, or `shakenbreak`
                `SnB_input_files/default_POTCARs.yaml`, to see what the default
                `POTCAR` settings are.
                (Default: None)
            write_files (:obj:`bool`):
                Whether to write output files (Default: True)
            output_path (:obj:`str`):
                Path to directory in which to write distorted defect structures
                and calculation inputs.
                (Default is current directory = ".")
            verbose (:obj:`bool`):
                Whether to print distortion information (bond atoms and
                distances).
                (Default: False)

        Returns:
            :obj:`tuple`:
                tuple of dictionaries with new defects_dict (containing the
                distorted structures) and defect distortion parameters.
        """
        distorted_defects_dict, self.distortion_metadata = self.apply_distortions(
            verbose=verbose,
        )

        warnings.filterwarnings(
            "ignore", category=BadInputSetWarning
        )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting
        # POTCAR types

        # loop for each defect in dict
        for defect_name, defect_dict in distorted_defects_dict.items():

            dict_transf = {
                k: v for k, v in defect_dict.items() if k != "charges"
            }  # Single defect dict
            charged_defect = {}

            # loop for each charge state of defect
            for charge in defect_dict["charges"]:

                for key_distortion, struct in zip(
                    [
                        "Unperturbed",
                    ]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    [defect_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    poscar_comment = self._generate_structure_comment(
                        defect_name=defect_name,
                        charge=charge,
                        key_distortion=key_distortion,
                    )

                    charged_defect[key_distortion] = {
                        "Defect Structure": struct,
                        "POSCAR Comment": poscar_comment,
                        "Transformation Dict": copy.deepcopy(dict_transf),
                    }
                    charged_defect[key_distortion]["Transformation Dict"].update(
                        {"charge": charge}
                    )

                _create_vasp_input(
                    defect_name=f"{defect_name}_{charge}",
                    distorted_defect_dict=charged_defect,
                    incar_settings=incar_settings,
                    potcar_settings=potcar_settings,
                    output_path=output_path,
                )

        self.write_distortion_metadata(output_path=output_path)
        return distorted_defects_dict, self.distortion_metadata

    def write_espresso_files(
        self,
        pseudopotentials: Optional[dict] = None,
        input_parameters: Optional[str] = None,
        input_file: Optional[str] = None,
        write_structures_only: Optional[bool] = False,
        output_path: str = ".",
        verbose: Optional[bool] = False,
    ) -> Tuple[dict, dict]:
        """
        Generates input files for Quantum Espresso relaxations of all output
        structures.

        Args:
            pseudopotentials (:obj:`dict`, optional):
                Dictionary matching element to pseudopotential name.
                (Defaults: None)
            input_parameters (:obj:`dict`, optional):
                Dictionary of user Quantum Espresso input parameters, to
                overwrite/update `shakenbreak` default ones (see
                `SnB_input_files/qe_input.yaml`).
                (Default: None)
            input_file (:obj:`str`, optional):
                Path to Quantum Espresso input file, to overwrite/update
                `shakenbreak` default ones (see `SnB_input_files/qe_input.yaml`).
                If both `input_parameters` and `input_file` are provided,
                the input_parameters will be used.
            write_structures_only (:obj:`bool`, optional):
                Whether to only write the structure files (in CIF format)
                (without calculation inputs).
                (Default: False)
            output_path (:obj:`str`, optional):
                Path to directory in which to write distorted defect structures
                and calculation inputs.
                (Default is current directory: ".")
            verbose (:obj:`bool`):
                Whether to print distortion information (bond atoms and
                distances).
                (Default: False)

        Returns:
            :obj:`tuple`:
                Tuple of dictionaries with new defects_dict (containing the
                distorted structures) and defect distortion parameters.
        """
        distorted_defects_dict, self.distortion_metadata = self.apply_distortions(
            verbose=verbose,
        )

        # Update default parameters with user defined values
        if pseudopotentials and not write_structures_only:
            default_input_parameters = loadfn(
                os.path.join(MODULE_DIR, "../SnB_input_files/qe_input.yaml")
            )
            if input_file and not input_parameters:
                input_parameters = io.parse_qe_input(input_file)
            if input_parameters:
                for section in input_parameters:
                    for key in input_parameters[section]:
                        if section in default_input_parameters:
                            default_input_parameters[section][key] = input_parameters[
                                section
                            ][key]
                        else:
                            default_input_parameters.update(
                                {section: {key: input_parameters[section][key]}}
                            )

        aaa = AseAtomsAdaptor()

        # loop for each defect in dict
        for defect_name, defect_dict in distorted_defects_dict.items():

            for charge in defect_dict["charges"]:  # loop for each charge state

                for dist, struct in zip(
                    [
                        "Unperturbed",
                    ]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    [defect_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    atoms = aaa.get_atoms(struct)
                    _create_folder(f"{output_path}/{defect_name}_{charge}/{dist}")

                    if not pseudopotentials or write_structures_only:
                        # only write structures
                        warnings.warn(
                            "Since `pseudopotentials` have not been specified, "
                            "will only write input structures."
                        )
                        ase.io.write(
                            filename=f"{output_path}/"
                            + f"{defect_name}_{charge}/{dist}/espresso.pwi",
                            images=atoms,
                            format="espresso-in",
                        )
                    elif pseudopotentials and not write_structures_only:
                        # write complete input file
                        default_input_parameters["SYSTEM"][
                            "tot_charge"
                        ] = charge  # Update defect charge

                        calc = Espresso(
                            pseudopotentials=pseudopotentials,
                            tstress=False,
                            tprnfor=True,
                            kpts=(1, 1, 1),
                            input_data=default_input_parameters,
                        )
                        calc.write_input(atoms)
                        os.replace(
                            "./espresso.pwi",
                            f"{output_path}/"
                            + f"{defect_name}_{charge}/{dist}/espresso.pwi",
                        )
        return distorted_defects_dict, self.distortion_metadata

    def write_cp2k_files(
        self,
        input_file: Optional[str] = f"{MODULE_DIR}/../SnB_input_files/cp2k_input.inp",
        write_structures_only: Optional[bool] = False,
        output_path: str = ".",
        verbose: Optional[bool] = False,
    ) -> Tuple[dict, dict]:
        """
        Generates input files for CP2K relaxations of all output structures.

        Args:
            input_file (:obj:`str`, optional):
                Path to CP2K input file. If not set, default input file will be
                used (see `shakenbreak/SnB_input_files/cp2k_input.inp`).
            write_structures_only (:obj:`bool`, optional):
                Whether to only write the structure files (in CIF format)
                (without calculation inputs).
                (Default: False)
            output_path (:obj:`str`, optional):
                Path to directory in which to write distorted defect structures
                and calculation inputs.
                (Default is current directory: ".")
            verbose (:obj:`bool`, optional):
                Whether to print distortion information (bond atoms and
                distances).
                (Default: False)

        Returns:
            :obj:`tuple`:
                Tuple of dictionaries with new defects_dict (containing the
                distorted structures) and defect distortion parameters.
        """
        if os.path.exists(input_file) and not write_structures_only:
            cp2k_input = Cp2kInput.from_file(input_file)
        elif (
            os.path.exists(f"{MODULE_DIR}/../SnB_input_files/cp2k_input.inp")
            and not write_structures_only
        ):
            warnings.warn(
                f"Specified input file {input_file} does not exist! Using"
                " default CP2K input file "
                "(see shakenbreak/shakenbreak/cp2k_input.inp)"
            )
            cp2k_input = Cp2kInput.from_file(
                f"{MODULE_DIR}/../SnB_input_files/cp2k_input.inp"
            )

        distorted_defects_dict, self.distortion_metadata = self.apply_distortions(
            verbose=verbose,
        )

        # loop for each defect in dict
        for defect_name, defect_dict in distorted_defects_dict.items():
            # loop for each charge state of defect
            for charge in defect_dict["charges"]:

                if not write_structures_only and cp2k_input:
                    cp2k_input.update({"FORCE_EVAL": {"DFT": {"CHARGE": charge}}})

                for dist, struct in zip(
                    [
                        "Unperturbed",
                    ]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    [defect_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    _create_folder(f"{output_path}/{defect_name}_{charge}/{dist}")
                    struct.to(
                        "cif",
                        f"{output_path}/{defect_name}_{charge}/"
                        + f"{dist}/structure.cif",
                    )
                    if not write_structures_only and cp2k_input:
                        cp2k_input.write_file(
                            input_filename="cp2k_input.inp",
                            output_dir=f"{output_path}/{defect_name}_{charge}/"
                            + f"{dist}",
                        )

        return distorted_defects_dict, self.distortion_metadata

    def write_castep_files(
        self,
        input_file: Optional[str] = f"{MODULE_DIR}/../SnB_input_files/castep.param",
        write_structures_only: Optional[bool] = False,
        output_path: str = ".",
        verbose: Optional[bool] = False,
    ) -> Tuple[dict, dict]:
        """
        Generates input `.cell` and `.param` files for CASTEP relaxations of
        all output structures.

        Args:
            input_file (:obj:`str`, optional):
                Path to CASTEP input (`.param`) file. If not set, default input
                file will be used (see `shakenbreak/SnB_input_files/castep.param`).
            write_structures_only (:obj:`bool`, optional):
                Whether to only write the structure files (in CIF format)
                (without calculation inputs).
                (Default: False)
            output_path (:obj:`str`, optional):
                Path to directory in which to write distorted defect structures
                and calculation inputs.
                (Default is current directory: ".")
            verbose (:obj:`bool`, optional):
                Whether to print distortion information (bond atoms and
                distances).
                (Default: False)

        Returns:
            :obj:`tuple`:
                Tuple of dictionaries with new defects_dict (containing the
                distorted structures) and defect distortion parameters.
        """
        distorted_defects_dict, self.distortion_metadata = self.apply_distortions(
            verbose=verbose,
        )
        aaa = AseAtomsAdaptor()
        warnings.filterwarnings(
            "ignore", ".*Could not determine the version of your CASTEP binary.*"
        )
        warnings.filterwarnings(
            "ignore", ".*Generating CASTEP keywords JSON file... hang on.*"
        )
        # loop for each defect in dict
        for defect_name, defect_dict in distorted_defects_dict.items():
            # loop for each charge state of defect
            for charge in defect_dict["charges"]:

                for dist, struct in zip(
                    [
                        "Unperturbed",
                    ]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    [defect_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    atoms = aaa.get_atoms(struct)
                    _create_folder(f"{output_path}/{defect_name}_{charge}/{dist}")

                    if write_structures_only:
                        ase.io.write(
                            filename=f"{output_path}/{defect_name}_{charge}/"
                            + f"{dist}/castep.cell",
                            images=atoms,
                            format="castep-cell",
                        )
                    else:
                        try:
                            calc = Castep(
                                directory=f"{output_path}/"
                                + f"{defect_name}_{charge}/{dist}"
                            )
                            calc.set_kpts({"size": (1, 1, 1), "gamma": True})
                            calc.merge_param(input_file)
                            calc.param.charge = charge  # Defect charge state
                            calc.set_atoms(atoms)
                            calc.initialize()  # this writes the .param file
                        except Exception:
                            warnings.warn(
                                "Problem setting up the CASTEP `.param` file. "
                                "Only structures will be written "
                                "as `castep.cell` files."
                            )
                            ase.io.write(
                                filename=f"{output_path}/{defect_name}_{charge}/"
                                + f"{dist}/castep.cell",
                                images=atoms,
                                format="castep-cell",
                            )
        return distorted_defects_dict, self.distortion_metadata

    def write_fhi_aims_files(
        self,
        input_file: Optional[str] = None,
        ase_calculator: Optional[Aims] = None,
        write_structures_only: Optional[bool] = False,
        output_path: str = ".",
        verbose: Optional[bool] = False,
    ) -> Tuple[dict, dict]:
        """
        Generates input geometry and control files for FHI-aims relaxations
        of all output structures.

        Args:
            input_file (:obj:`str`, optional):
                Path to FHI-aims input file, to overwrite/update
                `shakenbreak` default ones.
                If both `input_file` and `ase_calculator` are provided,
                the ase_calculator will be used.
            ase_calculator (:obj:`ase.calculators.aims.Aims`, optional):
                ASE calculator object to use for FHI-aims calculations.
                If not set, `shakenbreak` default values will be used.
                Recommended to check these.
                (Default: None)
            write_structures_only (:obj:`bool`, optional):
                Whether to only write the structure files (in `geometry.in`
                format), (without the contro-in file).
            output_path (:obj:`str`, optional):
                Path to directory in which to write distorted defect structures
                and calculation inputs.
                (Default is current directory: ".")
            verbose (:obj:`bool`, optional):
                Whether to print distortion information (bond atoms and
                distances).
                (Default: False)

        Returns:
            :obj:`tuple`:
                Tuple of dictionaries with new defects_dict (containing the
                distorted structures) and defect distortion parameters.
        """
        distorted_defects_dict, self.distortion_metadata = self.apply_distortions(
            verbose=verbose,
        )
        aaa = AseAtomsAdaptor()

        if input_file and not ase_calculator:
            params = io.parse_fhi_aims_input(input_file)
            ase_calculator = Aims(
                k_grid=(1, 1, 1),
                **params,
            )
            # params is in the format key: (value, value)

        if not ase_calculator and not write_structures_only:
            ase_calculator = Aims(
                k_grid=(1, 1, 1),
                relax_geometry=("bfgs", 5e-3),
                xc=("hse06", 0.11),
                hse_unit="A",  # Angstrom
                spin="collinear",  # Spin polarized
                default_initial_moment=0,  # Needs to be set
                hybrid_xc_coeff=0.25,
                # By default symmetry is not preserved
            )
        # loop for each defect in dict
        for defect_name, defect_dict in distorted_defects_dict.items():

            # loop for each charge state of defect
            for charge in defect_dict["charges"]:
                if isinstance(ase_calculator, Aims) and not write_structures_only:
                    ase_calculator.set(charge=charge)  # Defect charge state

                    # Total number of electrons for net spin initialization
                    # Must set initial spin moments (otherwise FHI-aims will
                    # lead to 0 final spin)
                    struct = defect_dict["charges"][charge]["structures"]["Unperturbed"]
                    if struct.composition.total_electrons % 2 == 0:
                        # Even number of electrons -> net spin is 0
                        ase_calculator.set(default_initial_moment=0)
                    else:
                        ase_calculator.set(default_initial_moment=1)

                for dist, struct in zip(
                    [
                        "Unperturbed",
                    ]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    [defect_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        defect_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    atoms = aaa.get_atoms(struct)
                    _create_folder(f"{output_path}/{defect_name}_{charge}/{dist}")

                    ase.io.write(
                        filename=f"{output_path}/{defect_name}_{charge}"
                        + f"/{dist}/geometry.in",
                        images=atoms,
                        format="aims",
                        info_str=dist,
                    )  # write input structure file

                    if isinstance(ase_calculator, Aims) and not write_structures_only:
                        ase_calculator.write_control(
                            filename=f"{output_path}/{defect_name}_{charge}"
                            + f"/{dist}/control.in",
                            atoms=atoms,
                        )  # write parameters file

        return distorted_defects_dict, self.distortion_metadata
