"""
Module containing functions to generate rattled and bond-distorted structures, as well as input
files to run Gamma point relaxations with VASP
"""
import os
import copy
import json
import warnings
from typing import Optional
import datetime
import numpy as np
from monty.serialization import loadfn

from doped import vasp_input
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import UnknownPotcarWarning

from shakenbreak.distortions import distort, rattle
from shakenbreak.io import vasp_gam_files
from shakenbreak.analysis import _get_distortion_filename

# Load default INCAR settings for the ShakenBreak geometry relaxations
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_incar_settings = loadfn(os.path.join(MODULE_DIR, "incar.yml"))

warnings.filterwarnings(
    "ignore", category=UnknownPotcarWarning
)  # Ignore pymatgen POTCAR warnings


# format warnings output:
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """Output warning messages on one line."""
    # To set this as warnings.formatwarning, we need to be able to take in `file` and `line`,
    # but don't want to print them, so unused arguments here
    return f"{os.path.split(filename)[-1]}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line


# Helper functions
def _create_folder(folder_name: str) -> None:
    """
    Creates a folder at `./folder_name` if it doesn't already exist.
    """
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
    renamed to distortion_metadata_datetime.json and updated with new metadata.

    Args:
        new_metadata (:obj:`dict`):
            Distortion metadata containing distortion parameters used, as well as information
            about the defects and their charge states modelled.
        filename (:obj:`str`, optional):
            Filename to save metadata. Defaults to "distortion_metadata.json".
        output_path (:obj:`str`):
             Path to directory in which to write distortion_metadata.json file.
             (Default is current directory = "./")
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


def _update_struct_defect_dict(
    defect_dict: dict, structure: Structure, poscar_comment: str
) -> dict:
    """
    Given a Structure object and POSCAR comment, update the folders dictionary (generated with
    `doped.vasp_input.prepare_vasp_defect_inputs()`) with the given values.

    Args:
        defect_dict (:obj:`dict`):
            Dictionary with defect information, as generated with doped prepare_vasp_defect_inputs()
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a pymatgen object
        poscar_comment (:obj:`str`):
            Comment to include in the top line of the POSCAR file

    Returns:
        single defect dict in the `doped` format.
    """
    defect_dict_copy = copy.deepcopy(defect_dict)
    defect_dict_copy["Defect Structure"] = structure
    defect_dict_copy["POSCAR Comment"] = poscar_comment
    return defect_dict_copy


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
            Dictionary with distorted defects
        incar_settings (:obj:`dict`):
            Dictionary of user VASP INCAR settings, to overwrite/update the `doped` defaults
        potcar_settings (:obj:`dict`):
            Dictionary of user VASP POTCAR settings, to overwrite/update the `doped` defaults.
            Using `pymatgen` syntax (e.g. {'POTCAR': {'Fe': 'Fe_pv', 'O': 'O'}}).
        output_path (:obj:`str`):
             Path to directory in which to write distorted defect structures and calculation
             inputs. (Default is current directory = "./")
    """
    _create_folder(os.path.join(output_path, defect_name))  # create folder for defect
    for (
        distortion,
        single_defect_dict,
    ) in (
        distorted_defect_dict.items()
    ):  # for each distortion, create sub-subfolder folder
        potcar_settings_copy = copy.deepcopy(
            potcar_settings
        )  # vasp_gam_files empties `potcar_settings dict` (via pop()), so make a deepcopy each time
        vasp_gam_files(
            single_defect_dict=single_defect_dict,
            input_dir=f"{output_path}/{defect_name}/{distortion}",
            incar_settings=incar_settings,
            potcar_settings=potcar_settings_copy,
        )


def calc_number_electrons(
    defect_dict: dict,
    oxidation_states: dict,
    verbose: bool = False,
) -> int:
    """
    Calculates the number of extra/missing electrons of the defect species (in `defect_dict`)
    based on `oxidation_states`.

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the `doped.pycdt.core.defectsmaker.ChargedDefectsStructures()`
            format.
        oxidation_states (:obj:`dict`):
            Dictionary with oxidation states of the atoms in the material (e.g. {"Cd": +2,
            "Te": -2}).
        verbose (:obj:`bool`):
            If True, prints the number of extra/missing electrons for the defect species.

    Returns:
        Extra/missing charge (negative of the number of extra/missing electrons).
    """
    oxidation_states["Vac"] = 0  # A vacancy has an oxidation state of zero

    # Determine number of extra/missing electrons based on defect type and oxidation states
    if defect_dict["defect_type"] == "vacancy":
        site_specie = str(defect_dict["site_specie"])
        substituting_specie = "Vac"

    elif defect_dict["defect_type"] == "interstitial":
        substituting_specie = str(defect_dict["site_specie"])
        # Consider interstitials as substituting a vacant (zero oxidation-state position)
        site_specie = "Vac"

    elif defect_dict["defect_type"] == "antisite":
        site_specie = str(defect_dict["site_specie"])
        substituting_specie = defect_dict["substituting_specie"]

    elif defect_dict["defect_type"] == "substitution":
        site_specie = str(defect_dict["site_specie"])
        substituting_specie = defect_dict["substitution_specie"]

    else:
        raise ValueError(
            f"`defect_dict` has an invalid `defect_type`: {defect_dict['defect_type']}"
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


def calc_number_neighbours(num_electrons: int) -> int:
    """
    Calculate the number of neighbours to distort based off the number of extra/missing electrons.
    An octet rule approach is used; if the electron count change is less than or equal to 4,
    we distort that number of neighbours, if it is greater than 4, then we distort (8 -(electron
    change) neighbours.

    Args:
        num_electrons (:obj:`int`): Number of extra/missing electrons for the defect species

    Returns:
        Number of neighbours to distort (:obj:`int`)
    """

    if abs(num_electrons) > 4:
        num_neighbours = abs(8 - abs(num_electrons))
    else:
        num_neighbours = abs(num_electrons)
    return abs(num_neighbours)


def apply_rattle_bond_distortions(
    defect_dict: dict,
    num_nearest_neighbours: int,
    distortion_factor: float,
    stdev: float = 0.25,
    d_min: Optional[float] = None,
    active_atoms: Optional[list] = None,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """
    Applies rattle and bond distortions to the unperturbed defect structure (in `defect_dict`),
    by calling `distortion.distort` with either:
            - fractional coordinates (for vacancies) or
            - defect site index (other defect types).

        Args:
            defect_dict (:obj:`dict`):
                Defect dictionary in the format of `doped.vasp_input.prepare_vasp_defect_dict`
            num_nearest_neighbours (:obj:`int`):
                Number of defect nearest neighbours to apply bond distortions to
            distortion_factor (:obj:`float`):
                The distortion factor to apply to the bond distance between the defect and nearest
                neighbours. Typical choice is between 0.4 (-60%) and 1.4 (+60%).
            stdev (:obj:`float`):
                Standard deviation (in Angstroms) of the Gaussian distribution from which atomic
                displacement distances are drawn.
                (Default: 0.25)
            d_min (:obj:`float`):
                Minimum interatomic distance (in Angstroms) in the rattled structure. Monte Carlo
                rattle moves that put atoms at distances less than this will be heavily
                penalised. Default is to set this to 80% of the nearest neighbour distance
                in the defect supercell (ignoring interstitials).
            active_atoms (:obj:`list`, optional):
                List (or array) of which atomic indices should undergo Monte Carlo rattling.
                Default is to apply rattle to all atoms except the defect and the bond-distorted
                neighbours.
            distorted_element (:obj:`str`, optional):
                Neighbouring element to distort. If None, the closest neighbours to the defect will
                be chosen. (Default: None)
            verbose (:obj:`bool`):
                Whether to print distortion information. (Default: False)
            **kwargs:
                Additional keyword arguments to pass to `hiphive`'s `mc_rattle` function.

        Returns:
            Dictionary with distorted defect structure and the distortion parameters.
    """
    # Apply bond distortions to defect neighbours:
    if (
        defect_dict["defect_type"] == "vacancy"
    ):  # for vacancies, we need to use fractional coordinates (no atom site in structure!)
        bond_distorted_defect = distort(
            structure=defect_dict["supercell"]["structure"],
            num_nearest_neighbours=num_nearest_neighbours,
            distortion_factor=distortion_factor,
            frac_coords=defect_dict["bulk_supercell_site"].frac_coords,
            distorted_element=distorted_element,
            verbose=verbose,
        )
    else:
        defect_site_index = len(
            defect_dict["supercell"]["structure"]
        )  # defect atom comes last in structure, using doped or pymatgen
        bond_distorted_defect = distort(
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
        bond_distorted_defect["distorted_structure"] = rattle(
            structure=bond_distorted_defect["distorted_structure"],
            stdev=stdev,
            d_min=d_min,
            active_atoms=active_atoms,
            **kwargs,
        )
    except Exception as e:
        if "attempts" in str(e):
            distorted_defect_struc = bond_distorted_defect["distorted_structure"]
            sorted_distances = np.sort(distorted_defect_struc.distance_matrix.flatten())
            reduced_d_min = sorted_distances[len(distorted_defect_struc)] + (2 * stdev)
            bond_distorted_defect["distorted_structure"] = rattle(
                structure=bond_distorted_defect["distorted_structure"],
                stdev=stdev,
                d_min=reduced_d_min,  # min distance in supercell plus 2 stdevs
                active_atoms=active_atoms,
                **kwargs,
            )
            warnings.warn(
                f"Initial rattle with d_min {d_min:.2f} \u212B failed (some bond lengths "
                f"significantly smaller than this present), setting d_min to "
                f"{reduced_d_min:.2f} \u212B for this defect."
            )
        else:
            raise e

    return bond_distorted_defect


def apply_distortions(
    defect_dict: dict,
    num_nearest_neighbours: int,
    bond_distortions: list,
    stdev: float = 0.25,
    d_min: Optional[float] = None,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """
    Applies rattle and bond distortions to `num_nearest_neighbours` of the unperturbed defect
    structure (in `defect_dict`).

    Args:
        defect_dict (:obj:`dict`):
            Defect dictionary in the format of `doped.vasp_input.prepare_vasp_defect_dict`
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to
        bond_distortions (:obj:`list`):
            List of specific distortions to apply to defect nearest neighbours. (e.g. [-0.5, 0.5])
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution from which atomic
            displacement distances are drawn.
            (Default: 0.25)
        d_min (:obj:`float`, optional):
            Minimum interatomic distance (in Angstroms) in the rattled structure. Monte Carlo
            rattle moves that put atoms at distances less than this will be heavily
            penalised. Default is to set this to 80% of the nearest neighbour distance
            in the bulk supercell.
        distorted_element (:obj:`str`, optional):
            Neighbouring element to distort. If None, the closest neighbours to the defect will
            be chosen. (Default: None)
        verbose (:obj:`bool`):
            Whether to print distortion information. (Default: False)
        **kwargs:
            Additional keyword arguments to pass to `hiphive`'s `mc_rattle` function.

        Returns:
            Dictionary with distorted defect structure and the distortion parameters.
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
            bond_distorted_defect = apply_rattle_bond_distortions(
                defect_dict=defect_dict,
                num_nearest_neighbours=num_nearest_neighbours,
                distortion_factor=distortion_factor,
                stdev=stdev,
                d_min=d_min,
                distorted_element=distorted_element,
                verbose=verbose,
                **kwargs,
            )
            distorted_defect_dict["distortions"][
                _get_distortion_filename(distortion)
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
    ):  # when no extra/missing electrons, just rattle the structure. Likely to be a shallow defect.
        if defect_dict["defect_type"] == "vacancy":
            defect_site_index = None
        else:
            defect_site_index = len(
                defect_dict["supercell"]["structure"]
            )  # defect atom comes last in structure
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
                    f"{(1 / 0.8) * d_min} \u212B, which is almost certainly too small. "
                    f"Reverting to 2.25 \u212B. If this is too large, set `d_min` "
                    f"manually"
                )
                d_min = 2.25
        perturbed_structure = rattle(
            defect_dict["supercell"]["structure"], stdev=stdev, d_min=d_min, **kwargs
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


def apply_shakenbreak(
    defect_dict: dict,
    oxidation_states: dict,
    incar_settings: Optional[dict] = None,
    dict_number_electrons_user: Optional[dict] = None,
    distortion_increment: float = 0.1,
    bond_distortions: Optional[list] = None,
    stdev: float = 0.25,
    distorted_elements: Optional[dict] = None,
    potcar_settings: Optional[dict] = None,
    write_files: bool = True,
    output_path: str = ".",
    verbose: bool = False,
    **kwargs,
):
    """
    Applies rattle and bond distortion to all defects in `defect_dict` (in `doped`
    `ChargedDefectsStructures()` format), and generates the input files for `vasp_gam`
    relaxations of all output structures. Also creates a dictionary entries for each defect,
    which contain dictionaries with all bond-distorted (and undistorted) structures for each charge
    state of the defect, for reference.

    Args:
        defect_dict (:obj:`dict`):
            Dictionary of defects as generated with `doped` `ChargedDefectsStructures()`
        oxidation_states (:obj:`dict`):
            Dictionary of oxidation states for species in your material, used to determine the
            number of defect neighbours to distort (e.g {"Cd": +2, "Te": -2}).
        incar_settings (:obj:`dict`):
            Dictionary of user VASP INCAR settings (e.g. {"ENCUT": 300, ...}), to overwrite the
            `ShakenBreak` defaults for those tags.
            Highly recommended to look at output `INCAR`s, or `doped.vasp_input` source code and
            `incar.yml`, to see what the default `INCAR` settings are. (Default: None)
        dict_number_electrons_user (:obj:`dict`):
            Optional argument to set the number of extra/missing charge (negative of electron count
            change) for the input defects, as a dictionary with format {'defect_name':
            charge_change} where charge_change is the negative of the number of extra/missing
            electrons. (Default: None)
        distortion_increment (:obj:`float`):
            Bond distortion increment. Distortion factors will range from 0 to +/-0.6,
            in increments of `distortion_increment`. Recommended values: 0.1-0.3 (Default: 0.1)
        bond_distortions (:obj:`list`):
            List of bond distortions to apply to nearest neighbours, instead of the default set
            (e.g. [-0.5, 0.5]). (Default: None)
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution from which random atomic
            displacement distances are drawn during rattling. Recommended values: 0.25, or 0.15
            for strongly-bound/ionic materials. (Default: 0.25)
        distorted_elements (:obj:`dict`):
            Optional argument to specify the neighbouring elements to distort for each defect,
            in the form of a dictionary with format {'defect_name': ['element1', 'element2',
            ...]} (e.g {'vac_1_Cd': ['Te']}). If None, the closest neighbours to the defect are
            chosen. (Default: None)
        potcar_settings (:obj:`dict`):
            Dictionary of user VASP POTCAR settings, to overwrite/update the `doped` defaults.
            Using `pymatgen` syntax (e.g. {'POTCAR': {'Fe': 'Fe_pv', 'O': 'O'}}). Highly
            recommended to look at output `POTCAR`s, or `doped` `default_POTCARs.yaml`, to see what
            the default `POTCAR` settings are. (Default: None)
        write_files (:obj:`bool`):
            Whether to write output files (Default: True)
        output_path (:obj:`str`):
             Path to directory in which to write distorted defect structures and calculation
             inputs. (Default is current directory = "./")
        verbose (:obj:`bool`):
            Whether to print distortion information (bond atoms and distances). (Default: False)
        **kwargs:
            Additional keyword arguments to pass to `hiphive`'s `mc_rattle` function.

    Returns:
        tuple of dictionary with defect distortion parameters and dictionary with distorted
        structures
    """
    # TODO: Refactor to use extra/missing electrons (not charge) here, to reduce potential confusion
    vasp_defect_inputs = vasp_input.prepare_vasp_defect_inputs(
        copy.deepcopy(defect_dict)
    )
    dict_defects = {}  # dict to store bond distortions for all defects

    if bond_distortions:
        distortion_increment = None  # user specified bond_distortions, so no increment
        bond_distortions = list(
            np.around(bond_distortions, 3)
        )  # round to 3 decimal places
    else:  # If the user does not specify bond_distortions, use distortion_increment:
        bond_distortions = list(
            np.flip(np.around(np.arange(0, 0.601, distortion_increment), decimals=3))
            * -1
        )[:-1] + list(np.around(np.arange(0, 0.601, distortion_increment), decimals=3))

    # Create dictionary to keep track of the bond distortions applied
    distortion_metadata = {
        "distortion_parameters": {
            "distortion_increment": distortion_increment,  # None if user specified bond_distortions
            "bond_distortions": bond_distortions,
            "rattle_stdev": stdev,
        },
        "defects": {},
    }  # dict with distortion parameters, useful for posterior analysis

    print(
        "Applying ShakeNBreak...",
        "Will apply the following bond distortions:",
        f"{[f'{round(i,3)+0}' for i in bond_distortions]}.",
        f"Then, will rattle with a std dev of {stdev} \u212B \n",
    )

    for defect_type in [key for key in defect_dict.keys() if key != "bulk"]:
        # loop for vacancies, antisites, interstitials, substitutions
        for defect in defect_dict[defect_type]:  # loop for each defect in dict

            defect_name = defect["name"]  # name without charge state
            bulk_supercell_site = defect["bulk_supercell_site"]
            if distorted_elements:  # read the elements to distort
                try:
                    distorted_element = distorted_elements[defect_name]
                except KeyError:
                    print(
                        "Problem reading the keys in distorted_elements.",
                        "Are they the correct defect names (without charge states)?",
                        "Proceeding without discriminating which neighbour elements to distort.",
                    )
                    distorted_element = None
            else:
                distorted_element = None

            # If the user does not specify the electron count change, we calculate it:
            if dict_number_electrons_user:
                number_electrons = dict_number_electrons_user[defect_name]
            else:
                number_electrons = calc_number_electrons(defect, oxidation_states)

            dict_defects[defect_name] = {}
            distortion_metadata["defects"][defect_name] = {
                "unique_site": list(bulk_supercell_site.frac_coords),
                "charges": {},
            }
            print("\nDefect:", defect_name)
            if number_electrons < 0:
                print(f"Number of extra electrons in neutral state: {number_electrons}")
            elif number_electrons >= 0:
                print(
                    f"Number of missing electrons in neutral state: {number_electrons}"
                )

            for charge in defect["charges"]:  # loop for each charge state of defect
                charged_defect = {
                    "Unperturbed": copy.deepcopy(
                        vasp_defect_inputs[f"{defect_name}_{charge}"]
                    )
                }

                # Entry for the unperturbed defect to compare

                # Generate perturbed structures
                # Calculate extra/missing electrons accounting for the charge state of the defect
                num_electrons_charged_defect = (
                    number_electrons + charge
                )  # negative if extra e-, positive if missing e-
                num_nearest_neighbours = calc_number_neighbours(
                    num_electrons_charged_defect
                )  # Number of distorted neighbours for each charge state

                print(
                    f"\nDefect {defect_name} in charge state: {charge}. "
                    f"Number of distorted neighbours: {num_nearest_neighbours}"
                )
                distorted_structures = apply_distortions(
                    defect_dict=defect,
                    num_nearest_neighbours=num_nearest_neighbours,
                    bond_distortions=bond_distortions,
                    stdev=stdev,
                    distorted_element=distorted_element,
                    verbose=verbose,
                    **kwargs,
                )
                defect_site_index = distorted_structures["distortion_parameters"].get(
                    "defect_site_index"
                )
                if defect_site_index:
                    distortion_metadata["defects"][defect_name][
                        "defect_site_index"
                    ] = defect_site_index  # store site index of defect if not vacancy
                distortion_metadata["defects"][defect_name]["charges"].update(
                    {
                        int(charge): {
                            "num_nearest_neighbours": num_nearest_neighbours,
                            "distorted_atoms": distorted_structures[
                                "distortion_parameters"
                            ]["distorted_atoms"],
                            "distortion_parameters": {
                                "bond_distortions": bond_distortions,
                                # store distortions used for each charge state,
                                "rattle_stdev": stdev,
                                # in case posterior runs use finer mesh for only certain defects
                            },
                        }
                    }
                )  # store distortion parameters used for latter analysis

                for key_distortion, struct in distorted_structures[
                    "distortions"
                ].items():
                    poscar_comment = (
                        key_distortion.split("_")[
                            -1
                        ]  # Get distortion factor (-60.%) or 'Rattled'
                        + "__"
                        + vasp_defect_inputs[f"{defect_name}_{charge}"][
                            "POSCAR Comment"
                        ]
                        + "__num_neighbours="
                        + str(num_nearest_neighbours)
                    )
                    charged_defect[key_distortion] = _update_struct_defect_dict(
                        vasp_defect_inputs[f"{defect_name}_{charge}"],
                        struct,
                        poscar_comment,
                    )

                dict_defects[defect_name][
                    f"{defect_name}_{charge}"
                ] = charged_defect  # add charged defect entry to dict
                incar_dict = default_incar_settings.copy()
                if incar_settings is not None:
                    incar_dict.update(incar_settings)
                if write_files:
                    _create_vasp_input(
                        defect_name=f"{defect_name}_{charge}",
                        distorted_defect_dict=charged_defect,
                        incar_settings=incar_dict,
                        potcar_settings=potcar_settings,
                        output_path=output_path,
                    )
            print()
            if verbose:
                print(
                    "________________________________________________________"
                )  # output easier to read

    # save metadata
    _write_distortion_metadata(
        new_metadata=distortion_metadata,
        filename="distortion_metadata.json",
        output_path=output_path,
    )

    return (
        distortion_metadata,
        dict_defects,
    )  # TODO: Return both distorted defect structures and metadata
