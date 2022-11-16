"""
Module containing functions to generate rattled and bond-distorted structures,
as well as input files to run Gamma point relaxations with `VASP`, `CP2K`,
`Quantum-Espresso`, `FHI-aims` and `CASTEP`.
"""
import copy
import datetime
import functools
import itertools
import os
import shutil
import warnings
from collections import Counter
from importlib.metadata import version
from typing import Optional, Tuple, Type, Union

import ase
import numpy as np
from ase.calculators.aims import Aims
from ase.calculators.castep import Castep
from ase.calculators.espresso import Espresso
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Composition, Element, PeriodicSite, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.io.vasp.sets import BadInputSetWarning
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Voronoi
from scipy.spatial.distance import squareform

from shakenbreak import analysis, cli, distortions, io, vasp

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
            os.path.join(output_path, f"distortion_metadata_{current_datetime}.json"),
        )
        print(
            f"There is a previous version of {filename}. Will rename old metadata to "
            f"distortion_metadata_{current_datetime}.json"
        )
        try:
            print(f"Combining old and new metadata in {filename}.")
            old_metadata = loadfn(
                os.path.join(
                    output_path, f"distortion_metadata_{current_datetime}.json"
                )
            )
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
    dumpfn(obj=new_metadata, fn=filepath, indent=4)


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
    defect_name_wout_charge, charge = defect_name.rsplit(
        "_", 1
    )  # `defect_name` includes charge
    test_letters = [
        "h",
        "g",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
        "",
    ]  # reverse search to determine
    # last letter used
    try:
        matching_dirs = [
            dir
            for letter in test_letters
            for dir in os.listdir(output_path)
            if dir == f"{defect_name_wout_charge}{letter}_{charge}"
            and os.path.isdir(
                f"{output_path}/{defect_name_wout_charge}{letter}_{charge}"
            )
        ]
    except Exception:
        matching_dirs = []

    if len(matching_dirs) > 0:  # defect species with same name already present
        # check if Unperturbed structures match
        match_found = False
        for dir in matching_dirs:
            try:
                prev_unperturbed_struc = Structure.from_file(
                    f"{output_path}/{dir}/Unperturbed/POSCAR"
                )
                current_unperturbed_struc = distorted_defect_dict["Unperturbed"][
                    "Defect Structure"
                ].copy()
                for i in [prev_unperturbed_struc, current_unperturbed_struc]:
                    i.remove_oxidation_states()
                if prev_unperturbed_struc == current_unperturbed_struc:
                    warnings.warn(
                        f"The previously-generated defect folder {dir} in "
                        f"{os.path.basename(os.path.abspath(output_path))} "
                        f"has the same Unperturbed defect structure as the current "
                        f"defect species: {defect_name}. ShakeNBreak files in {dir} will "
                        f"be overwritten."
                    )
                    defect_name = dir
                    match_found = True
                    break

            except Exception:  # Unperturbed structure could not be parsed / compared to
                # distorted_defect_dict
                pass

        if not match_found:  # no matching structure found, assume inequivalent defects
            last_letter = [
                letter
                for letter in test_letters
                for dir in matching_dirs
                if dir == f"{defect_name_wout_charge}{letter}_{charge}"
            ][0]
            prev_dir_name = f"{defect_name_wout_charge}{last_letter}_{charge}"
            if last_letter == "":  # rename prev defect folder
                new_prev_dir_name = f"{defect_name_wout_charge}a_{charge}"
                new_current_dir_name = f"{defect_name_wout_charge}b_{charge}"
                warnings.warn(
                    f"A previously-generated defect folder {prev_dir_name} exists in "
                    f"{os.path.basename(os.path.abspath(output_path))}, "
                    f"and the Unperturbed defect structure could not be matched to the "
                    f"current defect species: {defect_name}. These are assumed to be "
                    f"inequivalent defects, so the previous {prev_dir_name} will be "
                    f"renamed to {new_prev_dir_name} and ShakeNBreak files for the "
                    f"current defect will be saved to {new_current_dir_name}, "
                    f"to prevent overwriting."
                )
                shutil.move(
                    f"{output_path}/{prev_dir_name}",
                    f"{output_path}/{new_prev_dir_name}",
                )
                defect_name = new_current_dir_name

            else:  # don't rename prev defect folder just rename current folder
                next_letter = test_letters[test_letters.index(last_letter) - 1]
                new_current_dir_name = (
                    f"{defect_name_wout_charge}{next_letter}_{charge}"
                )
                warnings.warn(
                    f"Previously-generated defect folders ({prev_dir_name}...) exist in "
                    f"{os.path.basename(os.path.abspath(output_path))}, "
                    f"and the Unperturbed defect structures could not be matched to the "
                    f"current defect species: {defect_name}. These are assumed to be "
                    f"inequivalent defects, so ShakeNBreak files for the current defect "
                    f"will be saved to {new_current_dir_name} to prevent overwriting."
                )
                defect_name = new_current_dir_name

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


def _get_bulk_comp(defect_object) -> Composition:
    """
    Convenience function to determine the chemical composition of the bulk
    structure for a given defect. Useful for auto-determing oxidation states.

    Args:
        defect_object (:obj:`Defect`):
           pymatgen.analysis.defects.core.Defect() object.

    Returns:
        Pymatgen Composition object for the bulk structure of the defect.
    """
    bulk_structure = defect_object.structure

    return bulk_structure.composition


def _get_bulk_defect_site(
    defect_object: Defect,
    defect_name: str,
) -> PeriodicSite:
    """Get defect site in the bulk structure (e.g.
    for a P substitution on Si, get the original Si site).
    """
    defect_type = str(defect_object.as_dict()["@class"].lower())
    if defect_type in ["antisite", "substitution"]:
        # get bulk_site
        poss_deflist = sorted(
            defect_object.structure.get_sites_in_sphere(  # Defect().structure is bulk_structure
                defect_object.site.coords, 0.01, include_index=True
            ),
            key=lambda x: x[1],
        )
        if not poss_deflist:
            raise ValueError(
                "Error in defect object generation; could not find substitution "
                f"site inside bulk structure for {defect_object.name}"
            )
        defindex = poss_deflist[0][2]
        sub_site_in_bulk = defect_object.structure[
            defindex
        ]  # bulk site of substitution

        unique_site = sub_site_in_bulk

    else:
        unique_site = defect_object.site
    return unique_site


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
    defect_object: Defect,
    defect_name: str,
    oxidation_states: dict,
    verbose: bool = False,
) -> int:
    """
    Calculates the number of extra/missing electrons of the neutral
    defect species (in `defect_object`) based on `oxidation_states`.

    Args:
        defect_object (:obj:`Defect`):
            pymatgen.analysis.defects.core.Defect object.
        defect_name (:obj:`str`):
            Name of the defect species.
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
    defect_type = defect_object.as_dict()["@class"].lower()
    # We use the following variables:
    # site_specie: Original species on the bulk site
    # substituting_specie: Site occuping the defect site
    if defect_type == "vacancy":
        site_specie = str(defect_object.site.specie.symbol)
        substituting_specie = "Vac"

    elif defect_type == "interstitial":
        substituting_specie = str(defect_object.site.specie.symbol)
        # Consider interstitials as substituting a vacant (zero oxidation-state
        # position)
        site_specie = "Vac"

    elif defect_type in ["antisite", "substitution"]:
        # get bulk_site
        sub_site_in_bulk = _get_bulk_defect_site(
            defect_object, defect_name
        )  # bulk site of substitution
        site_specie = sub_site_in_bulk.specie.symbol  # Species occuping the *bulk* site
        substituting_specie = str(
            defect_object.site.specie.symbol
        )  # Current species occupying the defect site (e.g. the substitution)

    else:
        raise ValueError(f"`defect_dict` has an invalid `defect_type`: {defect_type}")

    num_electrons = (
        oxidation_states[substituting_specie] - oxidation_states[site_specie]
    )

    if verbose:
        print(
            f"Number of extra/missing electrons of defect {defect_name}: "
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


def _get_voronoi_nodes(
    structure,
):
    """
    Get the Voronoi nodes of a structure.
    Templated from the TopographyAnalyzer class, added to
    pymatgen.analysis.defects.utils by Yiming Chen, but now deleted.
    Modified to map down to primitive, do Voronoi analysis, then map
    back to original supercell; much more efficient.

    Args:
        structure (:obj:`Structure`):
            pymatgen `Structure` object.
    """
    # map all sites to the unit cell; 0 ≤ xyz < 1.
    structure = Structure.from_sites(structure, to_unit_cell=True)
    # get Voronoi nodes in primitive structure and then map back to the
    # supercell
    prim_structure = structure.get_primitive_structure()

    # get all atom coords in a supercell of the structure because
    # Voronoi polyhedra can extend beyond the standard unit cell.
    coords = []
    cell_range = list(range(-1, 2))
    for shift in itertools.product(cell_range, cell_range, cell_range):
        for site in prim_structure.sites:
            shifted = site.frac_coords + shift
            coords.append(prim_structure.lattice.get_cartesian_coords(shifted))

    # Voronoi tessellation.
    voro = Voronoi(coords)

    # Only include voronoi polyhedra within the unit cell.
    vnodes = []
    tol = 1e-3
    for vertex in voro.vertices:
        frac_coords = prim_structure.lattice.get_fractional_coords(vertex)
        vnode = PeriodicSite("V-", frac_coords, prim_structure.lattice)
        if np.all([-tol <= coord < 1 + tol for coord in frac_coords]):
            if not any([p.distance(vnode) < tol for p in vnodes]):
                vnodes.append(vnode)

    # cluster nodes that are within a certain distance of each other
    voronoi_coords = [v.frac_coords for v in vnodes]
    dist_matrix = np.array(
        prim_structure.lattice.get_all_distances(voronoi_coords, voronoi_coords)
    )
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    cn = fcluster(z, 0.2, criterion="distance")  # cluster nodes with 0.2 Å
    merged_vnodes = []
    for n in set(cn):
        frac_coords = []
        for i, j in enumerate(np.where(cn == n)[0]):
            if i == 0:
                frac_coords.append(vnodes[j].frac_coords)
            else:
                fcoords = vnodes[j].frac_coords
                d, image = prim_structure.lattice.get_distance_and_image(
                    frac_coords[0], fcoords
                )
                frac_coords.append(fcoords + image)
        merged_vnodes.append(
            PeriodicSite("V-", np.average(frac_coords, axis=0), prim_structure.lattice)
        )
    vnodes = merged_vnodes

    # remove nodes less than 0.5 Å from sites in the structure
    vfcoords = [v.frac_coords for v in vnodes]
    sfcoords = prim_structure.frac_coords
    dist_matrix = prim_structure.lattice.get_all_distances(vfcoords, sfcoords)
    all_dist = np.min(dist_matrix, axis=1)
    vnodes = [v for i, v in enumerate(vnodes) if all_dist[i] > 0.5]

    # map back to the supercell
    sm = StructureMatcher(primitive_cell=False, attempt_supercell=True)
    mapping = sm.get_supercell_matrix(structure, prim_structure)
    voronoi_struc = Structure.from_sites(
        vnodes
    )  # Structure object with Voronoi nodes as sites
    voronoi_struc.make_supercell(mapping)  # Map back to the supercell

    # check if there was an origin shift between primitive and supercell
    regenerated_supercell = prim_structure.copy()
    regenerated_supercell.make_supercell(mapping)
    fractional_shift = sm.get_transformation(structure, regenerated_supercell)[1]
    if not np.allclose(fractional_shift, 0):
        voronoi_struc.translate_sites(
            range(len(voronoi_struc)), fractional_shift, frac_coords=True
        )

    vnodes = voronoi_struc.sites

    return vnodes


def _get_voronoi_multiplicity(site, structure):
    """Get the multiplicity of a Voronoi site in structure"""
    vnodes = _get_voronoi_nodes(structure)

    distances_and_species_list = []
    for vnode in vnodes:
        distances_and_species = [
            (np.round(vnode.distance(atomic_site), decimals=3), atomic_site.species)
            for atomic_site in structure.sites
        ]
        sorted_distances_and_species = sorted(distances_and_species)
        distances_and_species_list.append(sorted_distances_and_species)

    site_distances_and_species = [
        (np.round(site.distance(atomic_site), decimals=3), atomic_site.species)
        for atomic_site in structure.sites
    ]
    sorted_site_distances_and_species = sorted(site_distances_and_species)

    multiplicity = 0
    for distances_and_species in distances_and_species_list:
        if distances_and_species == sorted_site_distances_and_species:
            multiplicity += 1

    if multiplicity == 0:
        warnings.warn(
            f"Multiplicity of interstitial at site "
            f"{np.around(site.frac_coords, decimals=3)} could not be determined from "
            f"Voronoi analysis. Multiplicity set to 1."
        )
        multiplicity = 1

    return multiplicity


def identify_defect(
    defect_structure, bulk_structure, defect_coords=None, defect_index=None
) -> Defect:
    """
    By comparing the defect and bulk structures, identify the defect present and its site in
    the supercell, and generate a pymatgen defect object
    (pymatgen.analysis.defects.core.Defect) from this.

    Args:
        defect_structure (:obj:`Structure`):
            defect structure
        bulk_structure (:obj:`Structure`):
            bulk structure
        defect_coords (:obj:`list`):
            Fractional coordinates of the defect site in the supercell.
        defect_index (:obj:`int`):
            Index of the defect site in the supercell. For vacancies, this
            should be the site index in the bulk structure, while for substitutions
            and interstitials it should be the site index in the defect structure.

    Returns: :obj:`Defect`
    """
    natoms_defect = len(defect_structure)
    natoms_bulk = len(bulk_structure)
    if natoms_defect == natoms_bulk - 1:
        defect_type = "vacancy"
    elif natoms_defect == natoms_bulk + 1:
        defect_type = "interstitial"
    elif natoms_defect == natoms_bulk:
        defect_type = "substitution"
    else:
        raise ValueError(
            f"Could not identify defect type from number of atoms in defect ({natoms_defect}) "
            f"and bulk ({natoms_bulk}) structures. "
            "ShakeNBreak is currently only built for point defects, please contact the "
            "developers if you would like to use the method for complex defects"
        )

    # remove oxidation states before site-matching
    defect_struc = (
        defect_structure.copy()
    )  # copy to prevent overwriting original structures
    bulk_struc = bulk_structure.copy()
    defect_struc.remove_oxidation_states()
    bulk_struc.remove_oxidation_states()

    bulk_site_index = None
    defect_site_index = None

    if (
        defect_type == "vacancy" and defect_index
    ):  # defect_index should correspond to bulk struc
        bulk_site_index = defect_index
    elif defect_index:  # defect_index should correspond to defect struc
        if defect_type == "interstitial":
            defect_site_index = defect_index
        if (
            defect_type == "substitution"
        ):  # also want bulk site index for substitutions,
            # so use defect index coordinates
            defect_coords = defect_struc[defect_index].frac_coords

    if defect_coords is not None:
        if bulk_site_index is None and defect_site_index is None:
            site_displacement_tol = (
                0.01  # distance tolerance for site matching to identify defect, increases in
                # jumps of 0.1 Å
            )

            def _possible_sites_in_sphere(structure, frac_coords, tol):
                """Find possible sites in sphere of radius tol."""
                return sorted(
                    structure.get_sites_in_sphere(
                        structure.lattice.get_cartesian_coords(frac_coords),
                        tol,
                        include_index=True,
                    ),
                    key=lambda x: x[1],
                )

            max_possible_defect_sites_in_bulk_struc = _possible_sites_in_sphere(
                bulk_struc, defect_coords, 2.5
            )
            max_possible_defect_sites_in_defect_struc = _possible_sites_in_sphere(
                defect_struc, defect_coords, 2.5
            )
            expanded_possible_defect_sites_in_bulk_struc = _possible_sites_in_sphere(
                bulk_struc, defect_coords, 3.0
            )
            expanded_possible_defect_sites_in_defect_struc = _possible_sites_in_sphere(
                defect_struc, defect_coords, 3.0
            )

            # there should be one site (including specie identity) which does not match between
            # bulk and defect structures
            def _remove_matching_sites(bulk_site_list, defect_site_list):
                """Remove matching sites from bulk and defect structures."""
                bulk_sites_list = list(bulk_site_list)
                defect_sites_list = list(defect_site_list)
                for defect_site in defect_sites_list:
                    for bulk_site in bulk_sites_list:
                        if (
                            defect_site.distance(bulk_site) < 0.5
                            and defect_site.specie == bulk_site.specie
                        ):
                            if bulk_site in bulk_sites_list:
                                bulk_sites_list.remove(bulk_site)
                            if defect_site in defect_sites_list:
                                defect_sites_list.remove(defect_site)
                return bulk_sites_list, defect_sites_list

            non_matching_bulk_sites, _ = _remove_matching_sites(
                max_possible_defect_sites_in_bulk_struc,
                expanded_possible_defect_sites_in_defect_struc,
            )
            _, non_matching_defect_sites = _remove_matching_sites(
                expanded_possible_defect_sites_in_bulk_struc,
                max_possible_defect_sites_in_defect_struc,
            )

            if (
                len(non_matching_bulk_sites) == 0
                and len(non_matching_defect_sites) == 0
            ):
                warnings.warn(
                    f"Coordinates {defect_coords} were specified for (auto-determined) "
                    f"{defect_type} defect, but there are no extra/missing/different species "
                    f"within a 2.5 Å radius of this site when comparing bulk and defect "
                    f"structures. "
                    f"If you are trying to generate non-defect polaronic distortions, please use "
                    f"the distort() and rattle() functions in shakenbreak.distortions via the "
                    f"Python API. "
                    f"Reverting to auto-site matching instead."
                )

            else:
                searched = "bulk or defect"
                possible_defects = []
                while site_displacement_tol < 5:  # loop over distance tolerances
                    possible_defect_sites_in_bulk_struc = _possible_sites_in_sphere(
                        bulk_struc, defect_coords, site_displacement_tol
                    )
                    possible_defect_sites_in_defect_struc = _possible_sites_in_sphere(
                        defect_struc, defect_coords, site_displacement_tol
                    )
                    if (
                        defect_type == "vacancy"
                    ):  # defect site should be in bulk structure but not defect structure
                        possible_defects, _ = _remove_matching_sites(
                            possible_defect_sites_in_bulk_struc,
                            expanded_possible_defect_sites_in_defect_struc,
                        )
                        searched = "bulk"
                        if len(possible_defects) == 1:
                            bulk_site_index = possible_defects[0][2]
                            break

                    else:  # interstitial or substitution
                        # defect site should be in defect structure but not bulk structure
                        _, possible_defects = _remove_matching_sites(
                            expanded_possible_defect_sites_in_bulk_struc,
                            possible_defect_sites_in_defect_struc,
                        )
                        searched = "defect"
                        if len(possible_defects) == 1:
                            if defect_type == "substitution":
                                possible_defects_in_bulk, _ = _remove_matching_sites(
                                    possible_defect_sites_in_bulk_struc,
                                    expanded_possible_defect_sites_in_defect_struc,
                                )
                                if len(possible_defects_in_bulk) == 1:
                                    bulk_site_index = possible_defects_in_bulk[0][2]

                            defect_site_index = possible_defects[0][2]
                            break

                    site_displacement_tol += 0.1

                if bulk_site_index is None and defect_site_index is None:
                    warnings.warn(
                        f"Could not locate (auto-determined) {defect_type} defect site within a "
                        f"5 Å radius of specified coordinates {defect_coords} in {searched} "
                        f"structure (found {len(possible_defects)} possible defect sites). "
                        "Will attempt auto site-matching instead."
                    )

        else:  # both defect_coords and defect_index given
            warnings.warn(
                "Both defect_coords and defect_index were provided. Only one is needed, so "
                "just defect_index will be used to determine the defect site"
            )

    # try perform auto site-matching regardless of whether defect_coords/defect_index were given,
    # so we can warn user if manual specification and auto site-matching give conflicting results
    site_displacement_tol = (
        0.01  # distance tolerance for site matching to identify defect, increases in
        # jumps of 0.1 Å
    )
    auto_matching_bulk_site_index = None
    auto_matching_defect_site_index = None
    possible_defects = []
    site_matching_indices = []

    try:
        bulk_sites = [site.frac_coords for site in bulk_struc]
        defect_sites = [site.frac_coords for site in defect_struc]
        matched_bulk_indices = []
        matched_defect_indices = []
        dist_matrix = defect_struc.lattice.get_all_distances(bulk_sites, defect_sites)
        min_dist_with_index = [
            [
                min(dist_matrix[bulk_index]),
                int(bulk_index),
                int(dist_matrix[bulk_index].argmin()),
            ]
            for bulk_index in range(len(dist_matrix))
        ]  # list of [min dist, bulk ind, defect ind]

        while site_displacement_tol < 5:  # loop over distance tolerances
            for mindist, bulk_index, def_struc_index in min_dist_with_index:
                species_match = (
                    bulk_struc[bulk_index].specie
                    == defect_struc[def_struc_index].specie
                )

                matched_bulk_indices = [i[0] for i in site_matching_indices]
                matched_defect_indices = [i[1] for i in site_matching_indices]
                if (
                    mindist < site_displacement_tol
                    and species_match
                    and bulk_index not in matched_bulk_indices
                    and def_struc_index not in matched_defect_indices
                ):
                    site_matching_indices.append([bulk_index, def_struc_index])

                elif (
                    mindist < site_displacement_tol
                    and defect_type == "substitution"
                    and not species_match
                ):
                    possible_defects.append(
                        [def_struc_index, defect_sites[def_struc_index][:], bulk_index]
                    )

            if defect_type == "interstitial":
                if site_matching_indices:
                    possible_defects = [
                        [ind, fc[:], None]
                        for ind, fc in enumerate(defect_sites)
                        if ind not in matched_defect_indices
                    ]

            elif defect_type == "vacancy":
                if site_matching_indices:
                    possible_defects = [
                        [None, fc[:], ind]
                        for ind, fc in enumerate(bulk_sites)
                        if ind not in matched_bulk_indices
                    ]

            if len(possible_defects) == 1:
                auto_matching_defect_site_index = possible_defects[0][0]
                auto_matching_bulk_site_index = possible_defects[0][2]
                break

            if (
                site_matching_indices
                and auto_matching_bulk_site_index is None
                and auto_matching_defect_site_index is None
            ):
                # failed auto-site matching, rely on user input or raise error if no user input
                if len(set(np.array(site_matching_indices)[:, 0])) != len(
                    set(np.array(site_matching_indices)[:, 1])
                ):
                    raise ValueError(
                        "Error occurred in site_matching routine. Double counting of site matching "
                        f"occurred: {site_matching_indices}\nAbandoning structure parsing."
                    )

            site_displacement_tol += 0.1

    except Exception:
        pass  # failed auto-site matching, rely on user input or raise error if no user input

    if (
        defect_site_index is None
        and bulk_site_index is None
        and auto_matching_bulk_site_index is None
        and auto_matching_defect_site_index is None
    ):
        raise ValueError(
            "Defect coordinates could not be identified from auto site-matching. Check bulk and "
            "defect structures correspond to the same supercell and/or specify defect site with "
            "--defect-coords or --defect-index (if using the SnB CLI), or 'defect_coords' or "
            "'defect_index' keys in the input dictionary if using the SnB Python API."
        )

    if (
        defect_site_index is None
        and bulk_site_index is None
        and (
            auto_matching_defect_site_index is not None
            or auto_matching_bulk_site_index is not None
        )
    ):
        # user didn't specify coordinates or index, but auto site-matching found a defect site
        if auto_matching_bulk_site_index is not None:
            bulk_site_index = auto_matching_bulk_site_index
        if auto_matching_defect_site_index is not None:
            defect_site_index = auto_matching_defect_site_index

    if defect_type == "vacancy":
        defect_site = bulk_struc[bulk_site_index]
    elif defect_type == "substitution":
        defect_site_in_bulk = bulk_struc[bulk_site_index]
        defect_site = PeriodicSite(
            defect_struc[defect_site_index].specie,
            defect_site_in_bulk.frac_coords,
            bulk_struc.lattice,
        )
    else:
        defect_site = defect_struc[defect_site_index]

    if (defect_index is not None or defect_coords is not None) and (
        auto_matching_defect_site_index is not None
        or auto_matching_bulk_site_index is not None
    ):
        # user specified site, check if it matched the auto site-matching
        user_index = (
            defect_site_index if defect_site_index is not None else bulk_site_index
        )
        auto_index = (
            auto_matching_defect_site_index
            if auto_matching_defect_site_index is not None
            else auto_matching_bulk_site_index
        )
        if user_index != auto_index:
            if defect_type == "vacancy":
                auto_matching_defect_site = bulk_struc[auto_index]
            else:
                auto_matching_defect_site = defect_struc[auto_index]

            def _site_info(site):
                return (
                    f"{site.species_string} at [{site._frac_coords[0]:.3f},"
                    f" {site._frac_coords[1]:.3f}, {site._frac_coords[2]:.3f}]"
                )

            if defect_coords is not None:
                warnings.warn(
                    f"Note that specified coordinates {defect_coords} for (auto-determined)"
                    f" {defect_type} defect gave a match to defect site:"
                    f" {_site_info(defect_site)} in {searched} structure, but auto site-matching "
                    f"predicted a different defect site: {_site_info(auto_matching_defect_site)}. "
                    f"Will use user-specified site: {_site_info(defect_site)}."
                )
            else:
                warnings.warn(
                    f"Note that specified defect index {defect_index} for (auto-determined)"
                    f" {defect_type} defect gives defect site: {_site_info(defect_site)}, "
                    f"but auto site-matching predicted a different defect site:"
                    f" {_site_info(auto_matching_defect_site)}. "
                    f"Will use user-specified site: {_site_info(defect_site)}."
                )

    for_monty_defect = {
        "@module": "pymatgen.analysis.defects.core",
        "@class": defect_type.capitalize(),
        "structure": bulk_structure,
        "site": defect_site,
    }
    try:
        defect = MontyDecoder().process_decoded(for_monty_defect)
    except TypeError as exc:
        # This means we have the old version of pymatgen-analysis-defects, where the class
        # attributes were different (defect_site instead of site and no user_charges)
        v_ana_def = version("pymatgen-analysis-defects")
        v_pmg = version("pymatgen")
        if v_ana_def < "2022.9.14":
            raise TypeError(
                f"You have the version {v_ana_def} of the package `pymatgen-analysis-defects`,"
                " which is incompatible. Please update this package (with `pip install "
                "shakenbreak`) and try again."
            )
        if v_pmg < "2022.7.25":
            raise TypeError(
                f"You have the version {v_pmg} of the package `pymatgen`, which is incompatible. "
                f"Please update this package (with `pip install shakenbreak`) and try again."
            )
        else:
            raise exc

    return defect


def _get_defect_name_from_obj(defect):
    """Get the SnB defect name from defect object"""
    defect_type = str(defect.as_dict()["@class"].lower())
    if defect_type != "interstitial":
        defect_name = f"{defect.name}_s{defect.defect_site_index}"
    else:  # interstitial
        defect_name = (
            f"{defect.name}_"
            f"m{_get_voronoi_multiplicity(defect.site, defect.structure)}"
        )

    return defect_name


def _update_defect_dict(defect, defect_name, defect_dict):
    """Update `defect_dict` with {defect_name: defect}.
    If defect_name is already in defect_dict, rename it to "{defect_name}a",
    and iterate until a unique name is found for `defect`.
    """
    if defect_name in defect_dict:  # if name already exists, rename entry in
        # dict to {defect_name}a, and rename this entry to {defect_name}b
        prev_defect = defect_dict.pop(defect_name)
        defect_dict[f"{defect_name}a"] = prev_defect
        defect_name = f"{defect_name}b"
        defect_dict[defect_name] = defect

    elif defect_name in [name[:-1] for name in defect_dict.keys()]:
        # rename defect to {defect_name}{iterated letter}
        last_letters = [
            name[-1] for name in defect_dict.keys() if name[:-1] == defect_name
        ]
        last_letters.sort()
        last_letter = last_letters[-1]
        new_letter = chr(ord(last_letter) + 1)
        defect_name = f"{defect_name}{new_letter}"
        defect_dict[defect_name] = defect

    else:
        defect_dict[defect_name] = defect

    return defect_name  # return defect_name in case it was updated


# Main functions


def _apply_rattle_bond_distortions(
    defect_object: Defect,
    num_nearest_neighbours: int,
    distortion_factor: float,
    local_rattle: bool = False,
    stdev: float = 0.25,
    d_min: Optional[float] = 2.25,
    active_atoms: Optional[list] = None,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    **mc_rattle_kwargs,
) -> dict:
    """
    Applies rattle and bond distortions to the unperturbed defect structure
    (in `defect_dict`) by calling `distortion.distort` with either:
            - fractional coordinates (for vacancies) or
            - defect site index (other defect types).

    Args:
        defect_object (:obj:`Defect`):
            pymatgen.analysis.defects.core.Defect()
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to.
        distortion_factor (:obj:`float`):
            The distortion factor to apply to the bond distance between
            the defect and nearest neighbours. Typical choice is between
            0.4 (-60%) and 1.6 (+60%).
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
        **mc_rattle_kwargs:
            Additional keyword arguments to pass to `hiphive`'s
            `mc_rattle` function. These include:
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
                Seed for setting up NumPy random state from which rattle
                random displacements are generated.

    Returns:
        :obj:`dict`:
            Dictionary with distorted defect structure and the distortion
            parameters.
    """
    # Apply bond distortions to defect neighbours:
    defect_type = str(defect_object.as_dict()["@class"].lower())
    bulk_supercell_site = defect_object.site
    defect_structure = defect_object.defect_structure

    if defect_type == "vacancy":  # for vacancies, we need to use fractional coordinates
        # (no atom site in structure!)
        frac_coords = bulk_supercell_site.frac_coords
        defect_site_index = None
        bond_distorted_defect = distortions.distort(
            structure=defect_structure,
            num_nearest_neighbours=num_nearest_neighbours,
            distortion_factor=distortion_factor,
            frac_coords=frac_coords,
            distorted_element=distorted_element,
            verbose=verbose,
        )  # Dict with distorted struct, undistorted struct,
        # num_distorted_neighbours, distorted_atoms, defect_site_index/defect_frac_coords
    else:
        # .distort() assumes VASP indexing (starting at 1)
        defect_site_index = defect_object.defect_site_index + 1
        frac_coords = None  # only for vacancies
        if defect_site_index is not None:
            bond_distorted_defect = distortions.distort(
                structure=defect_structure,
                num_nearest_neighbours=num_nearest_neighbours,
                distortion_factor=distortion_factor,
                site_index=defect_site_index,
                distorted_element=distorted_element,
                verbose=verbose,
            )
        else:
            raise ValueError("Defect lacks defect_site_index!")

    # Apply rattle to the bond distorted structure
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
        rattling_atom_indices = np.arange(0, len(defect_structure))
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
                **mc_rattle_kwargs,
            )
        else:
            bond_distorted_defect["distorted_structure"] = distortions.rattle(
                structure=bond_distorted_defect["distorted_structure"],
                stdev=stdev,
                d_min=d_min,
                active_atoms=active_atoms,
                **mc_rattle_kwargs,
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
                    **mc_rattle_kwargs,
                )
            else:
                bond_distorted_defect["distorted_structure"] = distortions.rattle(
                    structure=bond_distorted_defect["distorted_structure"],
                    stdev=stdev,
                    d_min=reduced_d_min,  # min distance in supercell plus 1 stdev
                    active_atoms=active_atoms,
                    max_attempts=7000,  # default is 5000
                    **mc_rattle_kwargs,
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
    defect_object: dict,
    defect_name: str,
    num_nearest_neighbours: int,
    bond_distortions: list,
    local_rattle: bool = False,
    stdev: float = 0.25,
    d_min: Optional[float] = None,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    **mc_rattle_kwargs,
) -> dict:
    """
    Applies rattle and bond distortions to `num_nearest_neighbours` of the
    unperturbed defect structure (in `defect_dict`).

    Args:
        defect_object (:obj:`Defect`):
            pymatgen.analysis.defects.core.Defect() object.
        defect_name (:obj:`str`):
            Name of the defect species.
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
        **mc_rattle_kwargs:
            Additional keyword arguments to pass to `hiphive`'s
            `mc_rattle` function. These include:
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
                Seed from which rattle random displacements are generated. Default
                is to set seed = int(distortion_factor*100) (i.e. +40% distortion ->
                distortion_factor = 1.4 -> seed = 140, Rattled ->
                distortion_factor = 1 (no bond distortion) -> seed = 100)

    Returns:
        :obj:`dict`:
            Dictionary with distorted defect structure and the distortion
            parameters.
    """
    distorted_defect_dict = {
        "Unperturbed": defect_object,
        "distortions": {},
        "distortion_parameters": {},
    }

    defect_type = str(defect_object.as_dict()["@class"].lower())
    defect_structure = defect_object.defect_structure
    # Get defect site
    bulk_supercell_site = _get_bulk_defect_site(defect_object, defect_name)  # bulk site
    defect_site_index = defect_object.defect_site_index

    if not d_min:
        sorted_distances = np.sort(defect_structure.distance_matrix.flatten())
        d_min = (
            0.8 * sorted_distances[len(defect_structure) + 20]
        )  # ignoring interstitials by
        # ignoring the first 10 non-zero bond lengths (double counted in the distance matrix)
        if d_min < 1.0:
            warnings.warn(
                f"Automatic bond-length detection gave a bulk bond length of "
                f"{(1/0.8)*d_min} \u212B, which is almost certainly too small. "
                f"Reverting to 2.25 \u212B. If this is too large, set `d_min` manually"
            )
            d_min = 2.25

    seed = mc_rattle_kwargs.pop("seed", None)
    if num_nearest_neighbours != 0:
        for distortion in bond_distortions:
            distortion = (
                round(distortion, ndigits=3) + 0
            )  # ensure positive zero (not "-0.0%")
            if verbose:
                print(f"--Distortion {distortion:.1%}")
            distortion_factor = 1 + distortion
            if (
                not seed
            ):  # by default, set seed equal to distortion factor * 100 (e.g. 0.5 -> 50)
                # to avoid cases where a particular supercell rattle gets stuck in a local minimum
                seed = int(distortion_factor * 100)

            bond_distorted_defect = _apply_rattle_bond_distortions(
                defect_object=defect_object,
                num_nearest_neighbours=num_nearest_neighbours,
                distortion_factor=distortion_factor,
                local_rattle=local_rattle,
                stdev=stdev,
                d_min=d_min,
                distorted_element=distorted_element,
                verbose=verbose,
                seed=seed,
                **mc_rattle_kwargs,
            )
            distorted_defect_dict["distortions"][
                analysis._get_distortion_filename(distortion)
            ] = bond_distorted_defect["distorted_structure"]
            distorted_defect_dict["distortion_parameters"] = {
                "unique_site": bulk_supercell_site.frac_coords,
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
        if defect_type == "vacancy":
            defect_site_index = None
            frac_coords = bulk_supercell_site.frac_coords
        else:
            frac_coords = None  # only for vacancies!
            defect_site_index = defect_object.defect_site_index

        if (
            not seed
        ):  # by default, set seed equal to distortion factor * 100 (e.g. 0.5 -> 50)
            # to avoid cases where a particular supercell rattle gets stuck in a local minimum
            seed = 100  # distortion_factor = 1 when no bond distortion, just rattling

        if local_rattle:
            perturbed_structure = distortions.local_mc_rattle(
                defect_structure,
                site_index=defect_site_index,
                frac_coords=frac_coords,
                stdev=stdev,
                d_min=d_min,
                **mc_rattle_kwargs,
            )
        else:
            perturbed_structure = distortions.rattle(
                defect_structure,
                stdev=stdev,
                d_min=d_min,
                **mc_rattle_kwargs,
            )
        distorted_defect_dict["distortions"]["Rattled"] = perturbed_structure
        distorted_defect_dict["distortion_parameters"] = {
            "unique_site": bulk_supercell_site.frac_coords,
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
    Class to apply rattle and bond distortion to all defects in `defects`
    (each defect as a pymatgen.analysis.defects.core.Defect() object).
    """

    def __init__(
        self,
        defects: Union[list, dict],
        oxidation_states: Optional[dict] = None,
        padding: int = 1,
        dict_number_electrons_user: Optional[dict] = None,
        distortion_increment: float = 0.1,
        bond_distortions: Optional[list] = None,
        local_rattle: bool = False,
        distorted_elements: Optional[dict] = None,
        **mc_rattle_kwargs,
    ):
        """
        Args:
            defects (:obj:`dict_or_list_or_Defect`):
                List or dictionary of pymatgen.analysis.defects.core.Defect() objects.
                E.g.: [Vacancy(), Interstitial(), Substitution(), ...], or single Defect().
                In this case, generated defect folders will be named in the format:
                "{Defect.name}_m{Defect.multiplicity}" for interstitials and
                "{Defect.name}_s{Defect.defect_site_index}" for vacancies and substitutions.
                The labels "a", "b", "c"... will be appended for defects with multiple
                inequivalent sites.

                Alternatively, if specific defect folder names are desired, `defects` can
                be input as a dictionary in the format {"defect name": Defect()}.
                E.g.: {"vac_name": Vacancy(), "vac_2_name": Vacancy(), ...,
                "int_name": Interstitial(), "sub_name": Substitution(), ...}.

                Defect charge states (from which bond distortions are determined) are
                taken from the `Defect.user_charges` property. If this is not set,
                charge states are set to the range: 0 – {Defect oxidation state}
                with a `padding` (default = 1) on either side of this range.

                Alternatively, a defects dict generated by `ChargedDefectStructures`
                from `doped`/`PyCDT` can also be used as input, and the defect names
                and charge states generated by these codes will be used
                E.g.: {"bulk": {..}, "vacancies": [{...}, {...},], ...}
            oxidation_states (:obj:`dict`):
                Dictionary of oxidation states for species in your material,
                used to determine the number of defect neighbours to distort
                (e.g {"Cd": +2, "Te": -2}). If none is provided, the oxidation
                states will be guessed based on the bulk composition and most
                common oxidation states of any extrinsic species.
            padding (:obj:`int`):
                If `Defect.user_charges` is not set, charge states are set to
                the range: 0 – {Defect oxidation state}, with a `padding`
                (default = 1) on either side of this range.
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
                away from the defect site. Not recommended as typically worsens
                performance. If False (default), all supercell sites are rattled
                with the same amplitude (full rattle).
                (Default: False)
            distorted_elements (:obj:`dict`):
                Optional argument to specify the neighbouring elements to
                distort for each defect, in the form of a dictionary with
                format {'defect_name': ['element1', 'element2', ...]}
                (e.g {'vac_1_Cd': ['Te']}). If None, the closest neighbours to
                the defect are chosen.
                (Default: None)
            **mc_rattle_kwargs:
                Additional keyword arguments to pass to `hiphive`'s
                `mc_rattle` function. These include:
                - stdev (:obj:`float`):
                    Standard deviation (in Angstroms) of the Gaussian distribution
                    from which random atomic displacement distances are drawn during
                    rattling. Default is set to 10% of the nearest neighbour distance
                    in the bulk supercell.
                - d_min (:obj:`float`):
                    Minimum interatomic distance (in Angstroms) in the rattled
                    structure. Monte Carlo rattle moves that put atoms at distances
                    less than this will be heavily penalised. Default is to set this
                    to 80% of the nearest neighbour distance in the bulk supercell.
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
                    Seed from which rattle random displacements are generated. Default
                    is to set seed = int(distortion_factor*100) (i.e. +40% distortion ->
                    distortion_factor = 1.4 -> seed = 140, Rattled ->
                    distortion_factor = 1 (no bond distortion) -> seed = 100)

        """
        self.oxidation_states = oxidation_states
        self.distorted_elements = distorted_elements
        self.dict_number_electrons_user = dict_number_electrons_user
        self.local_rattle = local_rattle
        self.padding = int(padding)

        # To allow user to specify defect names (with CLI), `defects` can be either
        # a dict or list of Defects, or a single Defect
        if isinstance(defects, Defect):
            defects = [
                defects,
            ]
        # To account for this, here we refactor the list into a dict
        if isinstance(defects, list):
            self.defects_dict = {}
            if not all([isinstance(defect, Defect) for defect in defects]):
                raise TypeError(
                    "Some entries in `defects` list are not Defect objects (required "
                    "format, see docstring). Distortions can also be initialised from "
                    "pymatgen Structures using `Distortions.from_structures()`"
                )

            for defect in defects:
                defect_name = _get_defect_name_from_obj(defect)
                defect_name = _update_defect_dict(
                    defect, defect_name, self.defects_dict
                )

        elif isinstance(defects, dict):
            # check if it's a doped/PyCDT defect_dict:
            if any(
                [
                    key in defects
                    for key in [
                        "vacancies",
                        "antisites",
                        "substitutions",
                        "interstitials",
                    ]
                ]
            ):  # doped/PyCDT defect dict
                # Check bulk entry in doped/PyCDT defect_dict
                if "bulk" not in defects:  # No bulk entry - ask user to provide it
                    raise ValueError(
                        "Input `defects` dict matches `doped`/`PyCDT` format, but no 'bulk' "
                        "entry present. Please try again providing a `bulk` entry in `defects`."
                    )

                if (
                    self.padding != 1
                ):  # padding explicitly set but is ignored for doped/PyCDT dicts
                    warnings.warn(
                        f"`padding` was set to {self.padding} but this is ignored when "
                        f"generating distortions from a `doped`/`PyCDT` format dict ("
                        f"charge states specifiied in the input dict are used instead)"
                    )

                # Transform doped/PyCDT defect_dict to dictionary of {name: Defect()}
                self.defects_dict = {}
                for key, defect_dict_list in defects.items():
                    if (
                        key != "bulk"
                    ):  # loop for vacancies, antisites and  interstitials
                        for defect_dict in defect_dict_list:  # loop for each defect
                            # transform defect_dict to Defect object
                            self.defects_dict[
                                defect_dict["name"]
                            ] = cli.generate_defect_object(
                                single_defect_dict=defect_dict,
                                bulk_dict=defects["bulk"],
                            )

            else:
                if not all([isinstance(defect, Defect) for defect in defects.values()]):
                    raise TypeError(
                        "Some entries in `defects` dict are not Defect objects (required format, "
                        "see docstring). Distortions can also be initialised from pymatgen "
                        "Structures using `Distortions.from_structures()`"
                    )

                self.defects_dict = defects  # {"defect name": Defect}

        else:
            raise TypeError(
                f"`defect` must be a list or dict of defects, but got type "
                f"{type(defects)} instead. From `Distortions()` docstring:\n"
                "`defects`: List or dictionary of "
                "pymatgen.analysis.defects.core.Defect() objects.\n"
                "E.g.: [Vacancy(), Interstitial(), Substitution(), ...]\n"
                "In this case, generated defect folders will be named in the format:"
                "'{Defect.name}_s{Defect.defect_site_index}_m{Defect.multiplicity}'\n\n"
                "Alternatively, if specific defect folder names are desired, "
                "`defects` can be input as a dictionary in the format "
                "{'defect name': Defect()}\n"
                "E.g.: {'vac_name': Vacancy(), 'vac_2_name': Vacancy(), ..., "
                "'int_name': Interstitial(), 'sub_name': Substitution(), ...}.\n\n"
                "Alternatively, a defects dict generated by `ChargedDefectStructures` "
                "from `doped`/`PyCDT` can also be used as input, and the defect names "
                "generated by these codes will be used.\n"
                "E.g.: {'bulk': {..}, 'vacancies': [{...}, {...},], ...}\n\n"
                "Distortions can also be initialised from pymatgen Structures using "
                "`Distortions.from_structures()`"
            )

        # if user_charges not set for all defects, print info about how charge states will be
        # determined
        if all(not defect.user_charges for defect in self.defects_dict.values()):
            print(
                "Defect charge states will be set to the range: 0 – {Defect oxidation state}, "
                f"with a `padding = {padding}` on either side of this range."
            )

        defect_object = list(self.defects_dict.values())[0]
        bulk_comp = defect_object.structure.composition
        if "stdev" in mc_rattle_kwargs:
            self.stdev = mc_rattle_kwargs.pop("stdev")
        else:
            bulk_supercell = defect_object.structure
            sorted_distances = np.sort(bulk_supercell.distance_matrix.flatten())
            self.stdev = 0.1 * sorted_distances[len(bulk_supercell)]
            if self.stdev > 0.4 or self.stdev < 0.02:
                warnings.warn(
                    f"Automatic bond-length detection gave a bulk bond length of {10*self.stdev} "
                    f"\u212B and thus a rattle `stdev` of {self.stdev} ( = 10% bond length), "
                    f"which is unreasonable. Reverting to 0.25 \u212B. If this is too large, "
                    f"set `stdev` manually"
                )
                self.stdev = 0.25

        if len(list(self.defects_dict.values())) == 0:
            raise IndexError(
                "Problem parsing input defects; no input defects found. Please check `defects`."
            )

        # Check if all expected oxidation states are provided
        guessed_oxidation_states = bulk_comp.oxi_state_guesses()[0]

        for defect in self.defects_dict.values():
            if defect.site.specie.symbol not in guessed_oxidation_states:
                # extrinsic substituting/interstitial species not in bulk composition
                extrinsic_specie = defect.site.specie.symbol
                likely_substitution_oxi = _most_common_oxi(extrinsic_specie)
                guessed_oxidation_states[extrinsic_specie] = likely_substitution_oxi

        if self.oxidation_states is None:
            print(
                f"Oxidation states were not explicitly set, thus have been "
                f"guessed as {guessed_oxidation_states}. If this is unreasonable "
                f"you should manually set oxidation_states"
            )
            self.oxidation_states = guessed_oxidation_states

        elif guessed_oxidation_states.keys() > self.oxidation_states.keys():
            # some oxidation states are missing, so use guessed versions for these and inform user
            missing_oxidation_states = {
                k: v
                for k, v in sorted(guessed_oxidation_states.items(), key=lambda x: x[0])
                if k in (guessed_oxidation_states.keys() - self.oxidation_states.keys())
            }  # missing oxidation states in sorted dict for clean printing
            print(
                f"Oxidation states for {[k for k in missing_oxidation_states.keys()]} were not "
                f"explicitly set, thus have been guessed as {missing_oxidation_states}. "
                f"If this is unreasonable you should manually set oxidation_states"
            )
            self.oxidation_states.update(missing_oxidation_states)

        # Setup distortion parameters
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

        self._mc_rattle_kwargs = mc_rattle_kwargs

        # Create dictionary to keep track of the bond distortions applied
        rattle_parameters = self._mc_rattle_kwargs.copy()
        rattle_parameters["stdev"] = self.stdev
        self.distortion_metadata = {
            "distortion_parameters": {
                "distortion_increment": self.distortion_increment,  # None if user specified
                # bond_distortions
                "bond_distortions": self.bond_distortions,
                "local_rattle": self.local_rattle,
                "mc_rattle_parameters": rattle_parameters,
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
                    "Problem reading the keys in distorted_elements. Are they the correct defect "
                    "names (without charge states)? Proceeding without discriminating which "
                    "neighbour elements to distort.",
                )
                distorted_element = None
            else:
                # Check distorted element is a valid element symbol
                try:
                    if isinstance(distorted_elements[defect_name], list):
                        for element in distorted_elements[defect_name]:
                            Element(element)
                    elif isinstance(distorted_elements[defect_name], str):
                        Element(distorted_elements[defect_name])
                except ValueError:
                    warnings.warn(
                        "Problem reading the keys in distorted_elements. Are they correct element "
                        "symbols? Proceeding without discriminating which neighbour elements to "
                        "distort.",
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
        defect: Defect,
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
                Defect entry in dictionary of defects. Must be a
                pymatgen.analysis.defects.core.Defect() object.

        Returns:
            :obj:`int`:
                Number of extra/missing electrons for the defect.
        """
        # If the user does not specify the electron count change, we calculate it:
        if dict_number_electrons_user:
            number_electrons = dict_number_electrons_user[defect_name]
        else:
            number_electrons = _calc_number_electrons(
                defect, defect_name, oxidation_states
            )

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
            f"Then, will rattle with a std dev of {stdev:.2f} \u212B \n",
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
        rattle_parameters = self._mc_rattle_kwargs.copy()
        rattle_parameters["stdev"] = self.stdev
        distortion_metadata["defects"][defect_name]["charges"].update(
            {
                int(charge): {
                    "num_nearest_neighbours": num_nearest_neighbours,
                    "distorted_atoms": distorted_atoms,
                    "distortion_parameters": {  # store distortion parameters used for each charge
                        # state, in case posterior runs use different settings for certain defects
                        "distortion_increment": self.distortion_increment,  # None if user specified
                        # bond_distortions
                        "bond_distortions": self.bond_distortions,
                        "local_rattle": self.local_rattle,
                        "mc_rattle_parameters": rattle_parameters,
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
            + "__"
            + defect_name
        )
        return poscar_comment

    def _setup_distorted_defect_dict(
        self,
        defect: Defect,
        defect_name: str,
    ) -> dict:
        """
        Setup `distorted_defect_dict` with info for `defect`.

        Args:
            defect (:obj:`pymatgen.analysis.defects.core.Defect()`):
                Defect object to generate `distorted_defect_dict` from.
            defect_name (:obj:`str`):
                Name of the defect.

        Returns:
            :obj:`dict`
                Dictionary with information for `defect`.
        """
        # Check if user specified charge states
        if defect.user_charges:
            user_charges = defect.user_charges
        else:
            user_charges = defect.get_charge_states(padding=self.padding)
        try:
            distorted_defect_dict = {
                "defect_type": str(defect.as_dict()["@class"].lower()),
                "defect_site": defect.site,  # _get_bulk_defect_site(defect),
                "defect_supercell_site": _get_bulk_defect_site(defect, defect_name),
                "defect_multiplicity": defect.get_multiplicity(),
                "charges": {int(charge): {} for charge in user_charges},
            }  # General info about (neutral) defect
        except NotImplementedError:  # interstitial
            distorted_defect_dict = {
                "defect_type": str(defect.as_dict()["@class"].lower()),
                "defect_site": defect.site,  # _get_bulk_defect_site(defect),
                "defect_supercell_site": _get_bulk_defect_site(defect, defect_name),
                "defect_multiplicity": _get_voronoi_multiplicity(
                    defect.site, defect.structure
                ),
                "charges": {int(charge): {} for charge in user_charges},
            }  # General info about (neutral) defect
        if (
            str(defect.as_dict()["@class"].lower()) == "substitution"
        ):  # substitutions and antisites
            sub_site_in_bulk = cli._get_substituted_site(defect, defect.name)
            distorted_defect_dict[
                "substitution_specie"
            ] = sub_site_in_bulk.specie.symbol
            distorted_defect_dict["substituting_specie"] = defect.site.specie.symbol
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
        Applies rattle and bond distortion to all defects in `defects`.
        Returns a dictionary with the distorted (and undistorted) structures
        for each charge state of each defect.
        If file generation is desired, instead use the methods `write_<code>_files()`.

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

        for (
            defect_name,
            defect_object,
        ) in self.defects_dict.items():  # loop for each defect
            bulk_supercell_site = defect_object.site

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
                defect=defect_object,
            )

            self.distortion_metadata["defects"][defect_name] = {
                "unique_site": list(bulk_supercell_site.frac_coords),
                "charges": {},
            }

            distorted_defects_dict[defect_name] = self._setup_distorted_defect_dict(
                defect_object, defect_name
            )

            for charge in distorted_defects_dict[defect_name][
                "charges"
            ]:  # loop for each charge state of defect
                num_nearest_neighbours = self._get_number_distorted_neighbours(
                    defect_name=defect_name,
                    number_electrons=number_electrons,
                    charge=charge,
                )
                # Generate distorted structures
                defect_distorted_structures = apply_snb_distortions(
                    defect_object=defect_object,
                    defect_name=defect_name,
                    num_nearest_neighbours=num_nearest_neighbours,
                    bond_distortions=self.bond_distortions,
                    local_rattle=self.local_rattle,
                    stdev=self.stdev,
                    distorted_element=distorted_element,
                    verbose=verbose,
                    **self._mc_rattle_kwargs,
                )

                # Remove distortions with interatomic distances less than 1 Angstrom if Hydrogen
                # not present
                for dist, struct in list(
                    defect_distorted_structures["distortions"].items()
                ):
                    sorted_distances = np.sort(struct.distance_matrix.flatten())
                    shortest_interatomic_distance = sorted_distances[len(struct)]
                    if shortest_interatomic_distance < 1.0 and not any(
                        [el.symbol == "H" for el in struct.composition.elements]
                    ):
                        if verbose:
                            warnings.warn(
                                f"{dist} for defect {defect_name} gives an interatomic distance "
                                f"less than 1.0 Å ({shortest_interatomic_distance:.2} Å), "
                                f"which is likely to give explosive forces. Omitting this "
                                f"distortion."
                            )
                        defect_distorted_structures["distortions"].pop(dist)

                # Add distorted structures to dictionary
                distorted_defects_dict[defect_name]["charges"][charge]["structures"] = {
                    "Unperturbed": defect_distorted_structures[
                        "Unperturbed"
                    ].defect_structure,
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
        )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

        # loop for each defect in dict
        for defect_name, defect_dict in distorted_defects_dict.items():

            dict_transf = {
                k: v for k, v in defect_dict.items() if k != "charges"
            }  # Single defect dict

            # loop for each charge state of defect
            for charge in defect_dict["charges"]:
                charged_defect = {}

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
                    )  # Add charge state to transformation dict

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
                        fmt="cif",
                        filename=f"{output_path}/{defect_name}_{charge}/"
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

    @classmethod
    def from_structures(
        cls,
        defects: list,
        bulk: Structure,
        oxidation_states: Optional[dict] = None,
        padding: int = 1,
        dict_number_electrons_user: Optional[dict] = None,
        distortion_increment: float = 0.1,
        bond_distortions: Optional[list] = None,
        local_rattle: bool = False,
        distorted_elements: Optional[dict] = None,
        **mc_rattle_kwargs,
    ) -> None:
        """
        Initialise Distortions() class from defect and bulk structures.

        Args:
            defects (:obj:`list_or_Structure`):
                List of defect structures, or a single defect structure for
                which to generate distorted structures. If auto site-matching
                fails, the fractional coordinates or index of the defect site
                (in defect_structure for interstitials/substitutions, in
                bulk_structure for vacancies) can be provided in the format:
                [(defect Structure, frac_coords/index), ...] to aid site-matching.

                Defect charge states (from which bond distortions are determined) are
                set to the range: 0 – {Defect oxidation state}, with a `padding`
                (default = 1) on either side of this range.
            bulk (:obj:`pymatgen.core.structure.Structure`):
                Bulk supercell structure, matching defect supercells.
            oxidation_states (:obj:`dict`):
                Dictionary of oxidation states for species in your material,
                used to determine the number of defect neighbours to distort
                (e.g {"Cd": +2, "Te": -2}). If none is provided, the oxidation
                states will be guessed based on the bulk composition and most
                common oxidation states of any extrinsic species.
            padding (:obj:`int`):
                Defect charge states are set to the range:
                0 – {Defect oxidation state}, with a `padding` (default = 1)
                on either side of this range.
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
                away from the defect site. Not recommended as typically worsens
                performance. If False (default), all supercell sites are rattled
                with the same amplitude (full rattle).
                (Default: False)
            distorted_elements (:obj:`dict`):
                Optional argument to specify the neighbouring elements to
                distort for each defect, in the form of a dictionary with
                format {'defect_name': ['element1', 'element2', ...]}
                (e.g {'vac_1_Cd': ['Te']}). If None, the closest neighbours to
                the defect are chosen.
                (Default: None)
            **mc_rattle_kwargs:
                Additional keyword arguments to pass to `hiphive`'s
                `mc_rattle` function. These include:
                - stdev (:obj:`float`):
                    Standard deviation (in Angstroms) of the Gaussian distribution
                    from which random atomic displacement distances are drawn during
                    rattling. Default is set to 10% of the nearest neighbour distance
                    in the bulk supercell.
                - d_min (:obj:`float`):
                    Minimum interatomic distance (in Angstroms) in the rattled
                    structure. Monte Carlo rattle moves that put atoms at distances
                    less than this will be heavily penalised. Default is to set this
                    to 80% of the nearest neighbour distance in the bulk supercell.
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
                    Seed from which rattle random displacements are generated. Default
                    is to set seed = int(distortion_factor*100) (i.e. +40% distortion ->
                    distortion_factor = 1.4 -> seed = 140, Rattled ->
                    distortion_factor = 1 (no bond distortion) -> seed = 100)

        """
        # Transform structure to defect object
        pymatgen_defects_dict = {}
        if isinstance(defects, Structure):  # single defect, convert to list
            defects = [defects]

        if not isinstance(defects, list):
            raise TypeError(
                f"Wrong format for `defects`. Should be a list of pymatgen Structure objects, but "
                f"got {type(defects)} instead."
            )

        for defect_structure in defects:
            if isinstance(defect_structure, Structure):
                defect = identify_defect(
                    defect_structure=defect_structure,
                    bulk_structure=bulk,
                )
                if defect:
                    defect_name = _get_defect_name_from_obj(defect)
                    defect_name = _update_defect_dict(
                        defect, defect_name, pymatgen_defects_dict
                    )

            # Check if user gives dict with structure and defect_coords/defect_index
            elif isinstance(defect_structure, tuple) or isinstance(
                defect_structure, list
            ):
                if len(defect_structure) != 2:
                    raise ValueError(
                        "If an entry in `defects` is a tuple/list, it must be in the "
                        "format: (defect Structure, frac_coords/index)"
                    )
                elif isinstance(defect_structure[1], int) or isinstance(
                    defect_structure[1], float
                ):  # defect index
                    defect = identify_defect(
                        defect_structure=defect_structure[0],
                        bulk_structure=bulk,
                        defect_index=int(defect_structure[1]),
                    )
                elif (
                    isinstance(defect_structure[1], list)
                    or isinstance(defect_structure[1], tuple)
                    or isinstance(defect_structure[1], np.ndarray)
                ):
                    defect = identify_defect(
                        defect_structure=defect_structure[0],
                        bulk_structure=bulk,
                        defect_coords=defect_structure[1],
                    )
                else:
                    warnings.warn(
                        f"Unrecognised format for defect frac_coords/index: {defect_structure[1]} "
                        f"in `defects`. If specifying frac_coords, it should be a list or numpy "
                        f"array, or if specifying defect index, should be an integer. Got type"
                        f" {type(defect_structure[1])} instead. "
                        f"Will proceed with auto-site matching."
                    )
                    defect = identify_defect(
                        defect_structure=defect_structure[0], bulk_structure=bulk
                    )

                if defect:
                    defect_name = _get_defect_name_from_obj(defect)
                    defect_name = _update_defect_dict(
                        defect, defect_name, pymatgen_defects_dict
                    )

                else:
                    warnings.warn(
                        "Failed to identify defect from input structures. Please check bulk and "
                        "defect structures correspond to the same supercell and/or specify defect "
                        "site(s) by inputting `defects = [(defect Structure, frac_coords/index), "
                        "...]` instead."
                    )

            else:
                raise TypeError(
                    "Wrong format for `defects`. Should be a list of pymatgen Structure objects, "
                    f"but got a list of {[type(entry) for entry in defects]} instead."
                )

        # Check pymatgen_defects_dict not empty
        if len(pymatgen_defects_dict) == 0:
            raise ValueError(
                "Failed parsing defects from structures. Please check bulk and defect structures "
                "correspond to the same supercell and/or specify defect site(s) by inputting "
                "`defects = [(defect Structure, frac_coords/index), ...]` instead."
            )
        return cls(
            defects=pymatgen_defects_dict,
            oxidation_states=oxidation_states,
            padding=padding,
            dict_number_electrons_user=dict_number_electrons_user,
            distortion_increment=distortion_increment,
            bond_distortions=bond_distortions,
            local_rattle=local_rattle,
            distorted_elements=distorted_elements,
            **mc_rattle_kwargs,
        )
