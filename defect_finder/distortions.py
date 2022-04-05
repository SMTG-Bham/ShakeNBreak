"""
Module containing functions for applying distortions to defect structures
"""
from typing import Optional

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# TODO: from pymatgen.transformations.advanced_transformations import MonteCarloRattleTransformation


def bdm(
    structure: Structure,
    num_nearest_neighbours: int,
    distortion_factor: float,
    site_index: Optional[int] = None,  # starting from 1
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
    distorted_element: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> dict:
    """
    Applies bond distortions to `num_nearest_neighbours` of the defect (specified by
    `site_index` (for substitutions or interstitials) or `frac_coords`  (for vacancies))

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a pymatgen object
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to
        distortion factor (:obj:`float`):
            The distortion factor to apply to the bond distance between the defect and nearest
            neighbours. Typical choice is between 0.4 (-60%) and 1.4 (+60%).
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or interstitials), counting from 1
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for vacancies)
        distorted_element (:obj:`str`, optional):
            Neighbouring element to distort. If None, the closest neighbours to the defect will
            be chosen. (default: None)
        verbose (:obj:`bool`, optional):
            Whether to print distortion information. (default: False)

    Returns:
        Dictionary with distorted defect structure and the distortion parameters.
        """
    aaa = AseAtomsAdaptor()
    input_structure_ase = aaa.get_atoms(structure)

    if site_index:
        atom_number = site_index - 1  # Align atom number with python 0-indexing
    elif isinstance(frac_coords, np.ndarray):  # Only for vacancies!
        input_structure_ase.append("V")  # fake "V" at vacancy
        input_structure_ase.positions[-1] = np.dot(
            frac_coords, input_structure_ase.cell
        )
        atom_number = len(input_structure_ase) - 1
    else:
        raise ValueError(
            "Insufficient information to apply bond distortions, no `site_index` or "
            "`frac_coords` provided."
        )

    neighbours = (
        num_nearest_neighbours + 1
    )  # Prevent self-counting of the defect atom itself
    distances = [  # Get all distances between the selected atom and all other atoms
        (
            round(input_structure_ase.get_distance(atom_number, index, mic=True), 4),
            index + 1,
            symbol,
        )
        for index, symbol in zip(
            list(range(len(input_structure_ase))),
            input_structure_ase.get_chemical_symbols(),
        )
    ]
    distances = sorted(  # Sort the distances shortest->longest
        distances, key=lambda tup: tup[0]
    )

    if (
        distorted_element
    ):  # filter the neighbours that match the element criteria and are
        # closer than 4.5 Angstroms
        nearest = []
        for i in distances[1:]:
            if (
                len(nearest) < (neighbours - 1)
                and i[2] == distorted_element
                and i[0] < 4.5
            ):
                nearest.append(i)
            elif len(nearest) == (neighbours - 1):
                break
        # if the number of nearest neighbours not reached, add other neighbouring elements
        while len(nearest) < (neighbours - 1):
            for i in distances[1:]:
                if i not in nearest and i[0] < 4.5:
                    nearest.append(i)
    else:
        nearest = distances[
            1:neighbours
        ]  # Extract the nearest neighbours according to distance
    distorted = [
        (i[0] * distortion_factor, i[1], i[2]) for i in nearest
    ]  # Distort the nearest neighbour distances according to the distortion factor
    for i in distorted:
        input_structure_ase.set_distance(
            atom_number, i[1] - 1, i[0], fix=0, mic=True
        )  # Set the distorted distances in the ASE Atoms object

    if isinstance(frac_coords, np.ndarray):
        input_structure_ase.pop(-1)  # remove fake V from vacancy structure

    distorted_structure = aaa.get_structure(input_structure_ase)
    distorted_atoms = [(i[1], i[2]) for i in nearest]  # element and site number

    # Create dictionary with distortion info & distorted structure
    bond_distorted_defect = {
        "distorted_structure": distorted_structure,
        "number_distorted_neighbours": num_nearest_neighbours,
        "distorted_atoms": distorted_atoms,
        "undistorted_structure": structure,
    }
    if site_index:
        bond_distorted_defect["defect_site_index"] = site_index
    elif frac_coords:
        bond_distorted_defect["defect_frac_coords"] = frac_coords

    if verbose:
        distorted = [(round(i[0], 2), i[1], i[2]) for i in distorted]
        nearest = [(round(i[0], 2), i[1], i[2]) for i in nearest]  # round numbers
        print("     Defect Site Index / Frac Coords:", site_index or frac_coords)
        print("     Original Neighbour Distances:", nearest)
        print("     Distorted Neighbour Distances:\n", distorted)

    return bond_distorted_defect


def rattle(structure: Structure, stdev: Optional[float] = 0.25) -> Structure:
    """
    Given a pymatgen Structure object, apply random displacements to all atomic positions,
    with the displacement distances randomly drawn from a Gaussian distribution of standard
    deviation `stdev`.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        stdev (:obj:`float`, optional):
            Standard deviation (in Angstroms) of the Gaussian distribution from which atomic
            displacement distances are drawn.
            (default: 0.25)

    Returns:
        Rattled Structure object
    """
    aaa = AseAtomsAdaptor()
    ase_struct = aaa.get_atoms(structure)
    ase_struct.rattle(stdev=stdev)
    rattled_structure = aaa.get_structure(ase_struct)

    return rattled_structure


def localized_rattle(
    structure: Structure,
    defect_coords: np.array,
    cutoff: Optional[float] = 5.0,
    stdev: Optional[float] = 0.25,
):
    """
    Given a pymatgen Structure object, apply random displacements to all atomic positions within
    a specified radius in Angstroms (default 5) from the defect site, with the displacement
    distances randomly drawn from a Gaussian distribution of standard deviation `stdev`.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        defect_coords (:obj:`np.array`):
            Cartesian coordinates of the defect site
        cutoff (:obj:`float`, optional):
            Radius (in Angstroms) within which atomic displacements are applied.
        stdev (:obj:`float`, optional):
            Standard deviation (in Angstroms) of the Gaussian distribution from which atomic
            displacement distances are drawn.
            (default: 0.25)
    Returns:
        Rattled Structure object"""
    aaa = AseAtomsAdaptor()
    structure_copy = structure.copy()

    # Classify sites in 2 lists: inside or outside cutoff sphere
    sites_inside_cutoff, sites_outside_cutoff = [], []
    for site in structure_copy:
        distance, _image = site.distance_and_image_from_frac_coords(defect_coords)[:2]
        if distance < cutoff:
            sites_inside_cutoff.append(site)
        else:
            sites_outside_cutoff.append(site)

    # Apply rattle to sites within cutoff sphere
    structure_inside_cutoff = structure_copy.from_sites(sites_inside_cutoff)
    ase_struct = aaa.get_atoms(structure_inside_cutoff)
    ase_struct.rattle(stdev=stdev)
    rattled_structure = aaa.get_structure(ase_struct)

    # Add the sites outside the cutoff sphere to the rattled structure
    for site_outside_cutoff in sites_outside_cutoff:
        rattled_structure.append(
            site_outside_cutoff.specie, site_outside_cutoff.frac_coords
        )

    return rattled_structure


# TODO: Add MonteCarloRattleTransformation function
