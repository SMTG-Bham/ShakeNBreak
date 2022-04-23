"""
Module containing functions for applying distortions to defect structures
"""
import sys
import os
import warnings
from typing import Optional

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from hiphive.structure_generation.rattle import generate_mc_rattled_structures


# format warnings output:
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{os.path.split(filename)[-1]}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line


def distort(
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
            neighbours. Typical choice is between 0.4 (-60%) and 1.6 (+60%).
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or interstitials), counting from 1
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for vacancies)
        distorted_element (:obj:`str`, optional):
            Neighbouring element to distort. If None, the closest neighbours to the defect will
            be chosen. (Default: None)
        verbose (:obj:`bool`, optional):
            Whether to print distortion information. (Default: False)

    Returns:
        Dictionary with distorted defect structure and the distortion parameters.
        :rtype: object
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
        for dist, index, element in distances[1:]:
            if (
                element == distorted_element
                and dist < 4.5
                and len(nearest) < num_nearest_neighbours
            ):
                nearest.append((dist, index, element))

        # if the number of nearest neighbours not reached, add other neighbouring elements
        if len(nearest) < num_nearest_neighbours:
            for i in distances[1:]:
                if (
                    len(nearest) < num_nearest_neighbours
                    and i not in nearest
                    and i[0] < 4.5
                ):
                    nearest.append(i)
            warnings.warn(
                f"{distorted_element} was specified as the nearest neighbour element to distort, "
                f"with `distortion_factor` {distortion_factor} but did not find "
                f"`num_nearest_neighbours` ({num_nearest_neighbours}) of these elements within "
                f"4.5 \u212B of the defect site. For the remaining neighbours to distort, "
                f"we ignore the elemental identity. The final distortion information is:"
            )
            sys.stderr.flush()  # ensure warning message printed before distortion info
            verbose = True
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
        "num_distorted_neighbours": num_nearest_neighbours,
        "distorted_atoms": distorted_atoms,
        "undistorted_structure": structure,
    }
    if site_index:
        bond_distorted_defect["defect_site_index"] = site_index
    elif isinstance(frac_coords, np.ndarray):
        bond_distorted_defect["defect_frac_coords"] = frac_coords

    if verbose:
        distorted = [(round(i[0], 2), i[1], i[2]) for i in distorted]
        nearest = [(round(i[0], 2), i[1], i[2]) for i in nearest]  # round numbers
        print(
            f"""\tDefect Site Index / Frac Coords: {site_index or frac_coords}
        Original Neighbour Distances: {nearest}
        Distorted Neighbour Distances:\n\t{distorted}"""
        )

    return bond_distorted_defect


def rattle(
    structure: Structure,
    stdev: float = 0.25,
    d_min: float = 2.25,
    n_iter: int = 1,
    active_atoms: Optional[list] = None,
    nbr_cutoff: float = 5,
    width: float = 0.1,
    max_attempts: int = 5000,
    max_disp: float = 2.0,
    seed: int = 42,
) -> Structure:
    """
    Given a pymatgen Structure object, apply random displacements to all atomic positions,
    with the displacement distances randomly drawn from a Gaussian distribution of standard
    deviation `stdev`.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution from which atomic
            displacement distances are drawn.
            (Default: 0.25)
        d_min (:obj:`float`):
            Minimum interatomic distance (in Angstroms). Monte Carlo rattle moves that put atoms at
            distances less than this will be heavily penalised.
            (Default: 2.25)
        n_iter (:obj:`int`):
            Number of Monte Carlo cycles to perform.
            (Default: 1)
        active_atoms (:obj:`list`, optional):
            List of which atomic indices should undergo Monte Carlo rattling.
            (Default: None)
        nbr_cutoff (:obj:`float`):
            The radial cutoff distance (in Angstroms) used to construct the list of atomic
            neighbours for checking interatomic distances.
            (Default: 5)
        width (:obj:`float`):
            Width of the Monte Carlo rattling error function, in Angstroms.
            (Default: 0.1)
        max_disp (:obj:`float`):
            Maximum atomic displacement (in Angstroms) during Monte Carlo rattling. Rarely occurs
            and is used primarily as a safety net.
            (Default: 2.0)
        max_attempts (:obj:`int`):
            Maximum Monte Carlo rattle move attempts allowed for a single atom; if this limit is
            reached an `Exception` is raised.
            (Default: 5000)
        seed (:obj:`int`):
            Seed for NumPy random state from which random displacements are generated.
            (Default: 42)

    Returns:
        Rattled pymatgen Structure object
    """
    aaa = AseAtomsAdaptor()
    ase_struct = aaa.get_atoms(structure)

    rattled_ase_struct = generate_mc_rattled_structures(
        ase_struct,
        n_configs=1,
        rattle_std=stdev,
        d_min=d_min,
        n_iter=n_iter,
        active_atoms=active_atoms,
        nbr_cutoff=nbr_cutoff,
        width=width,
        max_attempts=max_attempts,
        max_disp=max_disp,
        seed=seed,
    )[0]
    rattled_structure = aaa.get_structure(rattled_ase_struct)

    return rattled_structure


# TODO: Implement rattle function where the rattle amplitude tails off as a function of distance
#  from the defect site, as an improved version of the localised rattle
