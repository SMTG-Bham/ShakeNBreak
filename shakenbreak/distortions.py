"""Module containing functions for applying distortions to defect structures."""

import os
import warnings
from typing import Optional, Union

import numpy as np
from ase.neighborlist import NeighborList
from hiphive.structure_generation.rattle import _probability_mc_rattle, generate_mc_rattled_structures
from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN
from pymatgen.core.structure import Structure
from pymatgen.core.bonds import get_bond_length
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.util.typing import SpeciesLike


def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """Format warnings output."""
    return f"{os.path.split(filename)[-1]}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = _warning_on_one_line


def _get_ase_defect_structure(
    structure: Structure,
    site_index: Optional[int] = None,  # 0-indexed
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
):
    """
    Convenience function to get an ASE Atoms object of the input structure
    and the defect site index (0-indexed).

    If the defect is a vacancy (i.e. ``frac_coords`` is provided), a fake
    "V" atom is appended to the structure for consistent behaviour with
    substitutions and interstitials.

    Returns a tuple of:

    - the ASE Atoms object of the input structure, with the defect site appended as
      a fake atom for consistent behaviour with substitutions and interstitials as
      for vacancies
    - the defect site index (0-indexed)

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a pymatgen object
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), using ``python`` / ``pymatgen`` 0-indexing.
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for
            vacancies).

    Returns:
        :obj:`tuple`:
            - the ASE Atoms object of the input structure, with the defect site appended as
              a fake atom for consistent behaviour with substitutions and interstitials as
              for vacancies
            - the defect site index (0-indexed)
    """
    input_structure_ase = structure.to_ase_atoms()

    if site_index is not None:  # site_index can be 0
        return input_structure_ase, site_index

    if isinstance(frac_coords, np.ndarray):  # Only for vacancies!
        input_structure_ase.append("V")  # fake "V" at vacancy, for consistent behaviour with subs/ints
        input_structure_ase.positions[-1] = np.dot(frac_coords, input_structure_ase.cell)
        site_index = -1
    else:
        raise ValueError(
            "Insufficient information to apply bond distortions, no `site_index` or `frac_coords` "
            "provided."
        )

    return input_structure_ase, site_index


def _get_nns_to_distort(
    structure: Structure,
    num_nearest_neighbours: int,
    site_index: Optional[int] = None,  # 0-indexed
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
    distorted_element: Optional[Union[str, list]] = None,
    distorted_atoms: Optional[list] = None,
):
    """
    Convenience function to get the nearest neighbours to distort, based on the input
    parameters.

    The nearest neighbours to distort are chosen by taking all sites (or those
    matching ``distorted_element`` / ``distorted_atoms``, if provided), then sorting
    by distance to the defect site (rounded to 2 decimal places) and site index, and
    then taking the first ``num_nearest_neighbours`` of these. If there are multiple
    non-degenerate combinations of (nearly) equidistant NNs to distort (e.g. cis vs
    trans when distorting 2 NNs in a 4 NN square coordination), then the combination
    with distorted NNs closest to each other is chosen.

    Returns a tuple of:

    - a list of the nearest neighbours to distort in the form of a list of tuples
      containing the distance to the defect site, the site index (0-indexed), and the
      element symbol
    - the defect site index (0-indexed)
    - the ASE Atoms object of the input structure, with the defect site appended as
      a fake atom for consistent behaviour with substitutions and interstitials as
      for vacancies

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a pymatgen object
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), using ``python`` / ``pymatgen`` 0-indexing.
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for
            vacancies).
        distorted_element (:obj:`str`, optional):
            Neighbouring element(s) to distort, as a string or list of strings.
            If None, the closest neighbours to the defect will be chosen.
            (Default: None)
        distorted_atoms (:obj:`list`, optional):
            List of atom indices to distort, using 0-indexing (i.e. python /
            pymatgen indexing). If None, the closest neighbours to the defect
            will be chosen. (Default: None)

    Returns:
        :obj:`tuple`:
            - a list of the nearest neighbours to distort in the form of a list of tuples
                containing the distance to the defect site, the site index (0-indexed), and the
                element symbol
            - the defect site index (0-indexed)
            - the ASE Atoms object of the input structure, with the defect site appended as
              a fake atom for consistent behaviour with substitutions and interstitials as
              for vacancies
    """
    input_structure_ase, defect_site_index = _get_ase_defect_structure(structure, site_index, frac_coords)

    if distorted_atoms and len(distorted_atoms) < num_nearest_neighbours:
        warnings.warn(
            f"Only {len(distorted_atoms)} atoms were specified to distort in `distorted_atoms`, "
            f"but `num_nearest_neighbours` was set to {num_nearest_neighbours}. "
            f"Will overide the indices specified in `distorted_atoms` and distort the "
            f"{num_nearest_neighbours} closest neighbours to the defect site."
        )
        distorted_atoms = None

    if distorted_atoms is None:
        distorted_atoms = list(range(len(input_structure_ase)))

    if distorted_element:
        if isinstance(distorted_element, str):
            distorted_element = [distorted_element]
        distorted_atoms = [
            i
            for i in distorted_atoms
            if input_structure_ase.get_chemical_symbols()[i] in distorted_element
        ]
        if not distorted_atoms:
            raise ValueError(
                f"No atoms of `distorted_element` = {distorted_element} found in the defect structure, "
                f"cannot apply bond distortions."
            )

    nearest_neighbours = sorted(
        [
            (
                input_structure_ase.get_distance(defect_site_index, index, mic=True),
                index,  # 0-indexing (ASE/pymatgen)
                input_structure_ase.get_chemical_symbols()[index],
            )
            for index in distorted_atoms
            if index not in [defect_site_index, len(input_structure_ase) + defect_site_index]  # in case -1
            # ignore defect itself
        ],
        key=lambda tup: (round(tup[0], 2), tup[1]),  # sort by distance (to 2 dp), then by index
    )
    furthest_nn_to_distort = nearest_neighbours[num_nearest_neighbours - 1]

    # check if there are non-degenerate combinations of (nearly) equidistant NNs to distort:
    # non-degenerate combinations only possible when more than one NN to distort, and more than one
    # equidistant NN _not_ being distorted (e.g. distorting 3 of 4 equidistant NNs does not have
    # non-degenerate combinations):
    if (
        num_nearest_neighbours > 1
        and len([nn_tup for nn_tup in nearest_neighbours if nn_tup[0] <= furthest_nn_to_distort[0] * 1.05])
        < num_nearest_neighbours - 1
    ):
        nearest_neighbour = nearest_neighbours[0]
        # now re-sort by NN distance (to 2 dp), and then distance to first distorted NN (to 2 dp)
        # this is to ensure a deterministic choice in NNs to distort, in cases of degenerate choices of NNs
        # in terms of distance, but non-degenerate in terms of combination of NNs to distort
        # e.g. square coordination, indexed clockwise 1-4, then distorting 1 & 2 is different to 1 & 3 (
        # i.e. cis vs trans essentially); ShakeNBreak default is to favour NNs which are closer to each
        # other (i.e. favouring cis distortions, and thus dimer/trimer formation or other cluster-type
        # rebonding)
        nearest_neighbours = sorted(
            nearest_neighbours,
            key=lambda tup: (
                round(tup[0], 2),
                round(input_structure_ase.get_distance(nearest_neighbour[1], tup[1], mic=True), 2),
                tup[1],
            ),
        )

    nns_to_distort = nearest_neighbours[:num_nearest_neighbours]  # defect site itself already cut

    return nns_to_distort, defect_site_index, input_structure_ase


def distort(
    structure: Structure,
    num_nearest_neighbours: int,
    distortion_factor: float,
    site_index: Optional[int] = None,  # 0-indexed
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
    distorted_element: Optional[Union[str, list]] = None,
    distorted_atoms: Optional[list] = None,
    verbose: Optional[bool] = False,
) -> dict:
    """
    Applies bond distortions to ``num_nearest_neighbours`` of the defect (specified
    by ``site_index`` (for substitutions or interstitials, counting from 1) or
    ``frac_coords`` (for vacancies)).

    The nearest neighbours to distort are chosen by taking all sites (or those
    matching ``distorted_element`` / ``distorted_atoms``, if provided), then sorting
    by distance to the defect site (rounded to 2 decimal places) and site index, and
    then taking the first ``num_nearest_neighbours`` of these. If there are multiple
    non-degenerate combinations of (nearly) equidistant NNs to distort (e.g. cis vs
    trans when distorting 2 NNs in a 4 NN square coordination), then the combination
    with distorted NNs closest to each other is chosen.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a ``pymatgen`` ``Structure`` object.
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to
        distortion_factor (:obj:`float`):
            The distortion factor to apply to the bond distance between the
            defect and nearest neighbours. Typical choice is between 0.4 (-60%)
            and 1.6 (+60%).
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), using ``python`` / ``pymatgen`` 0-indexing.
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for
            vacancies).
        distorted_element (:obj:`str`, optional):
            Neighbouring element(s) to distort, as a string or list of strings.
            If None, the closest neighbours to the defect will be chosen.
            (Default: None)
        distorted_atoms (:obj:`list`, optional):
            List of atom indices to distort, using 0-indexing (i.e. python /
            pymatgen indexing). If None, the closest neighbours to the defect
            will be chosen. (Default: None)
        verbose (:obj:`bool`, optional):
            Whether to print distortion information. (Default: False)

    Returns:
        :obj:`dict`:
            Dictionary with distorted defect structure and the distortion parameters.
    """
    if frac_coords is not None and not isinstance(frac_coords, np.ndarray):
        frac_coords = np.array(frac_coords)

    nns_to_distort, defect_site_index, input_structure_ase = _get_nns_to_distort(
        structure,
        num_nearest_neighbours,
        site_index,  # 0-indexed
        frac_coords,
        distorted_element,
        distorted_atoms,  # 0-indexed
    )

    distorted = [  # Note: i[1] in nns_to_distort is 0-indexed, matching ASE/pymatgen etc
        (i[0] * distortion_factor, i[1], i[2]) for i in nns_to_distort
    ]  # Distort the nearest neighbour distances according to the distortion factor
    for i in distorted:
        input_structure_ase.set_distance(  # fix=0 keeps ``defect_site_index`` fixed
            defect_site_index, i[1], i[0], fix=0, mic=True
        )  # Set the distorted distances in the ASE Atoms object

    if isinstance(frac_coords, np.ndarray):
        input_structure_ase.pop(-1)  # remove fake V from vacancy structure

    distorted_structure = Structure.from_ase_atoms(input_structure_ase)
    distorted_atoms = [[i[1], i[2]] for i in nns_to_distort]  # site number and element; 0-indexed

    # Create dictionary with distortion info & distorted structure
    bond_distorted_defect = {
        "distorted_structure": distorted_structure,
        "num_distorted_neighbours": num_nearest_neighbours,
        "distorted_atoms": distorted_atoms,
        "undistorted_structure": structure,
    }
    if site_index is not None:  # site_index can be 0
        bond_distorted_defect["defect_site_index"] = site_index
    elif isinstance(frac_coords, np.ndarray):
        bond_distorted_defect["defect_frac_coords"] = frac_coords

    if verbose:
        distorted_info = [(round(i[0], 2), i[1], i[2]) for i in distorted]
        nearest_info = [(round(i[0], 2), i[1], i[2]) for i in nns_to_distort]  # round numbers
        site_index_or_frac_coords = (
            site_index if site_index is not None else np.around(frac_coords, decimals=3)
        )
        print(
            f"""\tDefect Site Index / Frac Coords: {site_index_or_frac_coords}
            Original Neighbour Distances: {nearest_info}
            Distorted Neighbour Distances:\n\t{distorted_info}"""
        )

    return bond_distorted_defect


def get_dimer_bond_length(
    species_1: SpeciesLike,
    species_2: SpeciesLike,
):
    """
    Get the estimated dimer bond length between two species,
    using the ``get_bond_length()`` function from ``pymatgen``
    (which uses a database of known covalent bond lengths), or
    if this fails, using the sum of the covalent radii of the
    two species.

    Args:
        species_1 (SpeciesLike):
            First species.
        species_2 (SpeciesLike):
            Second species.

    Returns:
        float:
            The estimated dimer bond length between the two species.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pmg_bond_length = get_bond_length(species_1, species_2)
        if w and any("No order" in str(warn.message) for warn in w):
            # use CovalentRadius values, rather than pmg defaulting to atomic radii
            return CovalentRadius.radius[str(species_1)] + CovalentRadius.radius[str(species_2)]

    return pmg_bond_length


def apply_dimer_distortion(
    structure: Structure,
    site_index: Optional[int] = None,  # 0-indexed
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
    dimer_bond_length: Optional[float] = None,
    verbose: Optional[bool] = False,
) -> dict:
    """
    Apply a dimer distortion to a defect structure.

    The defect nearest neighbours are determined (using ``CrystalNN`` with
    default settings), from them the two closest in distance are selected,
    which are pushed towards each other so that their inter-atomic distance
    is ``dimer_bond_length`` Å.

    If ``dimer_bond_length`` is ``None`` (default), then the dimer bond length
    is estimated using ``get_dimer_bond_length``, which uses a database of
    known covalent bond lengths, or if this fails, using the sum of the
    covalent radii of the two species.

    Args:
        structure (Structure):
            Defect structure.
        site_index (Optional[int]):
            Index of defect site (for non vacancy defects), using ``python``
            / ``pymatgen`` 0-indexing.
            Defaults to None.
        frac_coords (Optional[np.array]):
            Fractional coordinates of the defect site in the structure (for
            vacancies).
            Defaults to None.
        dimer_bond_length (float):
            The bond length to set the dimer to, in Å.
            If ``None`` (default), uses ``get_dimer_bond_length`` to estimate
            the dimer bond length.
        verbose (Optional[bool]):
            Print information about the dimer distortion.
            Defaults to False.

    Returns:
        obj:`Structure`:
            Distorted dimer structure
    """
    if frac_coords is not None and not isinstance(frac_coords, np.ndarray):
        frac_coords = np.array(frac_coords)

    # Note: Future work could extend this to allow the use of ``distorted_element`` and ``distorted_atoms``
    input_structure_ase, defect_site_index = _get_ase_defect_structure(structure, site_index, frac_coords)

    # Get defect nn
    input_structure = Structure.from_ase_atoms(input_structure_ase)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No oxidation states")
        warnings.filterwarnings("ignore", message="CrystalNN")
        cnn = CrystalNN()
        sites = [d["site"] for d in cnn.get_nn_info(input_structure, defect_site_index)]

    if len(sites) < 2:
        nns_to_distort, defect_site_index, input_structure_ase = _get_nns_to_distort(
            structure,
            2,
            site_index,  # 0-indexed
            frac_coords,
        )
        sites = [input_structure.sites[idx] for idx in [i[1] for i in nns_to_distort]]

    # Get distances between NN
    distances = {}
    sites = sorted(sites, key=lambda x: x.index)  # sort by site index for deterministic behaviour
    for i, site in enumerate(sites):
        for other_site in sites[i + 1 :]:
            distances[(site.index, other_site.index)] = site.distance(other_site)

    # Get defect NN with the smallest distance and lowest indices:
    site_indexes = tuple(
        sorted(min(distances, key=lambda k: (round(distances.get(k, 10), 2), k[0], k[1])))
    )
    if dimer_bond_length is None:
        dimer_bond_length = get_dimer_bond_length(
            input_structure_ase[site_indexes[0]].symbol, input_structure_ase[site_indexes[1]].symbol
        )

    # Set their distance to dimer_bond_length Å
    input_structure_ase.set_distance(
        a0=site_indexes[0], a1=site_indexes[1], distance=dimer_bond_length, fix=0.5, mic=True
    )  # fix=0.5 keeps the centre of line between these atoms fixed in position

    if type(frac_coords) in [list, tuple, np.ndarray]:
        input_structure_ase.pop(-1)  # remove fake V from vacancy structure

    distorted_structure = Structure.from_ase_atoms(input_structure_ase)
    # Create dictionary with distortion info & distorted structure
    # Get distorted atoms
    distorted_structure_wout_oxi = distorted_structure.copy()
    distorted_structure_wout_oxi.remove_oxidation_states()
    distorted_atoms = [
        [site_indexes[0], distorted_structure_wout_oxi[site_indexes[0]].species_string],
        [site_indexes[1], distorted_structure_wout_oxi[site_indexes[1]].species_string],
    ]
    bond_distorted_defect = {
        "distorted_structure": distorted_structure,
        "num_distorted_neighbours": 2,
        "distorted_atoms": distorted_atoms,
        "undistorted_structure": structure,
    }
    if site_index is not None:  # site_index can be 0
        bond_distorted_defect["defect_site_index"] = site_index
    elif type(frac_coords) in [np.ndarray, list]:
        bond_distorted_defect["defect_frac_coords"] = frac_coords
    if verbose:
        original_distance = round(structure.get_distance(site_indexes[0], site_indexes[1]), 2)
        print(
            f"""\tDefect Site Index / Frac Coords: {site_index or np.around(frac_coords, decimals=3)}
            Dimer Distorted Neighbours: {distorted_atoms}
            Original Distance: {original_distance}
            Distorted Neighbour Distances: {dimer_bond_length:.2f} Å"""
        )
    return bond_distorted_defect


# TODO: allow setting dimer bond length from higher level functions?


def rattle(
    structure: Structure,
    stdev: Optional[float] = None,
    d_min: Optional[float] = None,
    verbose: bool = False,
    n_iter: int = 1,
    active_atoms: Optional[list] = None,
    nbr_cutoff: float = 5,
    width: float = 0.1,
    max_attempts: int = 5000,
    max_disp: float = 2.0,
    seed: int = 42,
) -> Structure:
    """
    Given a ``pymatgen`` ``Structure`` object, apply random displacements to all
    atomic positions, with the displacement distances randomly drawn from a
    Gaussian distribution of standard deviation ``stdev``, using a Monte Carlo
    algorithm which disfavours moves that bring atoms closer than ``d_min``.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution
            from which random atomic displacement distances are drawn during
            rattling. Default is set to 10% of the bulk nearest neighbour
            distance.
        d_min (:obj:`float`):
            Minimum interatomic distance (in Angstroms) in the rattled
            structure. Monte Carlo rattle moves that put atoms at
            distances less than this will be heavily penalised.
            Default is to set this to 80% of the nearest neighbour
            distance in the defect supercell.
        verbose (:obj:`bool`):
            Whether to print information about the rattling process, if
            rattling initially fails with initial ``d_min``.
        n_iter (:obj:`int`):
            Number of Monte Carlo cycles to perform.
            (Default: 1)
        active_atoms (:obj:`list`, optional):
            List of which atomic indices should undergo Monte Carlo rattling.
            If not set, rattles all atoms in the structure.
            (Default: None)
        nbr_cutoff (:obj:`float`):
            The radial cutoff distance (in Angstroms) used to construct the
            list of atomic neighbours for checking interatomic distances.
            (Default: 5)
        width (:obj:`float`):
            Width of the Monte Carlo rattling error function, in Angstroms.
            (Default: 0.1)
        max_disp (:obj:`float`):
            Maximum atomic displacement (in Angstroms) during Monte Carlo
            rattling. Rarely occurs and is used primarily as a safety net.
            (Default: 2.0)
        max_attempts (:obj:`int`):
            Maximum Monte Carlo rattle move attempts allowed for a single atom;
            if this limit is reached an ``Exception`` is raised.
            (Default: 5000)
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)

    Returns:
        :obj:`Structure`:
            Rattled ``pymatgen`` ``Structure`` object
    """
    ase_struct = structure.to_ase_atoms()
    if active_atoms is not None:
        # select only the distances involving active_atoms
        distance_matrix = structure.distance_matrix[active_atoms, :][:, active_atoms]
    else:
        distance_matrix = structure.distance_matrix

    sorted_distances = np.sort(distance_matrix[distance_matrix > 0.8].flatten())

    if stdev is None:
        stdev = 0.1 * sorted_distances[0]
        if stdev > 0.4 or stdev < 0.02:
            warnings.warn(
                f"Automatic bond-length detection gave a bulk bond length of {10 * stdev} "
                f"\u212B and thus a rattle `stdev` of {stdev} ( = 10% bond length), "
                f"which is unreasonable. Reverting to 0.25 \u212B. If this is too large, "
                f"set `stdev` manually"
            )
            stdev = 0.25

    if d_min is None:
        d_min = 0.8 * sorted_distances[0]
        if d_min < 1.0:
            warnings.warn(
                f"Automatic bond-length detection gave a bulk bond length of "
                f"{(1/0.8)*d_min} \u212B, which is almost certainly too small. "
                f"Reverting to 2.25 \u212B. If this is too large, set `d_min` manually"
            )
            d_min = 2.25

    try:
        rattled_ase_struct = generate_mc_rattled_structures(
            ase_struct,
            1,  # n_configs in hiphive <= 1.1, n_structures in hiphive >= 1.2
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

    except Exception as ex:
        if "attempts" not in str(ex):
            raise ex

        for i in range(1, 10):  # reduce d_min in 10% increments
            reduced_d_min = d_min * float(1 - (i / 10))
            try:
                rattled_ase_struct = generate_mc_rattled_structures(
                    ase_struct,
                    1,  # n_configs in hiphive <= 1.1, n_structures in hiphive >= 1.2
                    rattle_std=stdev,
                    d_min=reduced_d_min,
                    n_iter=n_iter,
                    active_atoms=active_atoms,
                    nbr_cutoff=nbr_cutoff,
                    width=width,
                    max_attempts=max(max_attempts, 7000),  # default is 5000
                    max_disp=max_disp,
                    seed=seed,
                )[0]
                break
            except Exception as ex:
                if "attempts" in str(ex):
                    continue

                raise ex

        if verbose:
            warnings.warn(
                f"Initial rattle with d_min {d_min:.2f} \u212B failed (some bond lengths significantly "
                f"smaller than this present), setting d_min to {reduced_d_min:.2f} \u212B for this defect."
            )

    return Structure.from_ase_atoms(rattled_ase_struct)


def distort_and_rattle(
    structure: Structure,
    distortion_factor: Union[float, str],
    num_nearest_neighbours: int = 0,
    site_index: Optional[int] = None,  # 0-indexed
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
    local_rattle: bool = False,
    stdev: Optional[float] = None,
    d_min: Optional[float] = None,
    active_atoms: Optional[list] = None,
    distorted_element: Optional[str] = None,
    distorted_atoms: Optional[list] = None,
    verbose: bool = False,
    **mc_rattle_kwargs,
) -> dict:
    """
    Applies bond distortions and rattling to ``num_nearest_neighbours`` of the
    defect (specified by ``site_index`` (for substitutions or interstitials, counting
    from 1) or ``frac_coords`` (for vacancies)).

    Note that by default, rattling is not applied to the defect site or distorted
    neighbours, but to all other atoms in the supercell, however this can be
    controlled with the ``active_atoms`` kwarg. Rattling is performed by applying
    random displacements to atomic positions, with the displacement distances
    randomly drawn from a Gaussian distribution of standard deviation ``stdev``,
    using a Monte Carlo algorithm which disfavours moves that bring atoms closer
    than ``d_min``. Rattling behaviour can be controlled with ``mc_rattle_kwargs``,
    see the ``rattle()`` docstring for more info.
    Note that dimer distortions can be generated by setting ``distortion_factor``
    to "Dimer"; see ``apply_dimer_distortion()`` for more info.

    The nearest neighbours to distort are chosen by taking all sites (or those
    matching ``distorted_element`` / ``distorted_atoms``, if provided), then sorting
    by distance to the defect site (rounded to 2 decimal places) and site index, and
    then taking the first ``num_nearest_neighbours`` of these. If there are multiple
    non-degenerate combinations of (nearly) equidistant NNs to distort (e.g. cis vs
    trans when distorting 2 NNs in a 4 NN square coordination), then the combination
    with distorted NNs closest to each other is chosen.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a ``pymatgen`` ``Structure`` object.
        distortion_factor (:obj:`float`):
            The distortion factor or distortion name ("Dimer") to apply
            to the bond distance between the defect and nearest neighbours.
            Typical choice is between 0.4 (-60%) and 1.6 (+60%).
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to.
            Default is 0.
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), using ``python`` / ``pymatgen`` 0-indexing.
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for
            vacancies).
        local_rattle (:obj:`bool`):
            Whether to apply random displacements that tail off as we move
            away from the defect site. If False, all supercell sites are
            rattled with the same amplitude.
            (Default: False)
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution
            from which random atomic displacement distances are drawn during
            rattling. Default is set to 10% of the bulk nearest neighbour
            distance.
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
        distorted_atoms (:obj:`list`, optional):
            List of atom indices which can undergo bond distortions, using
            0-indexing (i.e. python / pymatgen indexing). If None, the closest
            neighbours to the defect will be chosen. (Default: None)
        verbose (:obj:`bool`):
            Whether to print distortion information.
            (Default: False)
        **mc_rattle_kwargs (:obj:`dict`):
            Additional keyword arguments to pass to ``hiphive``'s
            ``mc_rattle`` function. These include:

            - max_disp (:obj:`float`):
                Maximum atomic displacement (in Å) during Monte Carlo
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
    if frac_coords is not None and not isinstance(frac_coords, np.ndarray):
        frac_coords = np.array(frac_coords)

    if isinstance(distortion_factor, str) and distortion_factor.lower() == "dimer":
        bond_distorted_defect = apply_dimer_distortion(
            structure=structure,
            site_index=site_index,
            frac_coords=frac_coords,
            verbose=verbose,
        )
    else:
        bond_distorted_defect = distort(
            structure=structure,
            num_nearest_neighbours=num_nearest_neighbours,
            distortion_factor=distortion_factor,
            site_index=site_index,
            frac_coords=frac_coords,
            distorted_element=distorted_element,
            distorted_atoms=distorted_atoms,  # 0-indexed
            verbose=verbose,
        )  # Dict with distorted struct, undistorted struct,
        # num_distorted_neighbours, distorted_atoms, defect_site_index/defect_frac_coords

    # Apply rattle to the bond distorted structure
    if active_atoms is None:
        distorted_atom_indices = [i[0] for i in bond_distorted_defect["distorted_atoms"]]
        if "defect_site_index" in bond_distorted_defect:  # only present if not vacancy
            distorted_atom_indices += [bond_distorted_defect["defect_site_index"]]
        rattling_atom_indices = np.arange(0, len(structure))
        idx = np.in1d(rattling_atom_indices, distorted_atom_indices)  # returns True for matching indices
        active_atoms = rattling_atom_indices[~idx]  # remove matching indices

    if local_rattle:
        bond_distorted_defect["distorted_structure"] = local_mc_rattle(
            structure=bond_distorted_defect["distorted_structure"],
            frac_coords=frac_coords,
            site_index=site_index,
            stdev=stdev,
            d_min=d_min,
            verbose=verbose,
            active_atoms=active_atoms,
            **mc_rattle_kwargs,
        )
    else:
        bond_distorted_defect["distorted_structure"] = rattle(
            structure=bond_distorted_defect["distorted_structure"],
            stdev=stdev,
            d_min=d_min,
            verbose=verbose,
            active_atoms=active_atoms,
            **mc_rattle_kwargs,
        )

    return bond_distorted_defect


def _local_mc_rattle_displacements(
    atoms,
    site_index,
    rattle_std,
    d_min,
    width=0.1,
    n_iter=10,
    max_attempts=5000,
    max_disp=2.0,
    active_atoms=None,
    nbr_cutoff=None,
    seed=42,
) -> np.ndarray:
    # This function has been adapted from https://gitlab.com/materials-modeling/hiphive
    """
    Generate displacements using the Monte Carlo rattle method.
    The displacements tail off as we move away from the defect site.

    Args:
        atoms (:obj:`ase.Atoms`):
            prototype structure
        site_index (:obj:`int`):
            index of defect, starting from 0
        rattle_std (:obj:`float`):
            rattle amplitude (standard deviation in normal distribution)
        d_min (:obj:`float`):
            interatomic distance used for computing the probability for each rattle
            move. Center position of the error function
        width (:obj:`float`):
            width of the error function
        n_iter (:obj:`int`):
            number of Monte Carlo cycle
        max_disp (:obj:`float`):
            rattle moves that yields a displacement larger than max_disp will
            always be rejected. This rarley occurs and is more used as a safety net
            for not generating structures where two or more have swapped positions.
        max_attempts (:obj:`int`):
            limit for how many attempted rattle moves are allowed a single atom;
            if this limit is reached an ``Exception`` is raised.
        active_atoms (:obj:`list`):
            list of which atomic indices should undergo Monte Carlo rattling
        nbr_cutoff (:obj:`float`):
            The cutoff used to construct the neighborlist used for checking
            interatomic distances, defaults to 2 * d_min
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)

    Returns:
        :obj:`numpy.ndarray`:
            atomic displacements (Nx3)
    """

    def scale_stdev(disp, r_min, r):
        """
        Linearly scale the rattle standard deviation used for a given site
        according to its distance to the defect (e.g. sites further away
        from defect will experience a smaller distortion).
        """
        if r == 0:  # avoid dividing by 0 for defect site
            r = r_min
        return disp * r_min / r

    structure = Structure.from_ase_atoms(atoms)  # transform to ``pymatgen`` ``Structure``
    dist_defect_to_nn = max(
        structure[site_index].distance(_["site"])
        for _ in MinimumDistanceNN().get_nn_info(structure, site_index)
    )  # distance between defect and nearest neighbours

    # setup
    rs = np.random.RandomState(seed)

    if nbr_cutoff is None:
        nbr_cutoff = 2 * d_min

    if active_atoms is None:
        active_atoms = range(len(atoms))

    atoms_rattle = atoms.copy()
    reference_positions = atoms_rattle.get_positions()
    nbr_list = NeighborList(
        [nbr_cutoff / 2] * len(atoms_rattle),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    nbr_list.update(atoms_rattle)

    # run Monte Carlo
    for _ in range(n_iter):
        for i in active_atoms:
            i_nbrs = np.setdiff1d(nbr_list.get_neighbors(i)[0], [i])

            # Distance between defect and site i
            dist_defect_to_i = atoms.get_distance(site_index, i, mic=True)

            for _ in range(max_attempts):
                # generate displacement
                delta_disp = rs.normal(
                    0.0,
                    scale_stdev(rattle_std, dist_defect_to_nn, dist_defect_to_i),
                    3,
                )  # displacement tails off with distance from defect
                atoms_rattle.positions[i] += delta_disp
                disp_i = atoms_rattle.positions[i] - reference_positions[i]

                # if total displacement of atom is greater than max_disp, then reject delta_disp
                if np.linalg.norm(disp_i) > max_disp:
                    # revert delta_disp
                    atoms_rattle[i].position -= delta_disp
                    continue

                # compute min distance
                if len(i_nbrs) == 0:
                    min_distance = np.inf
                else:
                    min_distance = np.min(atoms_rattle.get_distances(i, i_nbrs, mic=True))

                # accept or reject delta_disp
                if _probability_mc_rattle(min_distance, d_min, width) > rs.rand():
                    # accept delta_disp
                    break

                # revert delta_disp
                atoms_rattle[i].position -= delta_disp
            else:
                raise Exception(f"Maximum attempts ({max_attempts}) for atom {i}")

    return atoms_rattle.positions - reference_positions


def _generate_local_mc_rattled_structures(
    atoms, site_index, n_configs, rattle_std, d_min, seed=42, **kwargs
) -> list:
    r"""
    Returns list of configurations after applying a Monte Carlo local
    rattle.
    Compared to the standard Monte Carlo rattle, here the displacements
    tail off as we move away from the defect site.

    Rattling atom ``i`` is carried out as a Monte Carlo move that is
    accepted with a probability determined from the minimum
    interatomic distance :math:`d_{ij}`.  If :math:`\\min(d_{ij})`` is
    smaller than :math:`d_{min}`` the move is only accepted with a low
    probability.

    This process is repeated for each atom a number of times meaning
    the magnitude of the final displacements is not *directly*
    connected to ``rattle_std``.

    This function has been adapted from https://gitlab.com/materials-modeling/hiphive

    Warning:
        Repeatedly calling this function *without* providing different
        seeds will yield identical or correlated results. To avoid this
        behavior it is recommended to specify a different seed for each
        call to this function.

    Notes:
        The procedure implemented here might not generate a symmetric
        distribution for the displacements ``kwargs`` will be forwarded to
        ``mc_rattle`` (see user guide for a detailed explanation)

    Args:
        atoms (:obj:`ase.Atoms`):
            Prototype structure
        site_index (:obj:`int`):
            Index of defect site in structure (for substitutions or
            interstitials), using ``python`` / ``pymatgen`` 0-indexing.
        n_configs (:obj:`int`):
            Number of structures to generate
        rattle_std (:obj:`float`):
            Rattle amplitude (standard deviation in normal distribution);
            note this value is not connected to the final
            average displacement for the structures
        d_min (:obj:`float`):
            Interatomic distance used for computing the probability for each rattle
            move
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)
        n_iter (:obj:`int`):
            Number of Monte Carlo cycles
        **kwargs:
            Additional keyword arguments to be passed to ``mc_rattle``

    Returns:
        :obj:`list`:
            list of ase.Atoms generated structures
    """
    rs = np.random.RandomState(seed)
    atoms_list = []
    for _ in range(n_configs):
        atoms_tmp = atoms.copy()
        seed = rs.randint(1, 1000000000)
        displacements = _local_mc_rattle_displacements(
            atoms_tmp, site_index, rattle_std, d_min, seed=seed, **kwargs
        )
        atoms_tmp.positions += displacements
        atoms_list.append(atoms_tmp)
    return atoms_list


def local_mc_rattle(
    structure: Structure,
    site_index: Optional[int] = None,  # 0-indexed
    frac_coords: Optional[np.array] = None,  # use frac coords for vacancies
    stdev: Optional[float] = None,
    d_min: Optional[float] = None,
    verbose: Optional[bool] = False,
    n_iter: int = 1,
    active_atoms: Optional[list] = None,
    nbr_cutoff: float = 5,
    width: float = 0.1,
    max_attempts: int = 5000,
    max_disp: float = 2.0,
    seed: int = 42,
) -> Structure:
    """
    Given a ``pymatgen`` ``Structure`` object, apply random displacements to all atomic
    positions, with the displacement distances randomly drawn from a Gaussian
    distribution of standard deviation ``stdev``. The random displacements
    tail off as we move away from the defect site.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), using ``python`` / ``pymatgen`` 0-indexing.
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for
            vacancies).
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution
            from which random atomic displacement distances are drawn during
            rattling. Default is set to 10% of the bulk nearest neighbour
            distance.
        d_min (:obj:`float`):
            Minimum interatomic distance (in Angstroms) in the rattled
            structure. Monte Carlo rattle moves that put atoms at
            distances less than this will be heavily penalised.
            Default is to set this to 80% of the nearest neighbour
            distance in the defect supercell (ignoring interstitials).
        verbose (:obj:`bool`):
            Whether to print information about the rattling process, if
            rattling initially fails with initial ``d_min``.
        n_iter (:obj:`int`):
            Number of Monte Carlo cycles to perform.
            (Default: 1)
        active_atoms (:obj:`list`, optional):
            List of which atomic indices should undergo Monte Carlo rattling.
            (Default: None)
        nbr_cutoff (:obj:`float`):
            The radial cutoff distance (in Angstroms) used to construct the
            list of atomic neighbours for checking interatomic distances.
            (Default: 5)
        width (:obj:`float`):
            Width of the Monte Carlo rattling error function, in Angstroms.
            (Default: 0.1)
        max_disp (:obj:`float`):
            Maximum atomic displacement (in Angstroms) during Monte Carlo
            rattling. Rarely occurs and is used primarily as a safety net.
            (Default: 2.0)
        max_attempts (:obj:`int`):
            Maximum Monte Carlo rattle move attempts allowed for a single atom;
            if this limit is reached an ``Exception`` is raised.
            (Default: 5000)
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)

    Returns:
        :obj:`Structure`:
            Rattled ``pymatgen`` ``Structure`` object
    """
    if frac_coords is not None and not isinstance(frac_coords, np.ndarray):
        frac_coords = np.array(frac_coords)

    ase_struct = structure.to_ase_atoms()
    if active_atoms is not None:
        # select only the distances involving active_atoms
        distance_matrix = structure.distance_matrix[active_atoms, :][:, active_atoms]
    else:
        distance_matrix = structure.distance_matrix

    sorted_distances = np.sort(distance_matrix[distance_matrix > 0.5].flatten())

    if isinstance(frac_coords, np.ndarray):  # Only for vacancies!
        ase_struct.append("V")  # fake "V" at vacancy
        ase_struct.positions[-1] = np.dot(frac_coords, ase_struct.cell)
        site_index = -1
    elif site_index is None:
        raise ValueError(
            "Insufficient information to apply local rattle, no `site_index` or `frac_coords` provided."
        )

    if stdev is None:
        stdev = 0.1 * sorted_distances[0]
        if stdev > 0.4 or stdev < 0.02:
            warnings.warn(
                f"Automatic bond-length detection gave a bulk bond length of {10 * stdev} "
                f"\u212B and thus a rattle `stdev` of {stdev} ( = 10% bond length), "
                f"which is unreasonable. Reverting to 0.25 \u212B. If this is too large, "
                f"set `stdev` manually"
            )
            stdev = 0.25

    if d_min is None:
        d_min = 0.8 * sorted_distances[0]

        if d_min < 1.0:
            warnings.warn(
                f"Automatic bond-length detection gave a bulk bond length of "
                f"{(1 / 0.8) * d_min} \u212B, which is almost certainly too small. "
                f"Reverting to 2.25 \u212B. If this is too large, set `d_min` manually"
            )
            d_min = 2.25

    try:
        local_rattled_ase_struct = _generate_local_mc_rattled_structures(
            ase_struct,
            site_index=site_index,
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

    except Exception as ex:
        if "attempts" in str(ex):
            reduced_d_min = sorted_distances[0] + stdev
            local_rattled_ase_struct = _generate_local_mc_rattled_structures(
                ase_struct,
                site_index=site_index,
                n_configs=1,
                rattle_std=stdev,
                d_min=reduced_d_min,
                n_iter=n_iter,
                active_atoms=active_atoms,
                nbr_cutoff=nbr_cutoff,
                width=width,
                max_attempts=max(7000, max_attempts),  # default is 5000
                max_disp=max_disp,
                seed=seed,
            )[0]

            if verbose:
                warnings.warn(
                    f"Initial rattle with d_min {d_min:.2f} \u212B failed (some bond lengths "
                    f"significantly smaller than this present), setting d_min to"
                    f" {reduced_d_min:.2f} \u212B for this defect."
                )

        else:
            raise ex

    if isinstance(frac_coords, np.ndarray):
        local_rattled_ase_struct.pop(-1)  # remove fake V from vacancy structure
    return Structure.from_ase_atoms(local_rattled_ase_struct)
