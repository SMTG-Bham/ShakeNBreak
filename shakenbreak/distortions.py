"""Module containing functions for applying distortions to defect structures"""
import os
import sys
import warnings
from typing import Optional

import numpy as np
from ase.neighborlist import NeighborList
from hiphive.structure_generation.rattle import (
    _probability_mc_rattle,
    generate_mc_rattled_structures,
)
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """Format warnings output"""
    return f"{os.path.split(filename)[-1]}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = _warning_on_one_line


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
    Applies bond distortions to `num_nearest_neighbours` of the defect (specified
    by `site_index` (for substitutions or interstitials) or `frac_coords`
    (for vacancies))

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Defect structure as a pymatgen object
        num_nearest_neighbours (:obj:`int`):
            Number of defect nearest neighbours to apply bond distortions to
        distortion factor (:obj:`float`):
            The distortion factor to apply to the bond distance between the
            defect and nearest neighbours. Typical choice is between 0.4 (-60%)
            and 1.6 (+60%).
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), counting from 1.
        frac_coords (:obj:`numpy.ndarray`, optional):
            Fractional coordinates of the defect site in the structure (for
            vacancies).
        distorted_element (:obj:`str`, optional):
            Neighbouring element to distort. If None, the closest neighbours to
            the defect will be chosen. (Default: None)
        verbose (:obj:`bool`, optional):
            Whether to print distortion information. (Default: False)

    Returns:
        :obj:`dict`:
            Dictionary with distorted defect structure and the distortion parameters.
    """
    aaa = AseAtomsAdaptor()
    input_structure_ase = aaa.get_atoms(structure)

    if site_index is not None:  # site_index can be 0
        atom_number = site_index - 1  # Align atom number with python 0-indexing
    elif isinstance(frac_coords, np.ndarray):  # Only for vacancies!
        input_structure_ase.append("V")  # fake "V" at vacancy
        input_structure_ase.positions[-1] = np.dot(
            frac_coords, input_structure_ase.cell
        )
        atom_number = len(input_structure_ase) - 1
    else:
        raise ValueError(
            "Insufficient information to apply bond distortions, no `site_index`"
            " or `frac_coords` provided."
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
        nearest = []  # list of nearest neighbours
        for dist, index, element in distances[1:]:
            if (
                element == distorted_element
                and dist < 4.5
                and len(nearest) < num_nearest_neighbours
            ):
                nearest.append((dist, index, element))

        # if the number of nearest neighbours not reached, add other neighbouring
        # elements
        if len(nearest) < num_nearest_neighbours:
            for i in distances[1:]:
                if (
                    len(nearest) < num_nearest_neighbours
                    and i not in nearest
                    and i[0] < 4.5
                ):
                    nearest.append(i)
            warnings.warn(
                f"{distorted_element} was specified as the nearest neighbour "
                f"element to distort, with `distortion_factor` {distortion_factor} "
                f"but did not find `num_nearest_neighbours` "
                f"({num_nearest_neighbours}) of these elements within 4.5 \u212B "
                f"of the defect site. For the remaining neighbours to distort, "
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
            f"""\tDefect Site Index / Frac Coords: {
            site_index or np.around(frac_coords, decimals=3)}
            Original Neighbour Distances: {nearest}
            Distorted Neighbour Distances:\n\t{distorted}"""
        )

    return bond_distorted_defect


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
    Given a pymatgen Structure object, apply random displacements to all atomic
    positions, with the displacement distances randomly drawn from a Gaussian
    distribution of standard deviation `stdev`.

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
            distance in the defect supercell (ignoring interstitials).
        verbose (:obj:`bool`):
            Whether to print information about the rattling process.
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
            if this limit is reached an `Exception` is raised.
            (Default: 5000)
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)

    Returns:
        :obj:`Structure`:
            Rattled pymatgen Structure object
    """
    aaa = AseAtomsAdaptor()
    ase_struct = aaa.get_atoms(structure)
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
            for i in range(1, 10):  # reduce d_min in 10% increments
                reduced_d_min = d_min * float(1 - (i / 10))
                try:
                    rattled_ase_struct = generate_mc_rattled_structures(
                        ase_struct,
                        n_configs=1,
                        rattle_std=stdev,
                        d_min=reduced_d_min,
                        n_iter=n_iter,
                        active_atoms=active_atoms,
                        nbr_cutoff=nbr_cutoff,
                        width=width,
                        max_attempts=7000,  # default is 5000
                        max_disp=max_disp,
                        seed=seed,
                    )[0]
                    break
                except Exception as ex:
                    if "attempts" in str(ex):
                        continue
                    else:
                        raise ex

            if verbose:
                warnings.warn(
                    f"Initial rattle with d_min {d_min:.2f} \u212B failed (some bond lengths "
                    f"significantly smaller than this present), setting d_min to"
                    f" {reduced_d_min:.2f} \u212B for this defect."
                )

        else:
            raise ex

    rattled_structure = aaa.get_structure(rattled_ase_struct)

    return rattled_structure


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
        site (:obj:`int`):
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
            if this limit is reached an `Exception` is raised.
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
            atomic displacements (`Nx3`)
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

    # Transform to pymatgen structure
    aaa = AseAtomsAdaptor()
    structure = aaa.get_structure(atoms)
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

            for n in range(max_attempts):
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
                    min_distance = np.min(
                        atoms_rattle.get_distances(i, i_nbrs, mic=True)
                    )

                # accept or reject delta_disp
                if _probability_mc_rattle(min_distance, d_min, width) > rs.rand():
                    # accept delta_disp
                    break
                else:
                    # revert delta_disp
                    atoms_rattle[i].position -= delta_disp
            else:
                raise Exception(f"Maxmium attempts ({n}) for atom {i}")
    displacements = atoms_rattle.positions - reference_positions
    return displacements


def _generate_local_mc_rattled_structures(
    atoms, site_index, n_configs, rattle_std, d_min, seed=42, **kwargs
) -> list:
    r"""
    Returns list of configurations after applying a Monte Carlo local
    rattle.
    Compared to the standard Monte Carlo rattle, here the displacements
    tail off as we move away from the defect site.

    Rattling atom `i` is carried out as a Monte Carlo move that is
    accepted with a probability determined from the minimum
    interatomic distance :math:`d_{ij}`.  If :math:`\\min(d_{ij})` is
    smaller than :math:`d_{min}` the move is only accepted with a low
    probability.

    This process is repeated for each atom a number of times meaning
    the magnitude of the final displacements is not *directly*
    connected to `rattle_std`.

    This function has been adapted from https://gitlab.com/materials-modeling/hiphive

    Warning:
        Repeatedly calling this function *without* providing different
        seeds will yield identical or correlated results. To avoid this
        behavior it is recommended to specify a different seed for each
        call to this function.

    Notes:
        The procedure implemented here might not generate a symmetric
        distribution for the displacements `kwargs` will be forwarded to
        `mc_rattle` (see user guide for a detailed explanation)

    Args:
        atoms (:obj:`ase.Atoms`):
            prototype structure
        site_index (:obj:`int`):
            Index of defect site in structure (for substitutions or
            interstitials), counting from 1.
        n_structures (:obj:`int`):
            number of structures to generate
        rattle_std (:obj:`float`):
            rattle amplitude (standard deviation in normal distribution);
            note this value is not connected to the final
            average displacement for the structures
        d_min (:obj:`float`):
            interatomic distance used for computing the probability for each rattle
            move
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)
        n_iter (:obj:`int`):
            number of Monte Carlo cycles

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
    site_index: Optional[int] = None,  # starting from 1
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
    Given a pymatgen Structure object, apply random displacements to all atomic
    positions, with the displacement distances randomly drawn from a Gaussian
    distribution of standard deviation `stdev`. The random displacements
    tail off as we move away from the defect site.

    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        site_index (:obj:`int`, optional):
            Index of defect site in structure (for substitutions or
            interstitials), counting from 1.
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
            Whether to print out information about the rattling process.
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
            if this limit is reached an `Exception` is raised.
            (Default: 5000)
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)

    Returns:
        :obj:`Structure`:
            Rattled pymatgen Structure object
    """
    aaa = AseAtomsAdaptor()
    ase_struct = aaa.get_atoms(structure)
    if active_atoms is not None:
        # select only the distances involving active_atoms
        distance_matrix = structure.distance_matrix[active_atoms, :][:, active_atoms]
    else:
        distance_matrix = structure.distance_matrix

    sorted_distances = np.sort(distance_matrix[distance_matrix > 0.5].flatten())

    if isinstance(site_index, int):
        atom_number = site_index - 1  # Align atom number with python 0-indexing
    elif isinstance(frac_coords, np.ndarray):  # Only for vacancies!
        ase_struct.append("V")  # fake "V" at vacancy
        ase_struct.positions[-1] = np.dot(frac_coords, ase_struct.cell)
        atom_number = len(ase_struct) - 1
    else:
        raise ValueError(
            "Insufficient information to apply local rattle, no `site_index`"
            " or `frac_coords` provided."
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
            site_index=atom_number,
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
                site_index=atom_number,
                n_configs=1,
                rattle_std=stdev,
                d_min=reduced_d_min,
                n_iter=n_iter,
                active_atoms=active_atoms,
                nbr_cutoff=nbr_cutoff,
                width=width,
                max_attempts=7000,  # default is 5000
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
    local_rattled_structure = aaa.get_structure(local_rattled_ase_struct)

    return local_rattled_structure
