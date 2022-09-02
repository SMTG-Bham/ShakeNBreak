"""ShakeNBreak command-line-interface (CLI)"""
import os
import pickle
from copy import deepcopy
import numpy as np
import click
import warnings
from subprocess import call
import fnmatch

# Monty and pymatgen
from monty.serialization import loadfn
from monty.json import MontyDecoder

from pymatgen.core.structure import Structure, Element
from pymatgen.io.vasp.inputs import Incar
from pymatgen.analysis.defects.core import Defect

# ShakeNBreak
from shakenbreak import io, input, analysis, plotting, energy_lowering_distortions


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
        defect_coords (:obj:`list`):
            Fractional coordinates of the defect site in the supercell.
        defect_index (:obj:`int`):
            Index of the defect site in the supercell.
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
            "ShakeNBreak CLI is currently only built for point defects, please contact the "
            "developers if you would like to use the method for complex defects"
        )

    if defect_coords is not None:
        if defect_index is None:
            site_displacement_tol = (
                0.1  # distance tolerance for site matching to identify defect
            )
            while site_displacement_tol < 0.8:  # loop over distance tolerances
                if defect_type == "vacancy":
                    possible_defects = sorted(
                        bulk_structure.get_sites_in_sphere(
                            defect_coords, site_displacement_tol, include_index=True
                        ),
                        key=lambda x: x[1],
                    )
                    searched = "bulk"
                else:
                    possible_defects = sorted(
                        defect_structure.get_sites_in_sphere(
                            defect_coords, site_displacement_tol, include_index=True
                        ),
                        key=lambda x: x[1],
                    )
                    searched = "defect"

                if len(possible_defects) == 1:
                    defect_index = possible_defects[0][2]
                    break

                site_displacement_tol += 0.1

            if defect_index is None:
                warnings.warn(
                    f"Coordinates {defect_coords} were specified for (auto-determined) "
                    f"{defect_type} "
                    f"defect, but could not find it in {searched} structure "
                    f"(found {len(possible_defects)} possible defect sites). "
                    "Will attempt auto site-matching instead."
                )

        else:  # both defect_coords and defect_index given
            warnings.warn(
                "Both defect_coords and defect_index were provided. Only one is needed, so "
                "just defect_index will be used to determine the defect site"
            )

    if defect_index is None:
        site_displacement_tol = (
            0.1  # distance tolerance for site matching to identify defect
        )
        while site_displacement_tol < 0.8:  # loop over distance tolerances
            bulk_sites = [site.frac_coords for site in bulk_structure]
            defect_sites = [site.frac_coords for site in defect_structure]
            dist_matrix = defect_structure.lattice.get_all_distances(
                bulk_sites, defect_sites
            )
            min_dist_with_index = [
                [
                    min(dist_matrix[bulk_index]),
                    int(bulk_index),
                    int(dist_matrix[bulk_index].argmin()),
                ]
                for bulk_index in range(len(dist_matrix))
            ]  # list of [min dist, bulk ind, defect ind]

            site_matching_indices = []
            possible_defects = []
            if defect_type in ["vacancy", "interstitial"]:
                for mindist, bulk_index, def_struc_index in min_dist_with_index:
                    if mindist < site_displacement_tol:
                        site_matching_indices.append([bulk_index, def_struc_index])
                    elif defect_type == "vacancy":
                        possible_defects.append([bulk_index, bulk_sites[bulk_index][:]])

                if defect_type == "interstitial":
                    possible_defects = [
                        [ind, fc[:]]
                        for ind, fc in enumerate(defect_sites)
                        if ind not in np.array(site_matching_indices)[:, 1]
                    ]

            elif defect_type == "substitution":
                for mindist, bulk_index, def_struc_index in min_dist_with_index:
                    species_match = (
                        bulk_structure[bulk_index].specie
                        == defect_structure[def_struc_index].specie
                    )
                    if mindist < site_displacement_tol and species_match:
                        site_matching_indices.append([bulk_index, def_struc_index])

                    elif not species_match:
                        possible_defects.append(
                            [def_struc_index, defect_sites[def_struc_index][:]]
                        )

            if len(set(np.array(site_matching_indices)[:, 0])) != len(
                set(np.array(site_matching_indices)[:, 1])
            ):
                raise ValueError(
                    "Error occurred in site_matching routine. Double counting of site matching "
                    f"occurred: {site_matching_indices}\nAbandoning structure parsing."
                )

            if len(possible_defects) == 1:
                defect_index = possible_defects[0][0]
                break

            site_displacement_tol += 0.1

        if defect_index is None:
            raise ValueError(
                "Defect coordinates could not be identified from site-matching. "
                f"Found {len(possible_defects)} possible defect sites – check bulk and defect "
                "structures correspond to the same supercell"
            )

    if defect_type == "vacancy":
        defect_site = bulk_structure[defect_index]
    else:
        defect_site = defect_structure[defect_index]

    for_monty_defect = {
        "@module": "pymatgen.analysis.defects.core",
        "@class": defect_type.capitalize(),
        "structure": bulk_structure,
        "defect_site": defect_site,
    }

    defect = MontyDecoder().process_decoded(for_monty_defect)
    return defect


def generate_defect_dict(defect_object, charges, defect_name) -> dict:
    """
    Create defect dictionary from a pymatgen Defect object.

    Args:
        defect_object (:obj:`Defect`):
            `pymatgen.analysis.defects.core.Defect` object.
        charges (:obj:`list`):
            List of charge states for the defect.
        defect_name(:obj:`str`):
            Name of the defect, to use as key in the defect dict.

    Returns: :obj:`dict`
    """
    single_defect_dict = {
        "name": defect_name,
        "bulk_supercell_site": defect_object.site,
        "defect_type": defect_object.as_dict()["@class"].lower(),
        "site_multiplicity": defect_object.multiplicity,
        "supercell": {
            "size": [1, 1, 1],
            "structure": defect_object.generate_defect_structure(),
        },
        "charges": charges,
    }

    if "Substitution" in str(type(defect_object)):
        # get bulk_site
        poss_deflist = sorted(
            defect_object.bulk_structure.get_sites_in_sphere(
                defect_object.site.coords, 0.01, include_index=True
            ),
            key=lambda x: x[1],
        )
        if not poss_deflist:
            raise ValueError(
                "Error in defect object generation; could not find substitution "
                f"site inside bulk structure for {defect_name}"
            )
        defindex = poss_deflist[0][2]
        sub_site_in_bulk = defect_object.bulk_structure[
            defindex
        ]  # bulk site of substitution

        single_defect_dict["unique_site"] = sub_site_in_bulk
        single_defect_dict["site_specie"] = sub_site_in_bulk.specie.symbol
        single_defect_dict["substitution_specie"] = defect_object.site.specie.symbol

    else:
        single_defect_dict["unique_site"] = defect_object.site
        single_defect_dict["site_specie"] = defect_object.site.specie.symbol

    if single_defect_dict["defect_type"] == "vacancy":
        defects_dict = {
            "vacancies": [
                single_defect_dict,
            ]
        }
    elif single_defect_dict["defect_type"] == "interstitial":
        defects_dict = {
            "interstitials": [
                single_defect_dict,
            ]
        }
    elif single_defect_dict["defect_type"] == "substitution":
        defects_dict = {
            "substitutions": [
                single_defect_dict,
            ]
        }

    return defects_dict


def _parse_defect_dirs(path) -> list:
    """Parse defect directories present in the specified path."""
    return [
        dir
        for dir in os.listdir(path)
        if os.path.isdir(f"{path}/{dir}")
        and any(
            [
                fnmatch.filter(os.listdir(f"{path}/{dir}"), f"{dist}*")
                for dist in ["Rattled", "Unperturbed", "Bond_Distortion"]
            ]
        )  # only parse defect directories that contain distortion folders
    ]


def CommandWithConfigFile(
    config_file_param_name,
):  # can also set CLI options using config file
    """
    Set CLI options using config file.

    Args:
        config_file_param_name (:obj:`str`):
            name of config file
    """

    class CustomCommandClass(click.Command):
        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                config_data = loadfn(config_file)
                for param, value in ctx.params.items():
                    if (
                        ctx.get_parameter_source(param)
                        == click.core.ParameterSource.DEFAULT
                        and param in config_data
                    ):
                        ctx.params[param] = config_data[param]
            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass


# CLI Commands:
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group("snb", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
def snb():
    """ShakeNBreak: Defect structure-searching"""


@snb.command(
    name="generate",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    cls=CommandWithConfigFile("config"),
)
@click.option(
    "--defect",
    "-d",
    help="Path to defect structure",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--bulk",
    "-b",
    help="Path to bulk structure",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--charge", "-c", help="Defect charge state", default=None, type=int)
@click.option(
    "--min-charge",
    "--min",
    help="Minimum defect charge state for which to generate distortions",
    default=None,
    type=int,
)
@click.option(
    "--max-charge",
    "--max",
    help="Maximum defect charge state for which to generate distortions",
    default=None,
    type=int,
)
@click.option(
    "--defect-index",
    "--idx",
    help="Index of defect site in defect structure, in case auto site-matching fails",
    default=None,
    type=int,
)
@click.option(
    "--defect-coords",
    "--def-coords",
    help="Fractional coordinates of defect site in defect structure, in case auto "
    "site-matching fails. In the form 'x y z' (3 arguments)",
    type=click.Tuple([float, float, float]),
    default=None,
)
@click.option(
    "--code",
    help="Code to generate relaxation input files for. "
    "Options: 'VASP', 'CP2K', 'espresso', 'CASTEP', 'FHI-aims'.",
    default="VASP",
    type=str,
    show_default=True,
)
@click.option(
    "--name",
    "-n",
    help="Defect name for folder and metadata generation. Defaults to "
    "pymatgen standard: '{Defect Type}_mult{Supercell "
    "Multiplicity}'",
    default=None,
    type=str,
)
@click.option(
    "--config",
    "-conf",
    help="Config file for advanced distortion settings. See example in"
    " SnB_input_files/example_generate_config.yaml",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "--input_file",
    "-inp",
    help="Input file for the code specified with `--code`, "
    "with relaxation parameters to override defaults (e.g. `INCAR` for `VASP`).",
    default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified defects and generated " "distortions",
    default=False,
    is_flag=True,
    show_default=True,
)
def generate(
    defect,
    bulk,
    charge,
    min_charge,
    max_charge,
    defect_index,
    defect_coords,
    code,
    name,
    config,
    input_file,
    verbose,
):
    """
    Generate the trial distortions and input files for structure-searching
    for a given defect.
    """
    if config is not None:
        user_settings = loadfn(config)
    else:
        user_settings = {}

    func_args = list(locals().keys())
    if user_settings:
        valid_args = [
            "defect",
            "bulk",
            "charge",
            "min_charge",
            "max_charge",
            "defect_index",
            "defect_coords",
            "code",
            "name",
            "config",
            "input_file",
            "verbose",
            "oxidation_states",
            "dict_number_electrons_user",
            "distortion_increment",
            "bond_distortions",
            "local_rattle",
            "distorted_elements",
            "stdev",
            "d_min",
            "n_iter",
            "active_atoms",
            "nbr_cutoff",
            "width",
            "max_attempts",
            "max_disp",
            "seed",
        ]
        for key in func_args:
            if key in user_settings:
                user_settings.pop(key, None)
        for key in list(user_settings.keys()):
            # remove non-sense keys from user_settings
            if key not in valid_args:
                user_settings.pop(key)

    defect_struc = Structure.from_file(defect)
    bulk_struc = Structure.from_file(bulk)

    defect_object = identify_defect(
        defect_structure=defect_struc,
        bulk_structure=bulk_struc,
        defect_index=defect_index,
        defect_coords=defect_coords,
    )
    if verbose and defect_index is None and defect_coords is None:
        click.echo(
            f"Auto site-matching identified {defect} to be "
            f"type {defect_object.as_dict()['@class']} "
            f"with site {defect_object.site}"
        )

    if charge is not None:
        charges = [
            charge,
        ]

    elif max_charge is not None or min_charge is not None:
        if max_charge is None or min_charge is None:
            raise ValueError(
                "If using min/max defect charge, both options must be set!"
            )

        charge_lims = [min_charge, max_charge]
        charges = list(
            range(min(charge_lims), max(charge_lims) + 1)
        )  # just in case user mixes min and max
        # because of different signs ("+1 to -3" etc)

    else:
        warnings.warn(
            "No charge (range) set for defect, assuming default range of +/-2"
        )
        charges = list(range(-2, +3))

    if name is None:
        name = defect_object.name

    defects_dict = generate_defect_dict(defect_object, charges, name)

    Dist = input.Distortions(defects_dict, **user_settings)
    if code.lower() == "vasp":
        if input_file:
            incar = Incar.from_file(input_file)
            incar_settings = incar.as_dict()
            [incar_settings.pop(key, None) for key in ["@class", "@module"]]
        else:
            incar_settings = None
        distorted_defects_dict, distortion_metadata = Dist.write_vasp_files(
            verbose=verbose,
            incar_settings=incar_settings,
        )
    elif code.lower() == "cp2k":
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_cp2k_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            distorted_defects_dict, distortion_metadata = Dist.write_cp2k_files(
                verbose=verbose,
            )
    elif code.lower() in [
        "espresso",
        "quantum_espresso",
        "quantum-espresso",
        "quantumespresso",
    ]:
        print("Code is espresso")
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            print("Writting espresso input files")
            distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
                verbose=verbose,
            )
    elif code.lower() == "castep":
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_castep_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            distorted_defects_dict, distortion_metadata = Dist.write_castep_files(
                verbose=verbose,
            )
    elif code.lower() in ["fhi-aims", "fhi_aims", "fhiaims"]:
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_fhi_aims_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            distorted_defects_dict, distortion_metadata = Dist.write_fhi_aims_files(
                verbose=verbose,
            )
    with open("./parsed_defects_dict.pickle", "wb") as fp:
        pickle.dump(defects_dict, fp)


@snb.command(
    name="generate_all",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    cls=CommandWithConfigFile("config"),
)
@click.option(
    "--defects",
    "-d",
    help="Path root directory with defect folders/files. "
    "Defaults to current directory ('./')",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--structure_file",
    "-s",
    help="File termination/name from which to"
    "parse defect structure from. Only required if defects are stored "
    "in individual directories.",
    type=str,
    default="POSCAR",
    show_default=True,
)
@click.option(
    "--bulk",
    "-b",
    help="Path to bulk structure",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--code",
    help="Code to generate relaxation input files for. "
    "Options: 'VASP', 'CP2K', 'espresso', 'CASTEP', 'FHI-aims'.",
    type=str,
    default="VASP",
    show_default=True,
)
@click.option(
    "--config",
    "-conf",
    help="Config file for advanced distortion settings. See example in "
    "/SnB_input_files/example_generate_all_config.yaml",
    default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    show_default=True,
)
@click.option(
    "--input_file",
    "-inp",
    help="Input file for the code specified with `--code`, "
    "with relaxation parameters to override defaults (e.g. `INCAR` for `VASP`).",
    default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified defects and generated distortions",
    default=False,
    is_flag=True,
    show_default=True,
)
def generate_all(
    defects,
    bulk,
    structure_file,
    code,
    config,
    input_file,
    verbose,
):
    """
    Generate the trial distortions and input files for structure-searching
    for all defects in a given directory.
    """
    bulk_struc = Structure.from_file(bulk)
    defects_dirs = os.listdir(defects)
    if config is not None:
        # In the config file, user can specify index/frac_coords and charges for each defect
        # This way they also provide the names, that should match either the defect folder names
        # or the defect file names (if they are not organised in folders)
        user_settings = loadfn(config)
        if user_settings.get("defects"):
            defect_settings = deepcopy(user_settings["defects"])
            user_settings.pop("defects", None)
        else:
            defect_settings = {}
    else:
        defect_settings, user_settings = {}, {}

    func_args = list(locals().keys())
    # Specified options take precedence over the ones in the config file
    pseudopotentials = None
    if user_settings:
        valid_args = [
            "defects",
            "bulk",
            "structure_file",
            "code",
            "config",
            "input_file",
            "verbose",
            "oxidation_states",
            "dict_number_electrons_user",
            "distortion_increment",
            "bond_distortions",
            "local_rattle",
            "distorted_elements",
            "stdev",
            "d_min",
            "n_iter",
            "active_atoms",
            "nbr_cutoff",
            "width",
            "max_attempts",
            "max_disp",
            "seed",
        ]
        for key in func_args:
            if key in user_settings:
                user_settings.pop(key, None)
        # Parse pseduopotentials from config file, if specified
        if "POTCAR" in user_settings.keys():
            pseudopotentials = {"POTCAR": deepcopy(user_settings["potcar"])}
            user_settings.pop("POTCAR", None)
        if "pseudopotentials" in user_settings.keys():
            pseudopotentials = deepcopy(user_settings["pseudopotentials"])
            user_settings.pop("pseudopotentials", None)
        for key in list(user_settings.keys()):
            # remove non-sense keys from user_settings
            if key not in valid_args:
                user_settings.pop(key)

    def parse_defect_name(defect, defect_settings, structure_file="POSCAR"):
        """Parse defect name from file/folder name"""
        defect_name = None
        # if user included cif/POSCAR as part of the defect
        # structure name, remove it
        for substring in ("cif", "POSCAR", structure_file):
            defect = defect.replace(substring, "")
            for symbol in ("-", "_", "."):
                if defect.endswith(symbol):  # trailing characters
                    defect = defect[:-1]
        # Check if defect specified in config file
        if defect_settings:
            defect_names = defect_settings.keys()
            if defect in defect_names:
                defect_name = defect
            else:
                warnings.warn(
                    f"Defect {defect} not found in config file {config}. "
                    f"Will parse defect name from folders/files."
                )
        if (not defect_name) and any(
            [
                substring in defect.lower()
                for substring in ("as", "vac", "int", "sub", "v", "i")
            ]
        ):
            # if user didnt specify defect names in config file,
            # check if defect filename correspond to standard defect abbreviations
            defect_name = defect
        if not defect_name:
            raise ValueError(
                "Error in defect name parsing; could not parse defect name "
                f"from {defect}. Please include its name in the 'defects' section of "
                "the config file."
            )
        return defect_name

    def parse_defect_charges(defect_name, defect_settings):
        charges = None
        if isinstance(defect_settings, dict):
            if defect_name in defect_settings:
                charges = defect_settings.get(defect_name).get("charges")
        if not charges:
            warnings.warn(
                f"No charge (range) set for defect {defect_name} in config file,"
                " assuming default range of +/-2"
            )
            charges = list(range(-2, +3))
        return charges

    def parse_defect_position(defect_name, defect_settings):
        if defect_settings:
            if defect_name in defect_settings:
                defect_index = defect_settings.get(defect_name).get("defect_index")
                if defect_index:
                    return int(defect_index), None
                else:
                    defect_coords = defect_settings.get(defect_name).get(
                        "defect_coords"
                    )
                    return None, defect_coords
        return None, None

    defects_dict = {}
    for defect in defects_dirs:
        if os.path.isfile(f"{defects}/{defect}"):
            try:  # try to parse structure from it
                defect_struc = Structure.from_file(f"{defects}/{defect}")
                defect_name = parse_defect_name(defect, defect_settings)
            except Exception:
                continue

        elif os.path.isdir(f"{defects}/{defect}"):
            if (
                len(os.listdir(f"{defects}/{defect}")) == 1
            ):  # if only 1 file in directory,
                # assume it's the defect structure
                defect_file = os.listdir(f"{defects}/{defect}")[0]
            else:
                defect_file = [
                    file.lower()
                    for file in os.listdir(f"{defects}/{defect}")
                    if structure_file.lower() in file or "cif" in file
                ][
                    0
                ]  # check for POSCAR and cif by default
            if defect_file:
                defect_struc = Structure.from_file(
                    os.path.join(defects, defect, defect_file)
                )
                defect_name = parse_defect_name(defect, defect_settings)
        else:
            raise FileNotFoundError(f"Could not parse defects from path {defects}")
        # Check if charges / indices are provided in config file
        charges = parse_defect_charges(defect_name, defect_settings)
        defect_index, defect_coords = parse_defect_position(
            defect_name, defect_settings
        )
        defect_object = identify_defect(
            defect_structure=defect_struc,
            bulk_structure=bulk_struc,
            defect_index=defect_index,
            defect_coords=defect_coords,
        )
        if verbose:
            click.echo(
                f"Auto site-matching identified {defect} to be "
                f"type {defect_object.as_dict()['@class']} "
                f"with site {defect_object.site}"
            )

        defect_dict = generate_defect_dict(defect_object, charges, defect_name)

        # Add defect entry to full defect_dict
        defect_type = str(list(defect_dict.keys())[0])
        if defect_type in defects_dict:  # vacancies, antisites or interstitials
            defects_dict[defect_type] += deepcopy(defect_dict[defect_type][0])
        else:
            defects_dict.update(
                {
                    defect_type: deepcopy(defect_dict[defect_type]),
                }
            )

    # Apply distortions and write input files
    Dist = input.Distortions(defects_dict, **user_settings)
    if code.lower() == "vasp":
        if input_file:
            incar = Incar.from_file(input_file)
            incar_settings = incar.as_dict()
            [incar_settings.pop(key, None) for key in ["@class", "@module"]]
        else:
            incar_settings = None
        distorted_defects_dict, distortion_metadata = Dist.write_vasp_files(
            verbose=verbose,
            potcar_settings=pseudopotentials,
            incar_settings=incar_settings,
        )
    elif code.lower() == "cp2k":
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_cp2k_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            distorted_defects_dict, distortion_metadata = Dist.write_cp2k_files(
                verbose=verbose,
            )
    elif code.lower() in [
        "espresso",
        "quantum_espresso",
        "quantum-espresso",
        "quantumespresso",
    ]:
        print("Code is espresso")
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
                verbose=verbose,
                pseudopotentials=pseudopotentials,
                input_file=input_file,
            )
        else:
            print("Writting espresso input files")
            distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
                verbose=verbose,
                pseudopotentials=pseudopotentials,
            )
    elif code.lower() == "castep":
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_castep_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            distorted_defects_dict, distortion_metadata = Dist.write_castep_files(
                verbose=verbose,
            )
    elif code.lower() in ["fhi-aims", "fhi_aims", "fhiaims"]:
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_fhi_aims_files(
                verbose=verbose,
                input_file=input_file,
            )
        else:
            distorted_defects_dict, distortion_metadata = Dist.write_fhi_aims_files(
                verbose=verbose,
            )

    # Dump dict with parsed defects to pickle
    with open("./parsed_defects_dict.pickle", "wb") as fp:
        pickle.dump(defects_dict, fp)


@snb.command(
    name="run",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,  # here we often would run with no options/arguments set
)
@click.option(
    "--submit-command",
    "-s",
    help="Job submission command for the HPC scheduler (qsub, sbatch etc).",
    type=str,
    default="qsub",
    show_default=True,
)
@click.option(
    "--job-script",
    "-j",
    help="Filename of the job script file to submit to HPC scheduler.",
    type=str,
    default="job",
    show_default=True,
)
@click.option(
    "--job-name-option",
    "-n",
    help="Flag for specifying the job name option for the HPC scheduler (e.g. '-N' for 'qsub -N "
    "JOBNAME job' (default)).",
    type=str,
    default=None,
)
@click.option(
    "--all",
    "-a",
    help="Loop through all defect folders (then through their distortion subfolders) in the "
    "current directory",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about calculations which have fully converged",
    default=False,
    is_flag=True,
    show_default=True,
)
def run(submit_command, job_script, job_name_option, all, verbose):
    """
    Loop through distortion subfolders for a defect, when run within a defect folder, or for all
    defect folders in the current (top-level) directory if the --all (-a) flag is set, and submit
    jobs to the HPC scheduler.
    """
    optional_flags = "-"
    if all:
        optional_flags += "a"
    if verbose:
        optional_flags += "v"
    if optional_flags == "-":
        optional_flags = ""

    if submit_command == "sbatch" and job_name_option is None:
        job_name_option = "-J"
    elif job_name_option is None:
        job_name_option = "-N"

    call(
        f"{os.path.dirname(__file__)}/bash_scripts/SnB_run.sh {optional_flags} {submit_command}"
        f" {job_script} {job_name_option}",
        shell=True,
    )


@snb.command(
    name="parse",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,  # can be run within defect directory with no options/arguments set
)
@click.option(
    "--defect",
    "-d",
    help="Name of defect species (folder) to parse (e.g. 'vac_1_Cd_0'), if run from "
    "top-level directory or above. Default is current directory name (assumes running from "
    "within defect folder).",
    type=str,
    default=None,
)
@click.option(
    "--all",
    "-a",
    help="Parse energies for all defects present in the specified/current directory",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--path",
    "-p",
    help="Path to the top-level directory containing the defect folder. "
    "Defaults to current directory ('./').",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--code",
    "-c",
    help="Code used to run the geometry optimisations. "
    "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'.",
    type=str,
    default="vasp",
    show_default=True,
)
def parse(defect, all, path, code):
    """
    Parse final energies of defect structures from relaxation output files.
    Parsed energies are written to a `yaml` file in the corresponding defect directory.
    """
    if defect:
        io.parse_energies(defect, path, code)
    elif all:
        defect_dirs = _parse_defect_dirs(path)
        [io.parse_energies(defect, path, code) for defect in defect_dirs]
    else:
        # assume current directory is the defect folder
        try:
            if path != ".":
                warnings.warn(
                    "`--path` option ignored when running from within defect folder (i.e. "
                    "when `--defect` is not specified."
                )
            cwd = os.getcwd()
            defect = cwd.split("/")[-1]
            path = cwd.rsplit("/", 1)[0]
            io.parse_energies(defect, path, code)
        except Exception:
            raise Exception(
                f"Could not parse defect '{defect}' in directory '{path}'. Please either specify "
                f"a defect to parse (with option --defect), run from within a single defect "
                f"directory (without setting --defect) or use the --all flag to parse all "
                f"defects in the specified/current directory."
            )


@snb.command(
    name="analyse",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,  # can be run within defect directory with no options/arguments set
)
@click.option(
    "--defect",
    "-d",
    help="Name of defect species (folder) to analyse and plot (e.g. 'vac_1_Cd_0'), if run from "
    "top-level directory or above. Default is current directory name (assumes running from "
    "within defect folder).",
    type=str,
    default=None,
)
@click.option(
    "--all",
    "-a",
    help="Analyse all defects present in specified directory",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--path",
    "-p",
    help="Path to the top-level directory containing the defect folder(s). "
    "Defaults to current directory.",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--code",
    "-c",
    help="Code used to run the geometry optimisations. "
    "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'.",
    type=str,
    default="vasp",
    show_default=True,
)
@click.option(
    "--ref_struct",
    "-ref",
    help="Structure to use as a reference for comparison "
    "(to compute atomic displacements). Given as a key from"
    "`defect_structures_dict`.",
    type=str,
    default="Unperturbed",
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified energy lowering distortions.",
    default=False,
    is_flag=True,
    show_default=True,
)
def analyse(defect, all, path, code, ref_struct, verbose):
    """
    Generate `csv` file mapping each distortion to its final energy (in eV) and its
    mean displacement (in Angstrom and relative to `ref_struct`).
    """

    def analyse_single_defect(defect, path, code, ref_struct, verbose):
        if not os.path.exists(f"{path}/{defect}") or not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {defect} in the directory {path}.")
        io.parse_energies(defect, path, code)
        defect_energies_dict = analysis.get_energies(
            defect_species=defect, output_path=path, verbose=verbose
        )
        defect_structures_dict = analysis.get_structures(
            defect_species=defect, output_path=path, code=code
        )
        dataframe = analysis.compare_structures(
            defect_structures_dict=defect_structures_dict,
            defect_energies_dict=defect_energies_dict,
            ref_structure=ref_struct,
        )
        dataframe.to_csv(f"{path}/{defect}/{defect}.csv")  # change name to results.csv?
        print(f"Saved results to {path}/{defect}/{defect}.csv")

    if all:
        defect_dirs = _parse_defect_dirs(path)
        for defect in defect_dirs:
            print(f"\nAnalysing {defect}...")
            analyse_single_defect(defect, path, code, ref_struct, verbose)
    elif defect is None:
        # assume current directory is the defect folder
        if path != ".":
            warnings.warn(
                "`--path` option ignored when running from within defect folder ("
                "i.e. when `--defect` is not specified."
            )
        cwd = os.getcwd()
        defect = cwd.split("/")[-1]
        path = cwd.rsplit("/", 1)[0]

    defect = defect.strip("/")  # Remove trailing slash if present
    # Check if defect present in path:
    if path == ".":
        path = os.getcwd()
    if defect in path:
        path = path.replace(defect, "")
    try:
        analyse_single_defect(defect, path, code, ref_struct, verbose)
    except Exception:
        raise Exception(
            f"Could not analyse defect '{defect}' in directory '{path}'. Please "
            f"either specify a defect to analyse (with option --defect), run from within a single "
            f"defect directory (without setting --defect) or use the --all flag to analyse all "
            f"defects in the specified/current directory."
        )


@snb.command(
    name="plot",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,  # can be run within defect directory with no options/arguments set
)
@click.option(
    "--defect",
    "-d",
    help="Name of defect species (folder) to analyse and plot (e.g. 'vac_1_Cd_0'), if run from "
    "top-level directory or above. Default is current directory name (assumes running from "
    "within defect folder).",
    type=str,
    default=None,
)
@click.option(
    "--all",
    "-a",
    help="Analyse all defects present in current/specified directory",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--path",
    "-p",
    help="Path to the top-level directory containing the defect folder(s). "
    "Defaults to current directory.",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--code",
    "-c",
    help="Code used to run the geometry optimisations. "
    "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'.",
    type=str,
    default="vasp",
    show_default=True,
)
@click.option(
    "--colorbar",
    "-cb",
    help="Whether to add a colorbar indicating structural"
    " similarity between each structure and the unperturbed one.",
    type=bool,
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--metric",
    "-m",
    help="If the option `--colorbar` is specified, determines the criteria used"
    " for the structural comparison. Can choose between the summed of atomic"
    " displacements ('disp') or the maximum distance between"
    " matched sites ('max_dist', default).",
    type=str,
    default="max_dist",
    show_default=True,
)
@click.option(
    "--format",
    "-f",
    help="Format to save the plot as.",
    type=str,
    default="svg",
    show_default=True,
)
@click.option(
    "--units",
    "-u",
    help="Units for energy, either 'eV' or 'meV'.",
    type=str,
    default="eV",
    show_default=True,
)
@click.option(
    "--max_energy",
    "-max",
    help="Maximum energy (in eV), relative to the unperturbed structure,"
    " to show on the plot.",
    type=float,
    default=0.5,
    show_default=True,
)
@click.option(
    "--no-title",
    "-nt",
    help="Don't add defect name as plot title.",
    type=bool,
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified energy lowering distortions.",
    default=False,
    is_flag=True,
    show_default=True,
)
def plot(
    defect,
    all,
    path,
    code,
    colorbar,
    metric,
    format,
    units,
    max_energy,
    no_title,
    verbose,
):
    """
    Generate energy vs distortion plots. Optionally, the structural
    similarity between configurations can be illustrated with a colorbar.
    """
    if all:
        defect_dirs = _parse_defect_dirs(path)
        for defect in defect_dirs:
            if verbose:
                print(f"Parsing {defect}...")
            io.parse_energies(defect, path, code)
        # Create defects_dict (matching defect name to charge states)
        defects_wout_charge = [defect.rsplit("_", 1)[0] for defect in defect_dirs]
        defects_dict = {
            defect_wout_charge: [] for defect_wout_charge in defects_wout_charge
        }
        for defect in defect_dirs:
            defects_dict[defect.rsplit("_", 1)[0]].append(defect.rsplit("_", 1)[1])
        plotting.plot_all_defects(
            defects_dict=defects_dict,
            output_path=path,
            add_colorbar=colorbar,
            metric=metric,
            units=units,
            save_format=format,
            add_title=not no_title,
            max_energy_above_unperturbed=max_energy,
        )
    elif defect is None:
        # assume current directory is the defect folder
        if path != ".":
            warnings.warn(
                "`--path` option ignored when running from within defect folder ("
                "i.e. when `--defect` is not specified."
            )
        cwd = os.getcwd()
        defect = cwd.split("/")[-1]
        path = cwd.rsplit("/", 1)[0]

    defect = defect.strip("/")  # Remove trailing slash if present
    # Check if defect present in path:
    if path == ".":
        path = os.getcwd()
    if defect in path:
        path = path.replace(defect, "")
    try:
        io.parse_energies(defect, path, code)
        defect_energies_dict = analysis.get_energies(
            defect_species=defect,
            output_path=path,
            verbose=verbose,
        )
        plotting.plot_defect(
            defect_species=defect,
            energies_dict=defect_energies_dict,
            output_path=path,
            add_colorbar=colorbar,
            metric=metric,
            save_format=format,
            units=units,
            add_title=not no_title,
            max_energy_above_unperturbed=max_energy,
        )
    except Exception:
        raise Exception(
            f"Could not analyse & plot defect '{defect}' in directory '{path}'. Please "
            f"either specify a defect to analyse (with option --defect), run from within a single "
            f"defect directory (without setting --defect) or use the --all flag to analyse all "
            f"defects in the specified/current directory."
        )


@snb.command(
    name="regenerate",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
)
@click.option(
    "--path",
    "-p",
    help="Path to the top-level directory containing the defect folders."
    " Defaults to current directory.",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--code",
    "-c",
    help="Code to generate relaxation input files for. "
    "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'.",
    type=str,
    default="vasp",
    show_default=True,
)
@click.option(
    "--filename",
    "-f",
    help="Name of the file containing the defect structures.",
    type=str,
    default="CONTCAR",
    show_default=True,
)
@click.option(
    "--min",
    help="Minimum energy difference (in eV) between the ground-state"
    " defect structure, relative to the `Unperturbed` structure,"
    " to consider it as having found a new energy-lowering"
    " distortion.",
    type=float,
    default=0.05,
    show_default=True,
)
@click.option(
    "--metastable",
    "-meta",
    help="Whether to also consider non-spontaneous metastable "
    "energy-lowering distortions, as these can become ground-state "
    "distortions for other charge states.",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified energy lowering distortions.",
    default=False,
    is_flag=True,
    show_default=True,
)
def regenerate(path, code, filename, min, metastable, verbose):
    """
    Identify defect species undergoing energy-lowering distortions and
    test these distortions for the other charge states of the defect.
    Considers all identified energy-lowering distortions for each defect
    in each charge state, and screens out duplicate distorted structures
    found for multiple charge states.
    """
    defect_charges_dict = energy_lowering_distortions.read_defects_directories()
    _ = energy_lowering_distortions.get_energy_lowering_distortions(
        defect_charges_dict=defect_charges_dict,
        output_path=path,
        code=code,
        structure_filename=filename,
        write_input_files=True,
        min_e_diff=min,
        metastable=metastable,
        verbose=verbose,
    )


@snb.command(
    name="groundstate",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,  # can be run within defect directory with no options/arguments set
)
@click.option(
    "--directory",
    "-d",
    help="Folder name where the ground state structure will be written to.",
    type=str,
    default="Groundstate",
    show_default=True,
)
@click.option(
    "--groundstate_filename",
    "-gsf",
    help="File name to save the ground state structure as.",
    type=str,
    default="POSCAR",
    show_default=True,
)
@click.option(
    "--structure_filename",
    "-sf",
    help="File name of the output structures/files.",
    type=str,
    default="CONTCAR",
    show_default=True,
)
@click.option(
    "--path",
    "-p",
    help="Path to the top-level directory containing the defect folders."
    " Defaults to current directory.",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about ground state structures and generated folders.",
    default=True,
    is_flag=True,
    show_default=True,
)
def groundstate(
    directory,
    groundstate_filename,
    structure_filename,
    path,
    verbose,
):
    """
    Generate folders with the identified ground state structures. A folder (named
    `directory`) is created with the ground state structure (named
    `groundstate_filename`) for each defect present in the specified path (if `path` is
    the top-level directory) or for the current defect if run within a defect folder.
    If the name of the structure/output files is not specified, the code assumes `CONTCAR`
    (e.g. geometry optimisations performed with VASP). If using a different code,
    please specify the name of the structure/output files.
    """
    # determine if running from within a defect directory or from the top level directory
    if any(
        [
            dir
            for dir in os.listdir()
            if os.path.isdir(dir)
            and any(
                [
                    substring in dir
                    for substring in ["Bond_Distortion", "Rattled", "Unperturbed"]
                ]
            )
        ]
    ):  # distortion subfolders in cwd
        cwd_name = os.getcwd().split("/")[-1]
        dummy_h = Element("H")
        if any(
            [
                substring in cwd_name.lower()
                for substring in ("as", "vac", "int", "sub", "v", "i")
            ]
        ) or any(
            [
                (dummy_h.is_valid_symbol(substring[-2:]) or substring[-2:] == "Va")
                for substring in cwd_name.split("_")
            ]  # underscore preceded by either an element symbol or "Va" (new pymatgen defect
            # naming convention)
        ):  # cwd is defect name, assume current directory is the defect folder
            if path != ".":
                warnings.warn(
                    "`--path` option ignored when running from within defect folder ("
                    "determined to be the case here based on current directory and "
                    "subfolder names)."
                )

            energy_lowering_distortions.write_groundstate_structure(
                all=False,
                output_path=os.getcwd(),
                groundstate_folder=directory,
                groundstate_filename=groundstate_filename,
                structure_filename=structure_filename,
                verbose=verbose,
            )

            return

    # otherwise, assume top level directory is the path
    energy_lowering_distortions.write_groundstate_structure(
        output_path=path,
        groundstate_folder=directory,
        groundstate_filename=groundstate_filename,
        structure_filename=structure_filename,
        verbose=verbose,
    )
