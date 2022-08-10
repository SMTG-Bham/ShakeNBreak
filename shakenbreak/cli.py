"""
ShakeNBreak command-line-interface (CLI)
"""
import os
import warnings
import pickle
import datetime
from copy import deepcopy
from typing import Optional
import numpy as np
import click
import warnings

# Monty and pymatgen
from monty.serialization import loadfn, dumpfn
from monty.json import MontyDecoder
from monty.re import regrep

from pymatgen.core.structure import Structure
from pymatgen.core.units import Energy

# ShakeNBreak
from shakenbreak.input import Distortions
from shakenbreak.analysis import _format_distortion_names, get_energies, get_structures, compare_structures
from shakenbreak.plotting import plot_all_defects, plot_defect
from shakenbreak.energy_lowering_distortions import read_defects_directories, get_energy_lowering_distortions


def identify_defect(
    defect_structure, bulk_structure, defect_coords=None, defect_index=None
):
    """
    By comparing the defect and bulk structures, identify the defect present and its site in
    the supercell, and generate a pymatgen defect object from this.

    Args:
        defect_structure (:obj:`Structure`):
            defect structure
        defect_coords (:obj:`list`):
            Fractional coordinates of the defect site in the supercell.
        defect_index (:obj:`int`):
            Index of the defect site in the supercell.

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


def _generate_defect_dict(defect_object, charges, defect_name) -> dict:
    """Create defect dictionary from a defect object"""
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
        dir for dir in os.listdir(path)
        if os.path.isdir(f"{path}/{dir}")
        and any([
            dist in os.listdir(f"{path}/{dir}") for dist in ["Rattled", "Unperturbed", "Bond_Distortion"]
        ])  # avoid parsing directories that aren't defects
    ]


def parse_energies(
    defect: str,
    path: Optional[str] = ".",
    code: Optional[str] = "vasp",
    filename: Optional[str] = "OUTCAR",
) -> None:
    """
    Parse final energy for all distortions present in the given defect
    directory and write them to a `yaml` file in the defect directory.

    Args:
        defect (:obj:`str`):
            Name of defect to parse, including charge state. Should match the
            name of the defect folder.
        path (:obj: `str`):
            Path to the top-level directory containing the defect folder.
            Defaults to current directory (".").
        code (:obj:`str`):
            Name of ab-initio code used to run the geometry optimisations, case
            insensitive. Options include: "vasp", "cp2k", "espresso", "castep"
            and "fhi-aims". Defaults to 'vasp'.
        filename (:obj:`str`):
            Filename of the output file, if different from the ShakeNBreak defaults
            that are defined in the default input files:
            (i.e. vasp: 'OUTCAR', cp2k: "relax.out", espresso: "espresso.out",
            castep: "*.castep", fhi-aims: "aims.out")
            Default to the ShakeNBreak default filenames.

    Returns: None
    """
    def _match(filename, grep_string):
        """
        Helper function to grep for a string in a file.
        """
        try:
            return regrep(
                filename=filename,
                patterns={"match": grep_string},
                reverse=True,
                terminate_on_match=True
            )["match"]
        except:
            return None

    def sort_energies(defect_energies_dict):
        """Order dict items by key (e.g. from -0.6 to 0 to +0.6)"""
        # sort distortions
        sorted_energies_dict = {
            "distortions": dict(
                sorted(
                    defect_energies_dict["distortions"].items(),
                    key=lambda k: (0, k[0]) if isinstance(k[0], float) else (1, k[0]),
                    # to deal with list of both floats and strings
                    # (https://www.geeksforgeeks.org/sort-mixed-list-in-python/)
                )
            )
        }
        if "Unperturbed" in defect_energies_dict:
            sorted_energies_dict["Unperturbed"] = defect_energies_dict["Unperturbed"]
        return sorted_energies_dict

    def save_file(energies, defect, path):
        """Save yaml file with final energies for each distortion."""
        # File to write energies to
        filename = f"{path}/{defect}/{defect}.yaml"
        # Check if previous version of file exists
        if os.path.exists(filename):
            old_file = loadfn(filename)
            if old_file != energies:
                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
                print(
                    f"Moving old {filename} to {filename.replace('.yaml', '')}_{current_datetime}.yaml to avoid overwriting"
                )
                os.rename(filename, f"{filename.replace('.yaml', '')}_{current_datetime}.yaml")  # Keep copy of old file
                dumpfn(energies, filename)
        else:
            dumpfn(energies, filename)

    def parse_vasp_energy(defect_dir, dist, energy, outcar):
        # regrep faster than using Outcar/vasprun class
        if os.path.exists(os.path.join(defect_dir, dist, "OUTCAR")):
            outcar = os.path.join(defect_dir, dist, "OUTCAR")
        if outcar:
            if _match(outcar, "required accuracy"):  # check if ionic relaxation is converged
                energy = _match(
                    outcar, r"energy\(sigma->0\)\s+=\s+([\d\-\.]+)"
                )[0][0][0] # Energy of first match
        return energy, outcar

    def parse_espresso_energy(defect_dir, dist, energy, outcar):
        if os.path.join(defect_dir, dist, "espresso.out"):  # Default SnB output filename
            outcar = os.path.join(defect_dir, dist, "espresso.out")
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            outcar = os.path.join(defect_dir, dist, filename)
        if outcar:
            energy_in_Ry = _match(
                outcar,
                r"!    total energy\s+=\s+([\d\-\.]+)"
            )
            if energy_in_Ry:
                # Energy of first match, in Rydberg
                energy = float(Energy(float(energy_in_Ry[0][0][0]), "Ry").to("eV"))
        return energy, outcar

    def parse_cp2k_energy(defect_dir, dist, energy, outcar):
        if os.path.join(defect_dir, dist, "relax.out"):
            outcar = os.path.join(defect_dir, dist, "relax.out")
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            outcar = os.path.join(defect_dir, dist, filename)
        if outcar and _match(outcar, "GEOMETRY OPTIMIZATION COMPLETED"):  # check if ionic relaxation is converged
            energy_in_Ha = _match(
                outcar, r"Total energy:\s+([\d\-\.]+)"
            )
            if energy_in_Ha:
                # Energy of first match in Hartree
                energy = float(Energy(energy_in_Ha[0][0][0], "Ha").to("eV"))
        return energy, outcar

    def parse_castep_energy(defect_dir, dist, energy, outcar):
        output_files = [file for file in os.listdir(f"{defect_dir}/{dist}") if ".castep" in file]
        if len(output_files) >= 1 and os.path.exists(f"{defect_dir}/{dist}/{output_files[0]}"):
            outcar = f"{defect_dir}/{dist}/{output_files[0]}"
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            outcar = os.path.join(defect_dir, dist, filename)
        if outcar and _match(outcar, "Geometry optimization completed successfully."):
                # check if ionic relaxation is converged
            # Convergence string deduced from:
            # https://www.tcm.phy.cam.ac.uk/castep/Geom_Opt/node20.html
            # and https://gitlab.mpcdf.mpg.de/nomad-lab/parser-castep/-/blob/master/test/examples/TiO2-geom.castep
            energy = _match(
                outcar, r"Final Total Energy\s+([\d\-\.]+)"
            )[0][0][0] # Energy of first match in eV
        return energy, outcar

    def parse_fhi_aims_energy(defect_dir, dist, energy, outcar):
        if os.path.join(defect_dir, dist, "aims.out"):
            outcar = os.path.join(defect_dir, dist, "aims.out")
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            outcar = os.path.join(defect_dir, dist, filename)
        if outcar and _match(outcar, "converged."): # check if ionic relaxation is converged
            # Convergence string deduced from:
            # https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/3-Periodic-Systems/
            # and https://gitlab.com/fhi-aims-club/tutorials/basics-of-running-fhi-aims/-/blob/master/Tutorial/3-Periodic-Systems/solutions/Si/PBE_relaxation/aims.out
            energy = _match(
                outcar,
                r"\| Total energy of the DFT / Hartree-Fock s.c.f. calculation\s+:\s+([\d\-\.]+)"
            )
            if energy:
                energy = energy[0][0][0] # Energy of first match in eV
        return energy, outcar

    defect_dir = f"{path}/{defect}"
    if os.path.isdir(defect_dir):
        dist_dirs = [
            dir for dir in os.listdir(defect_dir)
            if os.path.isdir(os.path.join(defect_dir, dir))
            and any([substring in dir for substring in ["Bond_Distortion", "Rattled", "Unperturbed"]])
        ]  # parse distortion directories

        # Parse energies and write them to file
        energies = {"distortions": {}}
        for dist in dist_dirs:
            outcar = None
            energy = None
            if code.lower() == "vasp":
                energy, outcar = parse_vasp_energy(defect_dir, dist, energy, outcar)
            elif code.lower() in ["espresso", "quantum_espresso", "quantum-espresso", "quantumespresso"]:
                energy, outcar = parse_espresso_energy(defect_dir, dist, energy, outcar)
            elif code.lower() == "cp2k":
                energy, outcar = parse_cp2k_energy(defect_dir, dist, energy, outcar)
            elif code.lower() == "castep":
                energy, outcar = parse_castep_energy(defect_dir, dist, energy, outcar)
            elif code.lower() in ["fhi-aims", "fhi_aims", "fhiaims"]:
                energy, outcar = parse_fhi_aims_energy(defect_dir, dist, energy, outcar)

            if energy:
                if "Unperturbed" in dist:
                    energies[_format_distortion_names(dist)] = float(energy)
                else:
                    energies["distortions"][_format_distortion_names(dist)] = float(energy)
            elif not outcar:
                warnings.warn(f"No output file in {dist} directory")
            else:
                print(f"{dist} not fully relaxed")

        if energies != {"distortions": {}}:  # only write energy file if energies have been parsed
            energies = sort_energies(energies)
            save_file(energies, defect, path)


def CommandWithConfigFile(
    config_file_param_name,
):  # can also set CLI options using config file
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


## CLI Commands:
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group("snb", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
def snb():
    """
    ShakeNBreak: Defect structure-searching
    """


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
@click.option(
    "--charge",
    "-c",
    help="Defect charge state",
    default=None,
    type=int
)
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
    "Options: 'VASP', 'CP2K', 'espresso', 'CASTEP', 'FHI-aims'. "
    "Defaults to 'VASP'",
    default="VASP",
    type=str,
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
    help="Config file for advanced distortion settings. See example in"
        "/input_files/example_generate_config.yaml",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified defects and generated "
        "distortions",
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
    verbose,
):
    """
    Generate the trial distortions for structure-searching for a given defect.
    """
    if config is not None:
        user_settings = loadfn(config)
    else:
        user_settings = {}

    func_args = list(locals().keys())
    if user_settings:
        for key in func_args:
            if key in user_settings:
                user_settings.pop(key, None)

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

    defects_dict = _generate_defect_dict(defect_object, charges, name)

    Dist = Distortions(defects_dict, **user_settings)
    if code.lower() == "vasp":
        distorted_defects_dict, distortion_metadata = Dist.write_vasp_files(
            verbose=verbose
        )
    elif code.lower() == "cp2k":
        distorted_defects_dict, distortion_metadata = Dist.write_cp2k_files(
            verbose=verbose
        )
    elif code.lower() == ["espresso", "quantum_espresso", "quantum-espresso", "quantumespresso"]:
        distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
            verbose=verbose
        )
    elif code.lower() == "castep":
        distorted_defects_dict, distortion_metadata = Dist.write_castep_files(
            verbose=verbose
        )
    elif code.lower() == ["fhi-aims", "fhi_aims", "fhiaims"]:
        distorted_defects_dict, distortion_metadata = Dist.write_fhi_aims_files(
            verbose=verbose
        )
    with open("./parsed_defects_dict.pickle", "wb") as fp:
        pickle.dump(defects_dict, fp)


# TODO: add option in config file with incar/potcar settings - and analogous for the other codes
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
    default="."
)
@click.option(
    "--structure_file",
    "-f",
    help="File termination/name from which to"
    "parse defect structure from. Only required if defects are stored "
    "in individual directories. Defaults to POSCAR",
    type=str,
    default="POSCAR",
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
    "Options: 'VASP', 'CP2K', 'espresso', 'CASTEP', 'FHI-aims'. "
    "Defaults to 'VASP'",
    type=str,
    default="VASP",
)
@click.option(
    "--config",
    help="Config file for advanced distortion settings. See example in "
        "/input_files/example_generate_all_config.yaml",
    default=None
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about identified defects and generated distortions",
    default=False,
    is_flag=True,
)
def generate_all(
    defects,
    bulk,
    structure_file,
    code,
    config,
    verbose
):
    """
    Generate the trial distortions for structure-searching for all defects
    in a given directory.
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
    if user_settings:
        for key in func_args:
            if key in user_settings:
                user_settings.pop(key, None)

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
            [substring in defect.lower() for substring in ("as", "vac", "int", "sub", "v", "i")]
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
            except:
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
            defect_coords=defect_coords
        )
        if verbose:
            click.echo(
                f"Auto site-matching identified {defect} to be "
                f"type {defect_object.as_dict()['@class']} "
                f"with site {defect_object.site}"
            )

        defect_dict = _generate_defect_dict(defect_object, charges, defect_name)

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
    Dist = Distortions(defects_dict, **user_settings)
    if code.lower() == "vasp":
        distorted_defects_dict, distortion_metadata = Dist.write_vasp_files(
            verbose=verbose
        )
    elif code.lower() == "cp2k":
        distorted_defects_dict, distortion_metadata = Dist.write_cp2k_files(
            verbose=verbose
        )
    elif code.lower() == ["espresso", "quantum_espresso", "quantum-espresso", "quantumespresso"]:
        distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
            verbose=verbose
        )
    elif code.lower() == "castep":
        distorted_defects_dict, distortion_metadata = Dist.write_castep_files(
            verbose=verbose
        )
    elif code.lower() == ["fhi-aims", "fhi_aims", "fhiaims"]:
        distorted_defects_dict, distortion_metadata = Dist.write_fhi_aims_files(
            verbose=verbose
        )
    with open("./parsed_defects_dict.pickle", "wb") as fp:
        pickle.dump(defects_dict, fp)


@snb.command(
    name="parse",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
)
@click.option(
    "--defect",
    "-d",
    help="Name of defect (including charge state) to parse energies for",
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
    help="Path to the top-level directory containing the defect folder."
        "Defaults to current directory ('./').",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--code",
    "-c",
    help ="Code to generate relaxation input files for. "
          "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'. "
          "Defaults to 'vasp'",
    type=str,
    default="vasp",
)
def parse(defect, all, path, code):
    """Parse final energies of defect structures from relaxation output files.
    Parsed energies are written to a `yaml` file in the corresponding defect directory.
    """
    if defect:
        parse_energies(defect, path, code)
    elif all:
        defect_dirs = _parse_defect_dirs(path)
        [parse_energies(defect, path, code) for defect in defect_dirs]
    else:
        # assume current directory is the defect folder
        try:
            if path == ".":
                path = os.getcwd()
            if path.endswith("/"):
                defect =  path.split("/")[-2]
                path = path.rsplit("/", 2)[0]
            else:
                defect = path.split("/")[-1]
                path = path.rsplit("/")[0]
            parse_energies(defect, path, code)
        except:
            warnings.warn(
                "Could not parse energies! Please specify a defect to parse "
                "(with option --defect) or use the --all flag to parse all defects"
                " present in the specified/current directory."
            )


@snb.command(
    name="analyse",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
)
@click.option(
    "--defect",
    "-d",
    help="Name of defect to analyse",
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
    help="Path to the top-level directory containing the defect folder."
        "Defaults to current directory.",
    type=click.Path(exists=True, dir_okay=True),
    default=".",
)
@click.option(
    "--code",
    "-c",
    help ="Code used to run the geometry optimisations. "
        "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'. "
        "Defaults to 'vasp'",
    type=str,
    default="vasp",
)
@click.option(
    "--ref_struct",
    help ="Structure to use as a reference for comparison"
        "(to compute atomic displacements). Given as a key from"
        "`defect_structures_dict`. Defaults to 'Unperturbed'",
    type=str,
    default="Unperturbed",
)
@click.option(
    "--verbose",
    "-v",
    help ="Print information about identified energy lowering distortions.",
    default=False,
    is_flag=True,
    show_default=True,
)
def analyse(defect, all, path, code, ref_struct, verbose):
    """
    Generate csv file mapping each distortion to its final energy and its
    mean displacement (relative to ref_struct).
    """
    def analyse_single_defect(defect, path, code, ref_struct, verbose):
        if not os.path.exists(f"{path}/{defect}") or not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {defect} in the directory {path}."
            )
        parse_energies(defect, path, code)
        defect_energies_dict = get_energies(
            defect_species=defect,
            output_path=path,
            verbose=verbose
        )
        defect_structures_dict = get_structures(
            defect_species=defect,
            output_path=path,
            code=code
        )
        dataframe = compare_structures(
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
    elif defect:
        # Check if defect present in path, in case path not specified and user is
        # analysing defect from inside the defect directory
        if path == ".":
            path = os.getcwd()
        if defect in path:
            path = path.replace(defect, "")
        analyse_single_defect(defect, path, code, ref_struct, verbose)
    elif not defect and not all:
        warnings.warn(
            "No defect specified. Please specify a defect to analyse (with the option --defect)"
            " Alternatively, set the option --all to analyse all defects present in"
            " the specified directory."
        )


@snb.command(
    name="plot",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
)
@click.option(
    "--defect",
    "-d",
    help="Name of defect (inncluding charge state) to analyse",
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
    help="Path to the top-level directory containing the defect folder."
         "Defaults to current directory.",
    type=click.Path(exists=True, dir_okay=True),
    default="."
)
@click.option(
    "--code",
    help ="Code used to run the geometry optimisations. "
        "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'. "
        "Defaults to 'vasp'",
    type=str,
    default="vasp",
)
@click.option(
    "--colorbar",
    help ="Whether to add a colorbar indicating structural"
        " similarity between each structure and the unperturbed one.",
    type=bool,
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--metric",
    "-m",
    help ="If add_colorbar is True, determines the criteria used for the"
        " structural comparison. Can choose between the summed of atomic"
        " displacements ('disp') or the maximum distance between"
        " matched sites ('max_dist', default).",
    type=str,
    default="max_dist",
)
@click.option(
    "--format",
    "-f",
    help ="Format to save the plot as. Defaults to svg.",
    type=str,
    default="svg",
)
@click.option(
    "--units",
    "-u",
    help ="Units for energy, either 'eV' or 'meV' (Default: 'eV')",
    type=str,
    default="eV",
)
@click.option(
    "--max_energy",
    "-max",
    help ="Maximum energy (in eV), relative to the unperturbed structure,"
          " to show on the plot. Defaults to 0.5 eV)",
    type=float,
    default=0.5,
)
@click.option(
    "--title",
    "-t",
    help ="Whether to add defect name as plot title. Defaults to True.",
    type=bool,
    default=True,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help ="Print information about identified energy lowering distortions.",
    default=False,
    is_flag=True,
    show_default=True,
)
def plot(defect, all, path, code, colorbar, metric, format, units, max_energy, title, verbose):
    """
    Generate energy vs distortion plots.
    """
    if all:
        defect_dirs = _parse_defect_dirs(path)
        for defect in defect_dirs:
            if verbose:
                print(f"Parsing {defect}...")
            parse_energies(defect, path, code)
        # Create defects_dict (matching defect name to charge states)
        defects_wout_charge = [defect.rsplit("_", 1)[0] for defect in defect_dirs]
        defects_dict = {
            defect_wout_charge: [] for defect_wout_charge in defects_wout_charge
        }
        for defect in defect_dirs:
            defects_dict[defect.rsplit("_", 1)[0]].append(defect.rsplit("_", 1)[1])
        plot_all_defects(
            defects_dict=defects_dict,
            output_path=path,
            add_colorbar=colorbar,
            metric=metric,
            units=units,
            save_format=format,
            add_title=title,
            max_energy_above_unperturbed=max_energy,
        )
    elif defect:
        parse_energies(defect, path, code)
        defect_energies_dict = get_energies(
            defect_species=defect,
            output_path=path,
            verbose=verbose,
        )
        plot_defect(
            defect_species=defect,
            energies_dict=defect_energies_dict,
            output_path=path,
            add_colorbar=colorbar,
            metric=metric,
            save_format=format,
            units=units,
            add_title=title,
            max_energy_above_unperturbed=max_energy,
        )
    elif not defect and not all:
        warnings.warn(
            "No defect specified. Please specify a defect to plot (with the option --defect)"
            " Alternatively, set the option --all to generate distortion plots for"
            " all defects present in the specified directory."
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
    help ="Code to generate relaxation input files for. "
        "Options: 'vasp', 'cp2k', 'espresso', 'castep', 'fhi-aims'. "
        "Defaults to 'vasp'",
    type=str,
    default="vasp",
)
@click.option(
    "--filename",
    "-f",
    help ="Name of the file containing the structure. Defaults to 'CONTCAR'",
    type=str,
    default="CONTCAR",
)
@click.option(
    "--min",
    help ="Minimum energy difference (in eV) between the ground-state"
        " defect structure, relative to the `Unperturbed` structure,"
        " to consider it as having found a new energy-lowering"
        " distortion. Default is 0.05 eV.",
    type=float,
    default=0.05,
)
@click.option(
    "--verbose",
    "-v",
    help ="Print information about identified energy lowering distortions.",
    default=False,
    is_flag=True,
    show_default=True,
)
def regenerate(path, code, filename, min, verbose):
    """
    Identify defect species undergoing energy-lowering distortions and
    test these distortions for the other charge states of the defect.
    Considers all identified energy-lowering distortions for each defect
    in each charge state, and screens out duplicate distorted structures
    found for multiple charge states.
    """
    defect_charges_dict = read_defects_directories()
    _ = get_energy_lowering_distortions(
        defect_charges_dict=defect_charges_dict,
        output_path=path,
        code=code,
        structure_filename=filename,
        write_input_files=True,
        min_e_diff=min,
        verbose=verbose,
    )
