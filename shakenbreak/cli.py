"""ShakeNBreak command-line-interface (CLI)"""
import fnmatch
import os
import sys
import warnings
from copy import deepcopy
from subprocess import call

import click

# Monty and pymatgen
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Outcar

# ShakeNBreak
from shakenbreak import analysis, energy_lowering_distortions, input, io, plotting


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
    "-min",
    help="Minimum defect charge state for which to generate distortions",
    default=None,
    type=int,
)
@click.option(
    "--max-charge",
    "-max",
    help="Maximum defect charge state for which to generate distortions",
    default=None,
    type=int,
)
@click.option(
    "--padding",
    "-p",
    help="If `--charge` or `--min-charge` & `--max-charge` are not set, "
    "defect charges will be set to the range: 0 – {Defect oxidation state}, "
    "with a `--padding` on either side of this range.",
    default=1,
    type=int,
)
@click.option(
    "--defect-index",
    "-idx",
    help="Index of defect site in defect structure (if substitution/interstitial) "
    "or bulk structure (if vacancy), in case auto site-matching fails",
    default=None,
    type=int,
)
@click.option(
    "--defect-coords",
    "-def-coords",
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
    "'{Defect.name}_m{Defect.multiplicity}' for interstitials and "
    "'{Defect.name}_s{Defect.defect_site_index}' for vacancies and substitutions.",
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
    help="Print information about identified defects and generated distortions",
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
    padding,
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
    pseudopotentials = None
    if user_settings:
        valid_args = [
            "defect",
            "bulk",
            "charge",
            "min_charge",
            "max_charge",
            "padding",
            "charges",
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
        # Parse pseudopotentials from config file, if specified
        if "POTCAR" in user_settings.keys():
            pseudopotentials = {"POTCAR": deepcopy(user_settings["POTCAR"])}
            user_settings.pop("POTCAR", None)
        if "pseudopotentials" in user_settings.keys():
            pseudopotentials = deepcopy(user_settings["pseudopotentials"])
            user_settings.pop("pseudopotentials", None)
        for key in list(user_settings.keys()):
            # remove non-sense keys from user_settings
            if key not in valid_args:
                user_settings.pop(key)

    defect_struc = Structure.from_file(defect)
    bulk_struc = Structure.from_file(bulk)

    # Note that here the Defect.defect_structure is the defect `supercell`
    # structure, not the defect `primitive` structure.
    defect_object = input.identify_defect(
        defect_structure=defect_struc,
        bulk_structure=bulk_struc,
        defect_index=defect_index,
        defect_coords=defect_coords,
    )
    if verbose and defect_index is None and defect_coords is None:
        site = defect_object.site
        site_info = (
            f"{site.species_string} at [{site._frac_coords[0]:.3f},"
            f" {site._frac_coords[1]:.3f}, {site._frac_coords[2]:.3f}]"
        )
        click.echo(
            f"Auto site-matching identified {defect} to be "
            f"type {defect_object.as_dict()['@class']} "
            f"with site {site_info}"
        )

    if charge is not None:
        charges = [
            charge,
        ]
        defect_object.user_charges = charges  # Update charge states

    elif max_charge is not None or min_charge is not None:
        if max_charge is None or min_charge is None:
            raise ValueError(
                "If using min/max defect charge, both options must be set!"
            )

        charge_lims = [min_charge, max_charge]
        charges = list(
            range(min(charge_lims), max(charge_lims) + 1)
        )  # just in case user mixes min and max because of different signs ("+1 to -3" etc)
        defect_object.user_charges = charges  # Update charge states

    if user_settings and "charges" in user_settings:
        charges = user_settings.pop("charges", None)
        if defect_object.user_charges:
            warnings.warn(
                "Defect charges were specified using the CLI option, but `charges` "
                "was also specified in the `--config` file – this will be ignored!"
            )
        else:
            defect_object.user_charges = charges  # Update charge states

    if name is None:
        name = input._get_defect_name_from_obj(defect_object)

    # Refactor Defect into list of DefectEntry objects
    defect_entries = [
        input._get_defect_entry_from_defect(defect_object, c)
        for c in defect_object.get_charge_states(padding)
    ]
    # if user_charges not set for all defects, print info about how charge states will be
    # determined
    if not defect_object.user_charges:
        print(
            "Defect charge states will be set to the range: 0 – {Defect oxidation state}, "
            f"with a `padding = {padding}` on either side of this range."
        )
    Dist = input.Distortions(
        defects={
            name: defect_entries,  # So that user can specify defect name.
        },
        **user_settings,
    )
    if code.lower() == "vasp":
        if input_file:
            incar = Incar.from_file(input_file)
            incar_settings = incar.as_dict()
            [incar_settings.pop(key, None) for key in ["@class", "@module"]]
            if not incar_settings:
                warnings.warn(
                    f"Input file {input_file} specified but no valid INCAR tags found. "
                    f"Should be in the format of VASP INCAR file."
                )
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
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
                verbose=verbose,
                pseudopotentials=pseudopotentials,
                input_file=input_file,
            )
        else:
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
    # Save Defect objects to file
    dumpfn(defect_object, "./parsed_defects_dict.json")


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
    "--padding",
    "-p",
    help="For any defects where `charge` is not set in the --config file, "
    "charges will be set to the range: 0 – {Defect oxidation state}, "
    "with a `--padding` on either side of this range.",
    default=1,
    type=int,
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
    padding,
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
            "charges",
            "charge",
            "padding",
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
        # Parse pseudopotentials from config file, if specified
        if "POTCAR" in user_settings.keys():
            pseudopotentials = {"POTCAR": deepcopy(user_settings["POTCAR"])}
            user_settings.pop("POTCAR", None)
        if "pseudopotentials" in user_settings.keys():
            pseudopotentials = deepcopy(user_settings["pseudopotentials"])
            user_settings.pop("pseudopotentials", None)
        for key in list(user_settings.keys()):
            # remove non-sense keys from user_settings
            if key not in valid_args:
                user_settings.pop(key)

    def parse_defect_name(defect, defect_settings, structure_file="POSCAR"):
        """Parse defect name from file name"""
        defect_name = None
        # if user included cif/POSCAR as part of the defect structure name, remove it
        for substring in ("cif", "POSCAR", structure_file):
            if defect != substring:
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
        if not defect_name:
            # if user didn't specify defect names in config file,
            # check if defect filename is recognised
            try:
                defect_name = plotting._format_defect_name(
                    defect, include_site_num_in_name=False
                )
            except Exception:
                try:
                    defect_name = plotting._format_defect_name(
                        f"{defect}_0", include_site_num_in_name=False
                    )
                except Exception:
                    pass
            if defect_name:
                defect_name = defect

        return defect_name

    def parse_defect_charges(defect_name, defect_settings):
        charges = None
        if isinstance(defect_settings, dict):
            if defect_name in defect_settings:
                charges = defect_settings.get(defect_name).get("charges", None)
                if charges is None:
                    charges = [
                        defect_settings.get(defect_name).get("charge", None),
                    ]
        return charges  # determing using padding if not set in config file

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
    for defect in defects_dirs:  # file or directory
        if os.path.isfile(f"{defects}/{defect}"):
            try:  # try to parse structure from it
                defect_struc = Structure.from_file(f"{defects}/{defect}")
                defect_name = parse_defect_name(
                    defect, defect_settings
                )  # None if not recognised

            except Exception:
                continue

        elif os.path.isdir(f"{defects}/{defect}"):
            if (
                len(os.listdir(f"{defects}/{defect}")) == 1
            ):  # if only 1 file in directory, assume it's the defect structure
                defect_file = os.listdir(f"{defects}/{defect}")[0]
            else:
                poss_defect_files = [  # check for POSCAR and cif by default
                    file.lower()
                    for file in os.listdir(f"{defects}/{defect}")
                    if structure_file.lower() in file.lower()
                    or "cif" in file
                    and "bulk" not in file.lower()
                ]
                if len(poss_defect_files) == 1:
                    defect_file = poss_defect_files[0]
                else:
                    warnings.warn(
                        f"Multiple structure files found in {defects}/{defect}, "
                        f"cannot uniquely determine determine which is the defect, "
                        f"skipping."
                    )
                    continue
            if defect_file:
                defect_struc = Structure.from_file(
                    os.path.join(defects, defect, defect_file)
                )
                defect_name = parse_defect_name(defect, defect_settings)
        else:
            warnings.warn(f"Could not parse {defects}/{defect} as a defect, skipping.")
            continue

        # Check if indices are provided in config file
        defect_index, defect_coords = parse_defect_position(
            defect_name, defect_settings
        )
        defect_object = input.identify_defect(
            defect_structure=defect_struc,
            bulk_structure=bulk_struc,
            defect_index=defect_index,
            defect_coords=defect_coords,
        )
        if verbose:
            site = defect_object.site
            site_info = (
                f"{site.species_string} at [{site._frac_coords[0]:.3f},"
                f" {site._frac_coords[1]:.3f}, {site._frac_coords[2]:.3f}]"
            )
            click.echo(
                f"Auto site-matching identified {defect} to be "
                f"type {defect_object.as_dict()['@class']} "
                f"with site {site_info}"
            )

        if defect_name is None:  # name based on defect object
            defect_name = input._get_defect_name_from_obj(defect_object)

        # Update charges if specified in config file
        charges = parse_defect_charges(defect_name, defect_settings)
        defect_object.user_charges = charges

        # Add defect entry to full defects_dict
        # If charges were not specified by use, set them using padding
        for charge in defect_object.get_charge_states(padding=padding):
            defect_entry = input._get_defect_entry_from_defect(defect_object, charge)
            defect_name = input._update_defect_dict(
                defect_entry, defect_name, defects_dict
            )
    # if user_charges not set for all defects, print info about how charge states will be
    # determined
    if all(
        not defect_entry_list[0].defect.user_charges
        for defect_entry_list in defects_dict.values()
    ):
        print(
            "Defect charge states will be set to the range: 0 – {Defect oxidation state}, "
            f"with a `padding = {padding}` on either side of this range."
        )
    # Apply distortions and write input files
    Dist = input.Distortions(defects_dict, **user_settings)
    if code.lower() == "vasp":
        if input_file:
            incar = Incar.from_file(input_file)
            incar_settings = incar.as_dict()
            [incar_settings.pop(key, None) for key in ["@class", "@module"]]
            if incar_settings == {}:
                warnings.warn(
                    f"Input file {input_file} specified but no valid INCAR tags found. "
                    f"Should be in the format of VASP INCAR file."
                )
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
        if input_file:
            distorted_defects_dict, distortion_metadata = Dist.write_espresso_files(
                verbose=verbose,
                pseudopotentials=pseudopotentials,
                input_file=input_file,
            )
        else:
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

        # Dump dict with parsed defects to json
        dumpfn(defects_dict, "./parsed_defects_dict.json")


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
    As well as submitting the initial geometry optimisations, can automatically continue and
    resubmit calculations that have not yet converged (and handle those which have failed),
    see: https://shakenbreak.readthedocs.io/en/latest/Generation.html#submitting-the-geometry-optimisations
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
        f"{os.path.dirname(__file__)}/scripts/SnB_run.sh {optional_flags} {submit_command} {job_script} "
        f"{job_name_option}",
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
    "(to compute atomic displacements). Given as a key from "
    "`defect_structures_dict` (e.g. '-0.4' for 'Bond_Distortion_-40.0%').",
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
    if defect == os.path.basename(
        os.path.normpath(path)
    ):  # remove defect from end of path if present:
        orig_path = path
        path = os.path.dirname(path)
    else:
        orig_path = None
    try:
        analyse_single_defect(defect, path, code, ref_struct, verbose)
    except Exception:
        try:
            analyse_single_defect(defect, orig_path, code, ref_struct, verbose)
        except Exception:
            raise Exception(
                f"Could not analyse defect '{defect}' in directory '{path}'. Please either "
                f"specify a defect to analyse (with option --defect), run from within a single "
                f"defect directory (without setting --defect) or use the --all flag to analyse "
                f"all defects in the specified/current directory."
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
    "--min_energy",
    "-min",
    help="Minimum energy difference (in eV) between the ground-state "
    "distortion and the `Unperturbed` structure to generate the "
    "distortion plot, when `--all` is set.",
    default=0.05,
    type=float,
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
    default="png",
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
    help="Maximum energy (in chosen `units`), relative to the "
    "unperturbed structure, to show on the plot.",
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
    min_energy,
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
        if defect is not None:
            warnings.warn(
                "The option `--defect` is ignored when using the `--all` flag. (All defects in "
                f"`--path` = {path} will be plotted)."
            )
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
        return plotting.plot_all_defects(
            defects_dict=defects_dict,
            output_path=path,
            add_colorbar=colorbar,
            metric=metric,
            units=units,
            min_e_diff=min_energy,
            save_format=format,
            add_title=not no_title,
            max_energy_above_unperturbed=max_energy,
            verbose=verbose,
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
    if defect == os.path.basename(
        os.path.normpath(path)
    ):  # remove defect from end of path if present:
        orig_path = path
        path = os.path.dirname(path)
    else:
        orig_path = None
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
            verbose=verbose,
        )
    except Exception:
        try:
            io.parse_energies(defect, orig_path, code)
            defect_energies_dict = analysis.get_energies(
                defect_species=defect,
                output_path=orig_path,
                verbose=verbose,
            )
            plotting.plot_defect(
                defect_species=defect,
                energies_dict=defect_energies_dict,
                output_path=orig_path,
                add_colorbar=colorbar,
                metric=metric,
                save_format=format,
                units=units,
                add_title=not no_title,
                max_energy_above_unperturbed=max_energy,
                verbose=verbose,
            )
        except Exception:
            raise Exception(
                f"Could not analyse & plot defect '{defect}' in directory '{path}'. Please either "
                f"specify a defect to analyse (with option --defect), run from within a single "
                f"defect directory (without setting --defect) or use the --all flag to analyse all "
                f"defects in the specified/current directory."
            )


@snb.command(
    name="regenerate",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
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
    "--min_energy",
    "-min",
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
    help="Also include metastable energy-lowering distortions that "
    "have been identified, as these can become ground-state "
    "distortions for other charge states.",
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
def regenerate(path, code, filename, min_energy, metastable, verbose):
    """
    Identify defect species undergoing energy-lowering distortions and
    test these distortions for the other charge states of the defect.
    Considers all identified energy-lowering distortions for each defect
    in each charge state, and screens out duplicate distorted structures
    found for multiple charge states. Defect folder names should end with
    charge state after an underscore (e.g. `vac_1_Cd_0` or `Va_Cd_0` etc).
    """
    if path == ".":
        path = os.getcwd()  # more verbose error if no defect folders found in path
    defect_charges_dict = energy_lowering_distortions.read_defects_directories(
        output_path=path
    )
    if not defect_charges_dict:
        raise FileNotFoundError(
            f"No defect folders found in directory '{path}'. Please check the "
            f"directory contains defect folders with names ending in a charge "
            f"state after an underscore (e.g. `vac_1_Cd_0` or `Va_Cd_0` etc)."
        )
    _ = energy_lowering_distortions.get_energy_lowering_distortions(
        defect_charges_dict=defect_charges_dict,
        output_path=path,
        code=code,
        structure_filename=filename,
        write_input_files=True,
        min_e_diff=min_energy,
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
    "--non_verbose",
    "-nv",
    help="Don't print information about ground state structures and generated folders.",
    default=False,
    is_flag=True,
    show_default=True,
)
def groundstate(
    directory,
    groundstate_filename,
    structure_filename,
    path,
    non_verbose,
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
        # check if defect folders also in cwd
        for dir in [dir for dir in os.listdir() if os.path.isdir(dir)]:
            defect_name = None
            try:
                defect_name = plotting._format_defect_name(
                    dir, include_site_num_in_name=False
                )
            except Exception:
                try:
                    defect_name = plotting._format_defect_name(
                        f"{dir}_0", include_site_num_in_name=False
                    )
                except Exception:
                    pass

            if (
                defect_name
            ):  # recognised defect folder found in cwd, warn user and proceed
                # assuming they want to just parse the distortion folders in cwd
                warnings.warn(
                    f"Both distortion folders and defect folders (i.e. {dir}) were "
                    f"found in the current directory. The defect folders will be "
                    f"ignored and the groundstate structure from the distortion folders "
                    f"in this directory will be generated."
                )
                break

        # assume current directory is the defect folder
        if path != ".":
            warnings.warn(
                "`--path` option ignored when running from within defect folder (assumed to be "
                "the case here as distortion folders found in current directory)."
            )

        energy_lowering_distortions.write_groundstate_structure(
            all=False,
            output_path=os.getcwd(),
            groundstate_folder=directory,
            groundstate_filename=groundstate_filename,
            structure_filename=structure_filename,
            verbose=not non_verbose,
        )

        return

    # otherwise, assume top level directory is the path
    energy_lowering_distortions.write_groundstate_structure(
        output_path=path,
        groundstate_folder=directory,
        groundstate_filename=groundstate_filename,
        structure_filename=structure_filename,
        verbose=not non_verbose,
    )


@snb.command(
    name="mag",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
)
@click.option(
    "--outcar",
    "-o",
    help="Path to OUTCAR file",
    default="OUTCAR",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--threshold",
    "-t",
    help="Atoms with absolute magnetisation below this value are considered un-magnetised / "
    "non-spin-polarised. The threshold for total magnetisation is 10x this value.",
    default=0.01,
    type=float,
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    help="Print information about the magnetisation of the system.",
    default=False,
    is_flag=True,
    show_default=True,
)
def mag(outcar, threshold, verbose):
    """
    Checks if the magnetisation (spin polarisation) values of all atoms in the
    VASP calculation are below a certain threshold, by pulling this data from the OUTCAR.
    Returns a shell exit status of 0 if magnetisation is below the threshold and 1 if above.
    """
    try:
        outcar_obj = Outcar(outcar)
        abs_mag_values = [abs(m["tot"]) for m in outcar_obj.magnetization]

    except Exception:
        if verbose:
            print(f"Could not read magnetisation from OUTCAR file at {outcar}")
        sys.exit(1)

    if (
        max(abs_mag_values) < threshold  # no one atomic moment greater than threshold
        and sum(abs_mag_values) < threshold * 10  # total moment less than 10x threshold
    ):
        if verbose:
            print(f"Magnetisation is below threshold (<{threshold} μB/atom)")
        sys.exit(0)
    else:
        if verbose:
            print(f"Magnetisation is above threshold (>{threshold} μB/atom)")
        sys.exit(1)
