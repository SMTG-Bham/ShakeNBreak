"""
Module to parse input and output files for VASP, Quantum Espresso,
FHI-aims, CASTEP and CP2K.
"""
import datetime
import os
import warnings
from typing import TYPE_CHECKING, Optional, Union

import ase
from ase.atoms import Atoms
from monty.re import regrep
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
from pymatgen.core.units import Energy
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

from shakenbreak import analysis

aaa = AseAtomsAdaptor()


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
            and "fhi-aims". Defaults to "vasp".
        filename (:obj:`str`):
            Filename of the output file, if different from the ShakeNBreak defaults
            that are defined in the default input files:
            (i.e. vasp: "OUTCAR", cp2k: "relax.out", espresso: "espresso.out",
            castep: "*.castep", fhi-aims: "aims.out")
            Default to the ShakeNBreak default filenames.

    Returns: None
    """

    def _match(filename, grep_string):
        """Helper function to grep for a string in a file."""
        try:
            return regrep(
                filename=filename,
                patterns={"match": grep_string},
                reverse=True,
                terminate_on_match=True,
            )["match"]
        except Exception:
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
                    f"Moving old {filename} to "
                    f"{filename.replace('.yaml', '')}_{current_datetime}.yaml "
                    "to avoid overwriting"
                )
                os.rename(
                    filename, f"{filename.replace('.yaml', '')}_{current_datetime}.yaml"
                )  # Keep copy of old file
                dumpfn(energies, filename)
        else:
            dumpfn(energies, filename)

    def parse_vasp_energy(defect_dir, dist, energy, outcar):
        """Parse VASP energy from OUTCAR file."""
        converged = False
        if os.path.exists(os.path.join(defect_dir, dist, "OUTCAR")):
            outcar = os.path.join(defect_dir, dist, "OUTCAR")
        if outcar:  # regrep faster than using Outcar/vasprun class
            try:
                energy = _match(outcar, r"energy\(sigma->0\)\s+=\s+([\d\-\.]+)")[0][0][
                    0
                ]  # Energy of first match
                converged = _match(
                    outcar, "required accuracy"
                )  # check if ionic relaxation converged
            except IndexError:  # no energy match found in OUTCAR, not converged
                pass
            if not converged:
                converged = _match(outcar, "considering this converged")
        return converged, energy, outcar

    def parse_espresso_energy(defect_dir, dist, energy, espresso_out):
        """Parse Quantum Espresso energy from espresso.out file."""
        if os.path.join(
            defect_dir, dist, "espresso.out"
        ):  # Default SnB output filename
            espresso_out = os.path.join(defect_dir, dist, "espresso.out")
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            espresso_out = os.path.join(defect_dir, dist, filename)
        if espresso_out:
            energy_in_Ry = _match(espresso_out, r"!    total energy\s+=\s+([\d\-\.]+)")
            if energy_in_Ry:
                # Energy of first match, in Rydberg
                energy = float(Energy(float(energy_in_Ry[0][0][0]), "Ry").to("eV"))
        return True, energy, espresso_out  # no check for ionic convergence in QE

    def parse_cp2k_energy(defect_dir, dist, energy, cp2k_out):
        """Parse CP2K energy from relax.out file."""
        converged = False
        if os.path.join(defect_dir, dist, "relax.out"):
            cp2k_out = os.path.join(defect_dir, dist, "relax.out")
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            cp2k_out = os.path.join(defect_dir, dist, filename)
        if cp2k_out:
            converged = _match(
                cp2k_out, "GEOMETRY OPTIMIZATION COMPLETED"
            )  # check if ionic
            # relaxation is converged
            energy_in_Ha = _match(
                cp2k_out, r"Total energy:\s+([\d\-\.]+)"
            )  # Energy of first
            # match in Hartree
            energy = float(Energy(energy_in_Ha[0][0][0], "Ha").to("eV"))
        return converged, energy, cp2k_out

    def parse_castep_energy(defect_dir, dist, energy, castep_out):
        """Parse CASTEP energy from .castep file."""
        converged = False
        output_files = [
            file for file in os.listdir(f"{defect_dir}/{dist}") if ".castep" in file
        ]
        if len(output_files) >= 1 and os.path.exists(
            f"{defect_dir}/{dist}/{output_files[0]}"
        ):
            castep_out = f"{defect_dir}/{dist}/{output_files[0]}"
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            castep_out = os.path.join(defect_dir, dist, filename)
        if castep_out:
            # check if ionic relaxation is converged
            # Convergence string deduced from:
            # https://www.tcm.phy.cam.ac.uk/castep/Geom_Opt/node20.html
            # and https://gitlab.mpcdf.mpg.de/nomad-lab/parser-castep/-/
            # blob/master/test/examples/TiO2-geom.castep
            converged = _match(
                castep_out, "Geometry optimization completed successfully."
            )
            energy = _match(castep_out, r"Final Total Energy\s+([\d\-\.]+)")[0][0][
                0
            ]  # Energy of first match in eV
        return converged, energy, castep_out

    def parse_fhi_aims_energy(defect_dir, dist, energy, aims_out):
        """Parse FHI-aims energy from .out file."""
        converged = False
        if os.path.join(defect_dir, dist, "aims.out"):
            aims_out = os.path.join(defect_dir, dist, "aims.out")
        elif os.path.exists(os.path.join(defect_dir, dist, filename)):
            aims_out = os.path.join(defect_dir, dist, filename)
        if aims_out:
            converged = _match(
                aims_out, "converged."
            )  # check if ionic relaxation is converged
            # Convergence string deduced from:
            # https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/3-Periodic-Systems/
            # and https://gitlab.com/fhi-aims-club/tutorials/basics-of-running-fhi-aims/-/
            # blob/master/Tutorial/3-Periodic-Systems/solutions/Si/PBE_relaxation/aims.out
            energy = _match(
                aims_out,
                r"\| Total energy of the DFT / Hartree-Fock s.c.f. calculation\s+:\s+([\d\-\.]+)",
            )
            if energy:
                energy = energy[0][0][0]  # Energy of first match in eV
        return converged, energy, aims_out

    energies = {
        "distortions": {}
    }  # maps each distortion to the energy of the optimised structure

    if defect == os.path.basename(os.path.normpath(path)) and not [
        dir
        for dir in path
        if (os.path.isdir(dir) and os.path.basename(os.path.normpath(dir)) == defect)
    ]:  # if `defect` is in end of `path` and `path` doesn't have a subdirectory called `defect`
        # then remove defect from end of path
        path = os.path.dirname(path)

    defect_dir = f"{path}/{defect}"
    if os.path.isdir(defect_dir):
        dist_dirs = [
            dir
            for dir in os.listdir(defect_dir)
            if os.path.isdir(os.path.join(defect_dir, dir))
            and (
                any(
                    [
                        substring in dir
                        for substring in ["Bond_Distortion", "Rattled", "Unperturbed"]
                    ]
                )
                and "High_Energy" not in dir
            )
        ]  # parse distortion directories

        # load previously-parsed energies file if present
        energies_file = f"{path}/{defect}/{defect}.yaml"
        if os.path.exists(energies_file):
            try:
                prev_energies_dict, _, _ = analysis._sort_data(
                    energies_file, verbose=False
                )
            except Exception:
                prev_energies_dict = {}
        else:
            prev_energies_dict = {}

        # Parse energies and write them to file
        for dist in dist_dirs:
            outcar = None
            energy = None
            converged = False
            if code.lower() == "vasp":
                converged, energy, outcar = parse_vasp_energy(
                    defect_dir, dist, energy, outcar
                )
            elif code.lower() in [
                "espresso",
                "quantum_espresso",
                "quantum-espresso",
                "quantumespresso",
            ]:
                converged, energy, outcar = parse_espresso_energy(
                    defect_dir, dist, energy, outcar
                )
            elif code.lower() == "cp2k":
                converged, energy, outcar = parse_cp2k_energy(
                    defect_dir, dist, energy, outcar
                )
            elif code.lower() == "castep":
                converged, energy, outcar = parse_castep_energy(
                    defect_dir, dist, energy, outcar
                )
            elif code.lower() in ["fhi-aims", "fhi_aims", "fhiaims"]:
                converged, energy, outcar = parse_fhi_aims_energy(
                    defect_dir, dist, energy, outcar
                )

            if analysis._format_distortion_names(dist) != "Label_not_recognized":
                dist_name = analysis._format_distortion_names(dist)
            else:
                dist_name = dist

            if energy:
                if "Unperturbed" in dist:
                    energies[dist_name] = float(energy)
                else:
                    energies["distortions"][dist_name] = float(energy)

                if not converged:
                    print(f"{dist} for {defect} is not fully relaxed")

            elif not outcar:
                # check if energy not found, but was previously parsed, then add to dict
                if dist_name in prev_energies_dict:
                    energies[dist_name] = prev_energies_dict[dist_name]
                elif (
                    "distortions" in prev_energies_dict
                    and dist_name in prev_energies_dict["distortions"]
                ):
                    energies["distortions"][dist_name] = prev_energies_dict[
                        "distortions"
                    ][dist_name]
                else:
                    warnings.warn(f"No output file in {dist} directory")

        # only write energy file if energies have been parsed
        if energies["distortions"]:
            if "Unperturbed" in energies and all(
                [
                    value - energies["Unperturbed"] > 0.1
                    for value in energies["distortions"].values()
                ]
            ):
                warnings.warn(
                    f"All distortions parsed for {defect} are >0.1 eV higher energy than "
                    f"unperturbed, indicating problems with the relaxations. You should first "
                    f"check if the calculations finished ok for this defect species and if this "
                    f"defect charge state is reasonable (often this is the result of an "
                    f"unreasonable charge state). If both checks pass, you likely need to adjust "
                    f"the `stdev` rattling parameter (can occur for hard/ionic/magnetic "
                    f"materials); see https://shakenbreak.readthedocs.io/en/latest/Tips.html#hard"
                    f"-ionic-materials. – This often indicates a complex PES with multiple minima, "
                    f"thus energy-lowering distortions particularly likely, so important to test "
                    f"with reduced `stdev`!"
                )

            elif (
                "Unperturbed" in energies
                and all(
                    [
                        value - energies["Unperturbed"] < -0.1
                        for value in energies["distortions"].values()
                    ]
                )
                and len(energies["distortions"]) > 2
            ):
                warnings.warn(
                    f"All distortions parsed for {defect} are <-0.1 eV lower energy than "
                    f"unperturbed. If this happens for multiple defects/charge states, "
                    f"it can indicate that a bulk phase transformation is occurring within your "
                    f"defect supercell. If so, see "
                    f"https://shakenbreak.readthedocs.io/en/latest/Tips.html#bulk-phase"
                    f"-transformations for advice on dealing with this phenomenon."
                )

            # if any entries in prev_energies_dict not in energies_dict, add to energies_dict and
            # warn user about output files not being parsed for this entry
            for key, value in prev_energies_dict.items():
                if key not in energies:
                    energies[key] = value
                    warnings.warn(
                        f"No output file in {key} directory, but previous parsed result "
                        f"found in {energies_file}, using this."
                    )
                elif key == "distortions":
                    for key2, value2 in prev_energies_dict["distortions"].items():
                        if key2 not in energies["distortions"]:
                            energies["distortions"][key2] = value2
                            warnings.warn(
                                f"No output file in {key2} directory, but previous parsed result "
                                f"found in {energies_file}, using this."
                            )

            energies = sort_energies(energies)
            save_file(energies, defect, path)
        else:
            # check if distortion directories were present but were all "*High_Energy*"
            try:
                high_energy_dist_dirs = [
                    dir
                    for dir in os.listdir(defect_dir)
                    if os.path.isdir(os.path.join(defect_dir, dir))
                    and (
                        any(
                            [
                                substring in dir
                                for substring in [
                                    "Bond_Distortion",
                                    "Rattled",
                                    "Unperturbed",
                                ]
                            ]
                        )
                        and "High_Energy" in dir
                    )
                ]
            except FileNotFoundError:  # no "*High_Energy*" distortion folders
                high_energy_dist_dirs = []
            if (
                high_energy_dist_dirs
            ):  # "*High_Energy*" distortion directories present and no distortion energies parsed
                warnings.warn(
                    f"All distortions for {defect} gave positive energies or forces errors, "
                    f"indicating problems with these relaxations. You should first check that no "
                    f"user INCAR setting is causing this issue. If not, you likely need to adjust "
                    f"the `stdev` rattling parameter (can occur for hard/ionic/magnetic "
                    f"materials); see https://shakenbreak.readthedocs.io/en/latest/Tips.html#hard"
                    f"-ionic-materials."
                )
            else:
                warnings.warn(
                    f"Energies could not be parsed for defect '{defect}' in '{path}'. "
                    f"If these directories are correct, check calculations have converged, "
                    f"and that distortion subfolders match ShakeNBreak naming (e.g. "
                    f"Bond_Distortion_xxx, Rattled, Unperturbed)"
                )
    else:
        warnings.warn(
            f"Energies could not be parsed for defect '{defect}' in '{path}'. "
            f"If these directories are correct, check calculations have converged, "
            f"and that distortion subfolders match ShakeNBreak naming (e.g. "
            f"Bond_Distortion_xxx, Rattled, Unperturbed)"
        )


# Parsing output structures of different codes
def read_vasp_structure(
    file_path: str,
) -> Union[Structure, str]:
    """
    Read VASP structure from `file_path` and convert to `pymatgen` Structure
    object.

    Args:
        file_path (:obj:`str`):
            Path to VASP `CONTCAR` file

    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    abs_path_formatted = file_path.replace("\\", "/")  # for Windows compatibility
    if not os.path.isfile(abs_path_formatted):
        warnings.warn(
            f"{abs_path_formatted} file doesn't exist, storing as "
            f"'Not converged'. Check path & relaxation"
        )
        struct = "Not converged"
    else:
        try:
            struct = Structure.from_file(abs_path_formatted)
        except Exception:
            warnings.warn(
                f"Problem obtaining structure from: {abs_path_formatted}, "
                f"storing as 'Not converged'. Check file & relaxation"
            )
            struct = "Not converged"
    return struct


def read_espresso_structure(
    filename: str,
) -> Union[Structure, str]:
    """
    Reads a structure from Quantum Espresso output and returns it as a pymatgen
    Structure.

    Args:
        filename (:obj:`str`):
            Path to the Quantum Espresso output file.

    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    # ase.io.espresso functions seem a bit buggy, so we use the following implementation
    if os.path.exists(filename):
        with open(filename, "r") as f:
            file_content = f.read()
    else:
        warnings.warn(
            f"{filename} file doesn't exist, storing as 'Not converged'. "
            f"Check path & relaxation"
        )
        structure = "Not converged"
    try:
        if "Begin final coordinates" in file_content:
            file_content = file_content.split("Begin final coordinates")[
                -1
            ]  # last geometry
        if "End final coordinates" in file_content:
            file_content = file_content.split("End final coordinates")[
                0
            ]  # last geometry
        # Parse cell parameters and atomic positions
        cell_lines = [
            line
            for line in file_content.split("CELL_PARAMETERS (angstrom)")[1]
            .split("ATOMIC_POSITIONS (angstrom)")[0]
            .split("\n")
            if line != "" and line != " " and line != "   "
        ]
        atomic_positions = file_content.split("ATOMIC_POSITIONS (angstrom)")[1]
        # Cell parameters
        cell_lines_processed = [
            [float(number) for number in line.split()]
            for line in cell_lines
            if len(line.split()) == 3
        ]
        # Atomic positions
        atomic_positions_processed = [
            [entry for entry in line.split()]
            for line in atomic_positions.split("\n")
            if len(line.split()) >= 4
        ]
        coordinates = [
            [float(entry) for entry in line[1:4]] for line in atomic_positions_processed
        ]
        symbols = [
            entry[0]
            for entry in atomic_positions_processed
            if entry != "" and entry != " " and entry != "  "
        ]
        # Check parsing is ok
        for entry in coordinates:
            assert (
                len(entry) == 3
            )  # Encure 3 numbers (xyz) are parsed from coordinates section
        assert len(symbols) == len(coordinates)  # Same number of atoms and coordinates
        atoms = Atoms(
            symbols=symbols,
            positions=coordinates,
            cell=cell_lines_processed,
            pbc=True,
        )
        aaa = AseAtomsAdaptor()
        structure = aaa.get_structure(atoms)
        structure = structure.get_sorted_structure()  # Sort by atom type
    except Exception:
        warnings.warn(
            f"Problem parsing structure from: {filename}, storing as 'Not "
            f"converged'. Check file & relaxation"
        )
        structure = "Not converged"
    return structure


def read_fhi_aims_structure(filename: str, format="aims") -> Union[Structure, str]:
    """
    Reads a structure from FHI-aims output and returns it as a pymatgen
    Structure.

    Args:
        filename (:obj:`str`):
            Path to the FHI-aims output file.
        format (:obj:`str`):
            either aims-output (output file) aims (geometry file)

    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    if os.path.exists(filename):
        try:
            aaa = AseAtomsAdaptor()
            atoms = ase.io.read(filename=filename, format=format)
            structure = aaa.get_structure(atoms)
            structure = structure.get_sorted_structure()  # Sort sites by
            # electronegativity
        except Exception:
            warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            structure = "Not converged"
    else:
        raise FileNotFoundError(f"File {filename} does not exist!")
    return structure


def read_cp2k_structure(
    filename: str,
) -> Union[Structure, str]:
    """
    Reads a structure from CP2K restart file and returns it as a pymatgen
    Structure.

    Args:
        filename (:obj:`str`):
            Path to the cp2k restart file.

    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    if os.path.exists(filename):
        try:
            aaa = AseAtomsAdaptor()
            atoms = ase.io.read(
                filename=filename,
                format="cp2k-restart",
            )
            structure = aaa.get_structure(atoms)
            structure = structure.get_sorted_structure()  # Sort sites by
            # electronegativity
        except Exception:
            warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            structure = "Not converged"
    else:
        raise FileNotFoundError(f"File {filename} does not exist!")
    return structure


def read_castep_structure(
    filename: str,
) -> Union[Structure, str]:
    """
    Reads a structure from CASTEP output (`.castep`) file and returns it as a
    pymatgen Structure.

    Args:
        filename (:obj:`str`):
            Path to the CASTEP output file.

    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    if os.path.exists(filename):
        try:
            aaa = AseAtomsAdaptor()
            atoms = ase.io.read(
                filename=filename,
                format="castep-castep",
            )
            structure = aaa.get_structure(atoms)
            structure = structure.get_sorted_structure()  # Sort sites by
            # electronegativity
        except Exception:
            warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            structure = "Not converged"
    else:
        raise FileNotFoundError(f"File {filename} does not exist!")
    return structure


def parse_structure(
    code: str,
    structure_path: str,
    structure_filename: str,
) -> Union[Structure, str]:
    """
    Parses the output structure from different codes (VASP, CP2K, Quantum Espresso,
    CATSEP, FHI-aims) and converts it to a pymatgen Structure object.

    Args:
        code (:obj:`str`):
            Code used for geometry optimizations. Valid code names are:
            "vasp", "espresso", "cp2k", "castep" and "fhi-aims" (case insensitive).
        structure_path (:obj:`str`):
            Path to directory containing the structure file.
        structure_filename (:obj:`str`):
            Name of the structure file or the output file containing the
            optimized structure. If not set, the following values will be used
            for each code:
            vasp: "CONTCAR",
            cp2k: "cp2k.restart" (The restart file is used),
            Quantum espresso: "espresso.out",
            castep: "castep.castep" (castep output file is used)
            fhi-aims: geometry.in.next_step

    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    if code.lower() == "vasp":
        if not structure_filename:
            structure_filename = "CONTCAR"
        structure = read_vasp_structure(f"{structure_path}/{structure_filename}")
    elif code.lower() == "espresso":
        if not structure_filename:
            structure_filename = "espresso.out"
        structure = read_espresso_structure(f"{structure_path}/{structure_filename}")
    elif code.lower() == "cp2k":
        if not structure_filename:
            structure_filename = "cp2k.restart"
        structure = read_cp2k_structure(
            filename=f"{structure_path}/{structure_filename}",
        )
    elif code.lower() == "fhi-aims":
        if not structure_filename:
            structure_filename = "geometry.in.next_step"
        structure = read_fhi_aims_structure(
            filename=f"{structure_path}/{structure_filename}",
        )
    elif code.lower() == "castep":
        if not structure_filename:
            structure_filename = "castep.castep"
        structure = read_castep_structure(
            filename=f"{structure_path}/{structure_filename}",
        )
    return structure


# Parse code input files
def parse_qe_input(path: str) -> dict:
    """
    Parse the input file of Quantum Espresso and return it as a dictionary
    of the parameters.

    Args:
        path (:obj:`str`):
            Path to the Quantum Espresso input file.

    Returns: :obj:`dict`
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist!")
    sections = [
        "ATOMIC_SPECIES",
        "ATOMIC_POSITIONS",
        "K_POINTS",
        "ADDITIONAL_K_POINTS",
        "CELL_PARAMETERS",
        "CONSTRAINTS",
        "OCCUPATIONS",
        "ATOMIC_VELOCITIES",
        "HUBBARD",
        "ATOMIC_FORCES",
        "SOLVENTS",
    ]
    with open(path, "r") as f:
        lines = f.readlines()
    params = {}
    for line in lines:
        line = line.strip().partition("#")[0]  # ignore in-line comments
        if line.startswith("&") or any([sec in line for sec in sections]):
            section = line.split()[0].replace("&", "")
            params[section] = {}
        elif line.startswith("#") or line.startswith("!") or line.startswith("/"):
            continue
        elif "=" in line:
            key, value = line.split("=")
            # Convent numeric values to float
            try:
                value = float(value.strip())
            except Exception:
                # string keywords in QE input file are enclosed in quotes
                # so we remove them to avoid too many quotes when generating
                # the input file with ase
                value = value.strip()
                value.replace("'", "")
                value.replace('"', "")
            params[section][key.strip()] = value
        elif len(line.split()) > 1:
            key, value = line.split()[0], " ".join(
                [str(val) for val in line.split()[1:]]
            )
            params[section][key.strip()] = value
    # Remove structure info (if present), as will be re-written with distorted structures
    for section in [
        "ATOMIC_POSITIONS",
        "K_POINTS",
        "ADDITIONAL_K_POINTS",
        "CELL_PARAMETERS",
    ]:
        params.pop(section, None)
    if "SYSTEM" in params.keys():
        for key in ["celldm(1)", "nat", "ntyp", "ibrav"]:
            params["SYSTEM"].pop(key, None)
    return params


def parse_fhi_aims_input(path: str) -> dict:
    """
    Parse the input file of FHI-aims and return it as a dictionary
    of the parameters.

    Args:
        path (:obj:`str`):
            Path to the Quantum Espresso input file.

    Returns: :obj:`dict`
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist!")
    with open(path, "r") as f:
        lines = f.readlines()
    params = {}
    for line in lines:
        line = line.strip().partition("#")[0]  # ignore in-line comments
        if line.startswith("#") or line.startswith("!") or line.startswith("/"):
            continue
        if len(line.split()) > 1:
            if len(line.split()) > 2:
                key, values = line.split()[0], line.split()[1:]
                # Convent numeric values to float
                # (necessary when feeding into the ASE calculator)
                for i in range(len(values)):
                    try:
                        values[i] = float(values[i])
                    except Exception:
                        pass
            else:
                key, values = line.split()[0], line.split()[1]
                try:  # Convent numeric values to float
                    values = float(values)
                except Exception:
                    pass
            params[key.strip()] = values
    return params
