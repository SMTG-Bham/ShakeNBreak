"""
Module to read/write structure files for VASP, Quantum Espresso,
FHI-aims, CASTEP and CP2K.
"""
import os
import warnings
from typing import TYPE_CHECKING

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import ase
from ase.atoms import Atoms


if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

aaa = AseAtomsAdaptor()


# Parsing output structures of different codes
def read_vasp_structure(
    file_path: str,
) -> Structure:
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
        except:
            warnings.warn(
                f"Problem obtaining structure from: {abs_path_formatted}, "
                f"storing as 'Not converged'. Check file & relaxation"
            )
            struct = "Not converged"
    return struct


def read_espresso_structure(
    filename: str,
) -> Structure:
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
        with open(filename, 'r') as f:
            file_content = f.read()
    else:
        warnings.warn(
            f"{filename} file doesn't exist, storing as 'Not converged'. "
            f"Check path & relaxation"
        )
        structure = "Not converged"
    try:
        if "Begin final coordinates" in file_content:
            file_content = file_content.split("Begin final coordinates")[-1] # last geometry
        if "End final coordinates" in file_content:
            file_content = file_content.split("End final coordinates")[0] # last geometry
        # Parse cell parameters and atomic positions
        cell_lines = [
            line for line in
            file_content.split("CELL_PARAMETERS (angstrom)")[1].split(
                'ATOMIC_POSITIONS (angstrom)')[0].split("\n")
            if line != "" and line != " " and line != "   "
        ]
        atomic_positions = file_content.split("ATOMIC_POSITIONS (angstrom)")[1]
        # Cell parameters
        cell_lines_processed = [
            [float(number) for number in line.split()] for line in cell_lines
            if len(line.split()) == 3
        ]
        # Atomic positions
        atomic_positions_processed = [
            [entry for entry in line.split()] for line
            in atomic_positions.split("\n") if len(line.split()) >= 4
        ]
        coordinates = [
            [float(entry) for entry in line[1:4]]
            for line in atomic_positions_processed
        ]
        symbols = [
            entry[0] for entry in atomic_positions_processed
            if entry != "" and entry != " " and entry != "  "
        ]
        # Check parsing is ok
        for entry in coordinates:
            assert len(entry) == 3 # Encure 3 numbers (xyz) are parsed from coordinates section
        assert len(symbols) == len(coordinates) # Same number of atoms and coordinates
        atoms = Atoms(
            symbols=symbols,
            positions=coordinates,
            cell=cell_lines_processed,
            pbc=True,
        )
        aaa = AseAtomsAdaptor()
        structure = aaa.get_structure(atoms)
        structure = structure.get_sorted_structure() # Sort by atom type
    except:
        warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
        )
        structure = "Not converged"
    return structure


def read_fhi_aims_structure(
    filename: str,
) -> Structure:
    """
    Reads a structure from fhi-aims output and returns it as a pymatgen
    Structure.

    Args:
        filename (:obj:`str`):
            Path to the fhi-aims output file.
    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    if os.path.exists(filename):
        try:
            aaa = AseAtomsAdaptor()
            atoms = ase.io.read(
                filename = filename,
                format="aims"
            )
            structure = aaa.get_structure(atoms)
            structure = structure.get_sorted_structure() # Sort sites by
            # electronegativity
        except:
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
) -> Structure:
    """
    Reads a structure from cp2k restart file and returns it as a pymatgen
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
            structure = structure.get_sorted_structure() # Sort sites by
            # electronegativity
        except:
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
) -> Structure:
    """
    Reads a structure from castep output (`.castep`) file and returns it as a
    pymatgen Structure.

    Args:
        filename (:obj:`str`):
            Path to the castep output file.
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
            structure = structure.get_sorted_structure() # Sort sites by
            # electronegativity
        except:
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
)-> Structure:
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
        structure = read_vasp_structure(
                f"{structure_path}/{structure_filename}"
            )
    elif code.lower() == "espresso":
        if not structure_filename:
            structure_filename = "espresso.out"
        structure = read_espresso_structure(
            f"{structure_path}/{structure_filename}"
        )
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
