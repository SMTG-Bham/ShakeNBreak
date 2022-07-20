"""
Submodule to generate input files for the ShakenBreak code.
"""
import os
from copy import deepcopy  # See https://stackoverflow.com/a/22341377/14020960 why
import warnings
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.serialization import loadfn

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.io.vasp.inputs import (
    incar_params,
    BadIncarWarning,
)
from pymatgen.io.vasp.sets import DictSet, BadInputSetWarning

import ase
from ase.atoms import Atoms

from doped.pycdt.utils.vasp import DefectRelaxSet, _check_psp_dir

if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(f"{MODULE_DIR}/../input_files/default_POTCARs.yaml")
# Load default INCAR settings for the ShakenBreak geometry relaxations
default_incar_settings = loadfn(
    os.path.join(MODULE_DIR, "../input_files/incar.yaml")
)

aaa = AseAtomsAdaptor()

# Duplicated code from doped
def scaled_ediff(natoms):  # 1e-5 for 50 atoms, up to max 1e-4
    ediff = float(f"{((natoms/50)*1e-5):.1g}")
    return ediff if ediff <= 1e-4 else 1e-4


def write_vasp_gam_files(
    single_defect_dict: dict,
    input_dir: str = None,
    incar_settings: dict = None,
    potcar_settings: dict = None,
) -> None:
    """
    Generates input files for VASP Gamma-point-only rough relaxation
    (before more expensive vasp_std relaxation)
    Args:
        single_defect_dict (:obj:`dict`):
            Single defect-dictionary from prepare_vasp_defect_inputs()
            output dictionary of defect calculations (see example notebook)
        input_dir (:obj:`str`):
            Folder in which to create vasp_gam calculation inputs folder
            (Recommended to set as the key of the prepare_vasp_defect_inputs()
            output directory)
            (default: None)
        incar_settings (:obj:`dict`):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
            default settings. Highly recommended to look at
            `/input_files/incar.yaml`, or output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
        potcar_settings (:obj:`dict`):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from
            doped.vasp_input to see what the (Pymatgen) syntax and doped
            default settings are.
            (default: None)
    """
    supercell = single_defect_dict["Defect Structure"]
    num_elements = len(supercell.composition.elements)  # for ROPT setting in INCAR
    poscar_comment = (
        single_defect_dict["POSCAR Comment"]
        if "POSCAR Comment" in single_defect_dict
        else None
    )

    # Directory
    vaspgaminputdir = input_dir + "/" if input_dir else "VASP_Files/"
    if not os.path.exists(vaspgaminputdir):
        os.makedirs(vaspgaminputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types
    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings[
                "POTCAR_FUNCTIONAL"
            ]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))

    defect_relax_set = DefectRelaxSet(
        supercell,
        charge=single_defect_dict["Transformation " "Dict"]["charge"],
        user_potcar_settings=potcar_dict["POTCAR"],
        user_potcar_functional=potcar_dict["POTCAR_FUNCTIONAL"],
    )
    potcars = _check_psp_dir()
    if potcars:
        defect_relax_set.potcar.write_file(vaspgaminputdir + "POTCAR")
    else:  # make the folders without POTCARs
        warnings.warn(
            "POTCAR directory not set up with pymatgen, so only POSCAR files "
            "will be generated (POTCARs also needed to determine appropriate "
            "NELECT setting in INCAR files)"
        )
        vaspgamposcar = defect_relax_set.poscar
        if poscar_comment:
            vaspgamposcar.comment = poscar_comment
        vaspgamposcar.write_file(vaspgaminputdir + "POSCAR")
        return  # exit here

    relax_set_incar = defect_relax_set.incar
    try:
        # Only set if change in NELECT
        nelect = relax_set_incar.as_dict()["NELECT"]
    except KeyError:
        # Get NELECT if no change (-dNELECT = 0)
        nelect = defect_relax_set.nelect

    # Update system dependent parameters
    default_incar_settings_copy = default_incar_settings.copy()
    default_incar_settings_copy.update({
        "NELECT": nelect,
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if strong spin polarisation or magnetic behaviour present",
        "EDIFF": f"{scaled_ediff(supercell.num_sites)} # May need to reduce for tricky relaxations",
        "ROPT": ("1e-3 " * num_elements).rstrip(),
    })
    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    f"Cannot find {k} from your incar_settings in the list of "
                    "INCAR flags",
                    BadIncarWarning,
                )
        default_incar_settings_copy.update(incar_settings)

    vaspgamincar = Incar.from_dict(default_incar_settings_copy)

    # kpoints
    vaspgamkpts = Kpoints().from_dict(
        {
            "comment": "Gamma-only KPOINTS from ShakeNBreak",
            "generation_style": "Gamma"
        }
    )

    vaspgamposcar = defect_relax_set.poscar
    if poscar_comment:
        vaspgamposcar.comment = poscar_comment
    vaspgamposcar.write_file(vaspgaminputdir + "POSCAR")
    with zopen(vaspgaminputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspgamincar.get_string())
    vaspgamkpts.write_file(vaspgaminputdir + "KPOINTS")


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
    Reads a structure from FHI-aims output and returns it as a pymatgen
    Structure.

    Args:
        filename (:obj:`str`):
            Path to the FHI-aims output file.
    Returns:
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
    Reads a structure from CP2K restart file and returns it as a pymatgen
    Structure.

    Args:
        filename (:obj:`str`):
            Path to the CP2K restart file.
    Returns:
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
    Reads a structure from CASTEP output (`.castep`) file and returns it as a
    pymatgen Structure.

    Args:
        filename (:obj:`str`):
            Path to the CASTEP output file.
    Returns:
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
    CASTEP, FHI-aims) and converts it to
    a pymatgen Structure object.

    Args:
        code (:obj:`str`):
            Code used for geometry optimizations. Valid code names are:
            "VASP", "espresso", "CP2K" and "FHI-aims".
        structure_path (:obj:`str`):
            Path to directory containing the structure file.
        structure_filename (:obj:`str`):
            Name of the structure file or the output file containing the
            optimized structure. If not set, the following values will be used
            for each code:
            VASP: "CONTCAR",
            CP2K: "cp2k.restart" (The restart file is used),
            Quantum espresso: "espresso.out",
            CASTEP: "castep.castep" (CASTEP output file is used)
            FHI-aims: geometry.in.next_step
    Returns:
        `pymatgen` Structure object
    """
    if code == "VASP":
        if not structure_filename:
            structure_filename = "CONTCAR"
        structure = read_vasp_structure(
                f"{structure_path}/{structure_filename}"
            )
    elif code == "espresso":
        if not structure_filename:
            structure_filename = "espresso.out"
        structure = read_espresso_structure(
            f"{structure_path}/{structure_filename}"
        )
    elif code == "CP2K":
        if not structure_filename:
            structure_filename = "cp2k.restart"
        structure = read_cp2k_structure(
            filename=f"{structure_path}/{structure_filename}",
        )
    elif code == "FHI-aims":
        if not structure_filename:
            structure_filename = "geometry.in.next_step"
        structure = read_fhi_aims_structure(
            filename=f"{structure_path}/{structure_filename}",
        )
    elif code == "CASTEP":
        if not structure_filename:
            structure_filename = "castep.castep"
        structure = read_castep_structure(
            filename=f"{structure_path}/{structure_filename}",
        )
    return structure
