"""
Submodule to generate input files for the ShakenBreak code.
"""

from genericpath import exists
import os
from copy import deepcopy  # See https://stackoverflow.com/a/22341377/14020960 why
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from typing import Optional
import yaml

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.io.vasp.inputs import (
    incar_params,
    BadIncarWarning,
    Kpoints_supported_modes,
)
from pymatgen.io.vasp.sets import DictSet, BadInputSetWarning

import ase
from ase.io.espresso import parse_pwo_start
from ase.calculators.espresso import Espresso
from  ase.calculators.castep import Castep
from  ase.calculators.aims import Aims

from doped.pycdt.utils.vasp import DefectRelaxSet, _check_psp_dir

if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(f"{MODULE_DIR}/../input_files/default_POTCARs.yaml")

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
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
        potcar_settings (:obj:`dict`):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
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
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
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
            "POTCAR directory not set up with pymatgen, so only POSCAR files will be "
            "generated (POTCARs also needed to determine appropriate NELECT setting in "
            "INCAR files)"
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

    # Variable parameters first
    vaspgamincardict = {
        "# ShakeNBreak INCAR with coarse settings to maximise speed with sufficient accuracy for "
        "qualitative structure searching": "",
        "# May want to change NCORE, KPAR, AEXX, ENCUT, NUPDOWN, ISPIN, POTIM": "",
        "NELECT": nelect,
        "IBRION": "2 # While often slower than '1' (RMM-DIIS), this is more stable and "
        "reliable, and vasp_gam relaxations are typically cheap enough to justify it",
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if strong spin polarisation or magnetic behaviour present",
        "ISPIN": "2 # Spin polarisation likely for defects",
        "NCORE": 12,
        "#KPAR": "# No KPAR, only one kpoint",
        "ENCUT": 300,
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "ALGO": "Normal",
        "EDIFF": f"{scaled_ediff(supercell.num_sites)} # May need to reduce for tricky relaxations",
        "EDIFFG": -0.01,
        "HFSCREEN": 0.2,
        "ICHARG": 1,
        "ISIF": 2,
        "ISYM": "0 # Symmetry breaking extremely likely for defects",
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 300,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "ROPT": "1e-3 " * num_elements,
        "SIGMA": 0.05,
    }
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
                    f"Cannot find {k} from your incar_settings in the list of INCAR flags",
                    BadIncarWarning,
                )
        vaspgamincardict.update(incar_settings)

    vaspgamkpts = Kpoints().from_dict(
        {"comment": "Gamma-only KPOINTS from ShakeNBreak", "generation_style": "Gamma"}
    )
    vaspgamincar = Incar.from_dict(vaspgamincardict)

    vaspgamposcar = defect_relax_set.poscar
    if poscar_comment:
        vaspgamposcar.comment = poscar_comment
    vaspgamposcar.write_file(vaspgaminputdir + "POSCAR")
    with zopen(vaspgaminputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspgamincar.get_string())
    vaspgamkpts.write_file(vaspgaminputdir + "KPOINTS")


def write_qe_input(
    structure: Structure,
    pseudopotentials: Optional[dict] = None,
    input_parameters: Optional[str] = None,
    charge: Optional[int] = None,
    output_path: str = ".",
)-> None:
    # Update default parameters with user values
    if input_parameters and pseudopotentials:
        with open(f"{MODULE_DIR}/../input_files/qe_input.yaml", "r") as f:
            default_input_parameters = yaml.safe_load(f)
        if charge:
            default_input_parameters["SYSTEM"]["tot_charge"] = charge
        for section in input_parameters:
            for key in input_parameters[section]:
                if section in default_input_parameters:
                    default_input_parameters[section][key] = input_parameters[section][key]
                else:
                    default_input_parameters.update({section: {key: input_parameters[section][key]}})
    atoms=aaa.get_atoms(structure)
    calc = Espresso(
            pseudopotentials=pseudopotentials,
            tstress=False, 
            tprnfor=True,
            kpts=(1, 1, 1), 
            input_data=default_input_parameters,
        )
    calc.write_input(atoms)
    if output_path != "." or output_path != "./":
        os.replace("./espresso.pwi", f"{output_path}/espresso.pwi")


# Parsing output structures of different codes
def read_vasp_structure(
    file_path: str,
) -> Structure:
    """
    Read VASP structure from `file_path` and convert to `pymatgen` Structure object.

    Args:
        file_path (:obj:`str`):
            Path to VASP `CONTCAR` file

    Returns:
        `pymatgen` Structure object
    """
    abs_path_formatted = file_path.replace("\\", "/")  # for Windows compatibility
    if not os.path.isfile(abs_path_formatted):
        warnings.warn(
            f"{abs_path_formatted} file doesn't exist, storing as 'Not converged'. Check path & "
            f"relaxation"
        )
        struct = "Not converged"
    else:
        try:
            struct = Structure.from_file(abs_path_formatted)
        except:
            warnings.warn(
                f"Problem obtaining structure from: {abs_path_formatted}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            struct = "Not converged"
    return struct


def read_espresso_structure(
    filename: str,
) -> Structure:
    """
    Reads a structure from Quantum Espresso output and returns it as a pymatgen Structure.
    
    Args:
        filename (:obj:`str`):
            Path to the Quantum Espresso output file.
    Returns:
        `pymatgen` Structure object
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            file_content = f.read()
        try:
            if "Begin final coordinates" in file_content:
                file_content = file_content.split("Begin final coordinates")[-1] # last geometry
            cell_lines = file_content.split("CELL_PARAMETERS")[1]
            parsed_info = parse_pwo_start(
                lines=cell_lines.split("\n")
            )
            aaa = AseAtomsAdaptor()
            structure = aaa.get_structure(parsed_info['atoms'])
            structure = structure.get_sorted_structure() # Sort sites by electronegativity
        except:
            warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            structure = "Not converged"
    else:
        raise FileNotFoundError(f"File {filename} does not exist!")
    return structure


def read_fhi_aims_structure(
    filename: str,
) -> Structure:
    """
    Reads a structure from FHI-aims output and returns it as a pymatgen Structure.
    
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
            structure = structure.get_sorted_structure() # Sort sites by electronegativity
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
    Reads a structure from CP2K restart file and returns it as a pymatgen Structure.
    
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
            structure = structure.get_sorted_structure() # Sort sites by electronegativity
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
    Reads a structure from CASTEP output (`.castep`) file and returns it as a pymatgen Structure.
    
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
            structure = structure.get_sorted_structure() # Sort sites by electronegativity
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