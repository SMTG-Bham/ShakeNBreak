"""
Submodule to generate input files for the ShakenBreak code.
"""

import os
from copy import deepcopy  # See https://stackoverflow.com/a/22341377/14020960 why
import warnings
from typing import TYPE_CHECKING
import numpy as np

from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.io.vasp.inputs import (
    incar_params,
    BadIncarWarning,
    Kpoints_supported_modes,
)
from pymatgen.io.vasp.sets import DictSet, BadInputSetWarning

from doped.pycdt.utils.vasp import DefectRelaxSet, _check_psp_dir


if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "default_POTCARs.yaml"))

# Duplicated code from doped
def scaled_ediff(natoms):  # 1e-5 for 50 atoms, up to max 1e-4
    ediff = float(f"{((natoms/50)*1e-5):.1g}")
    return ediff if ediff <= 1e-4 else 1e-4


def vasp_gam_files(
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
