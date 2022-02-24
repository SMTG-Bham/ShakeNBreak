"""
Code to generate rattle + BDM distorted structures & input files

"""
from typing import Optional
import numpy as np
import os
import json
from copy import deepcopy # See https://stackoverflow.com/a/22341377/14020960 why
from monty.io import zopen
from monty.serialization import loadfn

from ase.io.vasp import read_vasp, write_vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from doped import vasp_input

from defect_finder.distortions import bdm, rattle
from defect_finder.analyse_defects import *

# Load default INCAR settings for the defect-finder geometry relaxations 
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_incar_settings = loadfn(os.path.join(MODULE_DIR, "incar.yml"))

def update_struct_defect_dict(
    defect_dict: dict,
    structure: Structure, 
    poscar_comment: str) -> dict:
    """
    Given a structure and poscar comment updates the dictionary of folders \
        (generated with doped.vasp_input.prepare_vasp_defect_inputs()) with the given values.
    Args: 
        defect_dict (dict): 
            Dictionary with defect information, as generated with doped prepare_vasp_defect_inputs()
        structure (Structure): 
            defect structure as Structure class
        poscar_comment (str): 
            comment to include in POSCAR
    Returns:
        single defect dict in doped format.
        """
    defect_dict_copy = deepcopy(defect_dict)
    defect_dict_copy["Defect Structure"] = structure
    defect_dict_copy['POSCAR Comment'] = poscar_comment  
    return defect_dict_copy

def calc_number_electrons(
    defect_dict: dict, 
    oxidation_states: dict,
    ) -> int:
    """ 
    Given the defect_dict and oxidation_states, calculates the number of extra/missing e- of defect based on oxidation states.
    Args:
        defect_dict (dict): 
            Defect dictionary in doped.pycdt.core.defectsmaker.ChargedDefectsStructures() format,
        oxidation_states (dict): 
            Dictionary with oxidation states of the atoms in the material 
            (e.g. {"Cd": +2, "Te": -2} ).
    Returns: 
        number of extra/missing e-. Negative number if defect has extra e-, positive if missing e-.
    """   
    oxidation_states['Vac'] = 0 # A vacancy has oxidataion state of zero
    
    # Determine number of extra/missing e- based on defect type and oxidation states
    if defect_dict['defect_type'] == 'vacancy':
        site_specie = str(defect_dict['site_specie'])
        substituting_specie = 'Vac'
                
    elif defect_dict['defect_type'] == 'interstitial':
        substituting_specie = str(defect_dict['site_specie'])
        site_specie = 'Vac' # Consider an interstitial as the interstitial atom substituting a vacant position
    
    elif defect_dict['defect_type'] in ['antisite', 'substitution']:
        site_specie = str(defect_dict['site_specie'])
        substituting_specie = defect_dict['substituting_specie']
        
    num_electrons = - ( oxidation_states[substituting_specie] - oxidation_states[site_specie] ) 
    # print( f"Number of missing/extra electrons of defect {defect['name']}: ", num_electrons )   
    return int(num_electrons)

def calc_number_neighbours(num_electrons: int) -> int:
    """
    Calculates the number of neighbours to distort by considering \
    the number of extra/missing electrons. 
    If the change in the number of defect electrons is equal or lower than 4, then we distort \
        that number of neighbours.
    If it is higher than 4, then the number of neighbours to distort is 8-(change_in_number_electrons). 
    Args:
        num_electrons (int): number of missing/extra electrons in defect
    Returns:
        number of neighbours to distort (int)
    """
    if num_electrons < -4 or num_electrons > 4 : 
        # if number of missing/extra e- higher than 4, then distort a number of neighbours \
        #  given by (8 - num_electrons)
        num_neighbours = abs(8 - abs(num_electrons) )
    else:
        num_neighbours = abs(num_electrons)       
    return abs(num_neighbours)

def apply_rattle_BDM(
    defect_dict: dict, 
    num_nearest_neighbours: int, 
    distortion_factor: float, 
    stdev: float = 0.25,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    ) -> dict :
    """ 
    Applies rattle + BDM to the ideal defect structure. It calls distortion.bdm with either:
            - fractional coordinates (for vacancies) or 
            - atom site (other defect types).
        Args:
            defect_dict (dict): 
                defect dictionary in doped.vasp_input.preprare_vasp_defect_dict format
            num_nearest_neighbours (int):
                number of defect neighbours to distort
            distortion_factor (float): 
                distortion factor applied to defect neighbours
            stdev (float): 
                standard deviation used for rattle function (in A)
                (default: 0.25)
            distorted_element (str):
                Allows the user to specify the neighbouring element to distort. If None, the closest neighbours to the defect will be chosen.
                (default: None)
            verbose (bool):
                Whether to print distortion information (distorted atoms and distances) 
                (default: False)
        Returns:
            Dictionary with rattle+BDM distorted defect structure and BDM parameters used in the distortion
    """
    # Apply bond distortions to defect neighbours
    if defect_dict['defect_type'] == 'vacancy': # for vacancies, we need to use fractional coordinates (no atom site!)
        bdm_distorted_defect  = bdm(
                    structure = defect_dict["supercell"]["structure"],
                    num_nearest_neighbours = num_nearest_neighbours,
                    distortion_factor = distortion_factor,
                    frac_coords = defect_dict["bulk_supercell_site"].frac_coords,
                    distorted_element = distorted_element,
                    verbose = verbose,
                )
    else:
        my_atom_index = len(defect_dict['supercell']['structure']) # defect atom comes last in structure
        bdm_distorted_defect = bdm(
                    defect_dict["supercell"]["structure"],
                    num_nearest_neighbours,
                    distortion_factor,
                    atom_index = my_atom_index,
                    distorted_element = distorted_element,
                    verbose = verbose,
                )
    # Apply rattle to the bond distorted structure
    bdm_distorted_defect["distorted_structure"] = rattle(
        structure = bdm_distorted_defect["distorted_structure"],
        stdev = stdev
        ) 
    return bdm_distorted_defect

def apply_distortions(
    defect_dict: dict, 
    num_nearest_neighbours: int, 
    bdm_distortions: list,
    stdev: float = 0.25,
    distorted_element: Optional[str] = None,
    verbose: bool = False,
    ) -> dict:
    """ 
    Applies rattle (+ bdm) to defect structure depending on the number of neighbours.
    0 neighbours : only rattle; any other case: rattle + bdm.
    Returns a dictionary that maps the distotion to unperturbed and distorted structures
    For rattling only (no electron change), returns a dictionary with rattled and non-distorted structure.
    Args:
        defect_dict (dict): 
            Defect dictionary as generated with doped.
        num_nearest_neighbours (int): 
            Number of nearest neighbours to distort
        bdm_distortions (list):
            List of distortions to apply to nearest neighbours. (e.g. [-0.5, 0.5])
        stdev (float): 
            standard deviation used for rattling. 
            (Default: 0.25)
        distorted_element (str):
            Allows the user to specify the neighbouring element to distort. If None, the closest neighbours to the defect will be chosen.
            (default: None) 
        verbose (bool):
            Whether to print distortion information (distorted atoms and distances)
            (default: False)
    """  
    distorted_defect = {"Unperturbed_Defect": defect_dict, 
                        "Distortions": {},
                        "BDM_parameters": {}}
   
    if num_nearest_neighbours != 0:        
        for distortion in bdm_distortions:
            distortion = round(distortion, ndigits=3)
            if verbose:
                print(f"--Distortion {distortion:.1%}")
            distortion_factor = 1 + distortion 
            bdm_distorted_defect = apply_rattle_BDM(defect_dict, 
                                                    num_nearest_neighbours, 
                                                    distortion_factor = distortion_factor, 
                                                    stdev = stdev,
                                                    distorted_element = distorted_element,
                                                    verbose = verbose,
                                                    )
            distorted_defect["Distortions"][f"{distortion:.1%}_BDM_Distortion"] = bdm_distorted_defect["distorted_structure"]
            distorted_defect["BDM_parameters"] = {"defect_index": bdm_distorted_defect["defect_index"],
                                                  "unique_site" : defect_dict["bulk_supercell_site"].frac_coords,
                                                  "number_distorted_neighbours": num_nearest_neighbours,
                                                  "distorted_atoms": bdm_distorted_defect["distorted_atoms"],
                                                  }
            
    elif num_nearest_neighbours == 0: # when no extra/missing e-, just rattle the structure
        if defect_dict['defect_type'] == 'vacancy':
            defect_index = None
        else:
            defect_index = len(defect_dict['supercell']['structure']) # defect atom comes last in structure    
        perturbed_structure = rattle(defect_dict["supercell"]["structure"], 
                                     stdev) 
        distorted_defect["Distortions"]["only_rattled"] = perturbed_structure
        distorted_defect["BDM_parameters"] = {"defect_index": defect_index,
                                              "unique_site": defect_dict["bulk_supercell_site"].frac_coords, 
                                              "number_distorted_neighbours": num_nearest_neighbours,
                                              "distorted_atoms": None,
                                              }
    return distorted_defect

def create_folder(folder_name):
    """Creates folder"""
    path = os.getcwd()
    if not os.path.isdir(path + "/" + folder_name):
        try:
            os.mkdir(path + "/" + folder_name)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        
def create_vasp_input(
    defect_name: str, 
    charged_defect: dict, 
    incar_settings: dict,        
    potcar_settings: Optional[dict] = None,
    bdm_type: str= "BDM",
    ) -> None:
    """
    Creates folders for storing defect-finder files. 
    Args:
        defect_name (str): 
            folder name
        charged_defect (dict):
            dictionary with distorted defects
        bdm_type (str): 
            Type of distortion method. 
            Either 'BDM' (standard method) or 'champion' \
                (if the ground-state distortion found for other charge state is tried for the remaining charge states).
                (default: 'BDM')
        incar_settings (dict): 
            incar settings to update doped default ones
        """
    create_folder(defect_name) # create folder for defect
    create_folder(defect_name + "/" + bdm_type) # either folder (named BDM or champion) where BDM distortions will be written.
    for key, defect_dict in charged_defect.items(): # for each distortion, create its folder  
        potcar_settings_copy = deepcopy(potcar_settings) # vasp_gam_files empties the potcar_settings dict, so make a deepcopy
        vasp_input.vasp_gam_files(
            single_defect_dict = defect_dict, 
            input_dir = f"{defect_name}/{bdm_type}/{defect_name}_{key}", 
            incar_settings = incar_settings,
            potcar_settings = potcar_settings_copy,
            ) 

def apply_defect_finder(
    defects_dict: dict, 
    oxidation_states: dict, 
    incar_settings: dict = default_incar_settings,                 
    dict_number_electrons_user : Optional[dict] = None,
    bdm_increment: float = 0.1,
    bdm_distortions: Optional[list] = None,
    stdev: float = 0.25,
    distorted_elements: Optional[dict] = None,
    bdm_type: str = "BDM",
    potcar_settings: Optional[dict] = None,
    write_files: bool = True,
    verbose: bool = False,
    ):
    """ 
    Applies rattle + bond distortion to all defects given in defect dictionary (in doped format).
    It generates the input files for the vasp_gam relaxations of all distortions.
    It also creates a dictionary with an entry for every defect, which in turn links to a dictionary with 
    all bond distortions (and undistorted) for each charge state of the defect (in case wanna check something).
    Args:
        defects_dict (dict):
            Defect dictionary as generated with doped ChargedDefectsStructures()
        oxidation_states (dict):
            chemical oxidation states of your material, used to determine the number of neighbours to distort
            (e.g {"Cd": +2, "Te": -2} )
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are.
        dict_number_electrons_user (dict):
            Optional argument to set the number of extra/missing e- for the input defects.
            Dictionary with format { defect_name : number_of_electrons } where number_of_electrons is negative for extra e- and positive for missing e-.
            (default: None)
        bdm_increment (float):
            Distortion increment for BDM. Recommended values: 0.1-0.30
            (default: 0.1)
        bdm_distortions (list):
            List of distortions to apply to nearest neighbours instead of default ones. (e.g. [-0.5, 0.5])
            (default: None)
        stdev (float):
            Standard deviation of rattle function that will be aplied to defect structures.
            Recommended values: 0.15 or 0.25. 
            (default: 0.25),
        distorted_elements (dict): 
            Allows the user to specify the element to distort for each defect. 
            Dictionary mapping defect name (without charge state) to chemical symbol (e.g {'vac_1_Cd': 'Te'})
            (default: None)
        bdm_type (str) :
            Leave it as default ("BDM") for standard method. 
            The option "champion" is used when relaxing a defect from the ground state configuration found for other charge states. \
                In this case, only the Unperturbed and rattled configurations are relaxed.
            (default: "BDM")
        potcar_settings (dict): 
            Dictionary of element-potcar name (e.g { "POTCAR" : {"Ti":"Ti_pv"} } to override default potcar \
                (recommended to check default POTCAR's in doped default_POTCARs.yalm').
            (default: None)
        write_files (bool):
            Whether to write output files
            (default: True)
        verbose (bool):
            Whether to print distortion information (bond atoms and distances)
            (default: False)
        """
    vasp_defect_inputs = vasp_input.prepare_vasp_defect_inputs(deepcopy(defects_dict))
    dict_defects = {} # dict to store BDM distortions for all defects
    
    # If the user does not specify certain distortions, use bdm_increment to generate the distortions
    if not bdm_distortions:
        bdm_distortions = list(np.around(np.arange(-0.6, 0.601, bdm_increment), decimals=3)) 
    
    # Create dictionary to keep track of the distortions applied
    bdm_metadata = {
        "BDM_parameters": {
            "BDM_increment": bdm_increment,
            "BDM_distortions": bdm_distortions,
            "rattle_stdev": stdev},
        "defects": {},
        } # dict with BDM parameters, useful for posterior analysis
    
    print(f"Applying defect-finder...", 
    "Will apply the following bond distortions: {bdm_distortions}.", 
    "Then, will rattle with a std dev of {stdev} A \n")
    
    for defect_type in list(defects_dict.keys()):  # loop for vacancies, antisites, interstitials
        for defect in defects_dict[defect_type]: # loop for each defect in dict
        
            defect_name = defect["name"] # name without charge state
            bulk_supercell_site = defect["bulk_supercell_site"]
            if distorted_elements: # read the elements to distort
                try:
                    distorted_element = distorted_elements[defect_name]
                except KeyError:
                    print(f"Problem reading the keys in distorted_elements. Are they the correct defect names (without charge states)?")
                    distorted_element = None
            
            # If the user does not specify the number of missing/extra electrons in your defect, we calculate it
            if dict_number_electrons_user :
                number_electrons = dict_number_electrons_user[defect_name]
            else:
                number_electrons = calc_number_electrons(defect, oxidation_states) 
                
            dict_defects[defect_name] = {}
            bdm_metadata["defects"][defect_name] = {
                "unique_site" :  list(bulk_supercell_site.frac_coords), 
                "charges": {},
                }        
            print("\nDefect:", defect_name)
            if number_electrons < 0 :
                print(f"Number of extra electrons in neutral state: {number_electrons}")
            elif number_electrons >= 0 :
                print(f"Number of missing electrons in neutral state: {number_electrons}")
                
            for charge in defect["charges"]: # loop for each charge state of defect
                
                poscar_comment = vasp_defect_inputs[f"{defect_name}_{charge}"]["POSCAR Comment"]
                charged_defect = {}
    
                # Entry for the unperturbed defect to compare
                charged_defect["Unperturbed_Defect"] = deepcopy(vasp_defect_inputs[f"{defect_name}_{charge}"])
                                                                                
                # Generate perturbed structures
                # Calculate extra/missing e- accounting for the charge state of the defect 
                num_electrons_charged_defect = number_electrons + charge # negative if extra e-, positive if missing e-
                num_nearest_neighbours = calc_number_neighbours(num_electrons_charged_defect) # Number of distorted neighbours for each charge state
                
                print(f"\nDefect {defect_name} in charge state: {charge}. Number of distorted neighbours: {num_nearest_neighbours}")
                distorted_structures = apply_distortions(
                    defect, 
                    num_nearest_neighbours, 
                    bdm_distortions,
                    stdev,
                    distorted_element,
                    verbose = verbose,
                    )
                bdm_metadata["defects"][defect_name]["defect_index"] = distorted_structures["BDM_parameters"]["defect_index"] # store site number of defect
                bdm_metadata["defects"][defect_name]["charges"].update({
                    int(charge): {
                        "number_neighbours": num_nearest_neighbours,
                        "distorted_atoms" : distorted_structures["BDM_parameters"]["distorted_atoms"],
                        } 
                    }) # store BDM parameters used for latter analysis
                
                
                for key_distortion, struct in distorted_structures["Distortions"].items():
                    poscar_comment = key_distortion.split("Distortion")[0] + "_" + vasp_defect_inputs[f"{defect_name}_{charge}"]["POSCAR Comment"] + "__num_neighbours=" + str(num_nearest_neighbours)
                    charged_defect[key_distortion] = update_struct_defect_dict(vasp_defect_inputs[f"{defect_name}_{charge}"],
                                                                               struct,  
                                                                               poscar_comment,
                                                                               )
     
                dict_defects[defect_name][f"{defect_name}_{charge}"] = charged_defect # add charged defect entry to dict
                if write_files :
                    create_vasp_input(f"{defect_name}_{charge}", 
                                      charged_defect, 
                                      incar_settings = incar_settings,
                                      potcar_settings = potcar_settings,
                                      bdm_type = bdm_type,
                                      )
            print() 
            if verbose: print("________________________________________________________") # output easier to read
    
    with open('BDM_metadata.json', 'w') as convert_file:
        convert_file.write(json.dumps(bdm_metadata))
    return dict_defects
