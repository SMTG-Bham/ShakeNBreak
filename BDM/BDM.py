"""
Code to generate rattle + BDM distorted structures & input files

@author: Irea Mosquera
"""

import numpy as np
import os
import json
from copy import deepcopy # See https://stackoverflow.com/a/22341377/14020960 why

from ase.io.vasp import read_vasp, write_vasp
from pymatgen.io.ase import AseAtomsAdaptor, Structure
from pymatgen.core.structure import Structure
from doped import vasp_input

from BDM.analyse_defects import *

default_incar_settings={"ADDGRID": False, 
                        "ALGO": 'Normal', 
                        "EDIFFG": -0.01,
                        "IBRION": 2, 
                        "ISPIN": 2, 
                        "POTIM": 0.2,
                       "LVHAR": False, 
                       "LSUBROT": False, 
                       "LREAL": 'Auto', 
                       "LWAVE" : False}

def bdm(
    structure: Structure,
    num_nearest_neighbours: int,
    distortion_factor: float,
    atom_index: int = None, # starting from 1
    frac_coords: np.ndarray = None, # if it's a vacancy, use frac coords instead of site index (as a numpy array)
    verbose: bool = False,
    distorted_element: str = None,
):
    """Applies a bond distortion factor to num_nearest_neighbours of defect (indicated with atom_index (antisites/interstitials) or frac coordinates (vacancies))
    Args:
        structure: 
            defect structure in pymatgen format
        num_nearest_neighbours (int): 
            number of defect neighbours to apply distortion
        distortion factor: 
            bond distortion factor to apply to bond distance between defects and defect nearest neighbours. 
            Often good performance obtained with factors between 0.4 (-60%) and 1.4 (+60%)
        atom_index (int) : 
            atom index of defect, starting from 1 (for antisites or interstitials)
        frac_coords (np.array): 
            for vacancies.
        distorted_element (str): 
            Allows the user to specify the neighbouring element to distort. If None, the closest neighbours to the defect will be chosen.
            (default: None)
        verbose (bool) :
            whether to print distortion information
            (default: False)
    Returns:
        dictionary with BDM distorted defect structure and BDM parameters used in the distortion"""
    aaa = AseAtomsAdaptor()
    input_structure_ase = aaa.get_atoms(structure)

    if atom_index:
        atom_number = atom_index - 1  ## Align atom number with python indexing
    elif isinstance(frac_coords, np.ndarray): ## Only for vacancies!
        input_structure_ase.append("V") ## fake "V" at vacancy
        input_structure_ase.positions[-1] = np.dot(
            frac_coords, input_structure_ase.cell
        )
        atom_number = len(input_structure_ase) - 1
    else:
        print("Where are my atom coordinates or index?")

    neighbours = (
            num_nearest_neighbours + 1
        )  ## Prevent self counting of nearest neighbours
    frac_positions = (
        input_structure_ase.get_scaled_positions()
    )  ## Get fractional atomic coordinates
    distances = [
        ( round(input_structure_ase.get_distance(atom_number, index, mic=True), 4), 
         index + 1, symbol)
        for index, symbol in zip( list(range(len(input_structure_ase))), 
                                 input_structure_ase.get_chemical_symbols() 
                                 )
    ]  ## Get all distances between the selected atom and all other atoms
    distances = sorted(distances, key=lambda tup: tup[0]) ## Sort the distances shortest->longest
    
    if distorted_element : 
        nearest = []
        ## filter the neighbours that match the element criteria and are closer than 4.5 A
        for i in distances[1:]:
            if len(nearest) < (neighbours - 1) and i[2] == distorted_element and i[0] < 4.5 : 
                    nearest.append(i)
            elif len(nearest) == (neighbours - 1):
                break
        ## if the number of nearest neighbours not reached, add other neighbouring elements 
        [nearest.append(i) for i in distances[1:] if len(nearest) < (neighbours - 1) and i not in nearest and i[0]< 4.5] 
    else:        
        nearest = distances[1:neighbours]  ## Extract the nearest neighbours according to distance
    distorted = [
        (i[0] * distortion_factor, i[1], i[2]) for i in nearest
    ]  ## Distort the nearest neighbour distances according to sf
    [
        input_structure_ase.set_distance(atom_number, i[1] - 1, i[0], fix=0, mic=True)
        for i in distorted
    ]  ## Modify the ase.Atoms object for the distorted case
    
    if isinstance(frac_coords, np.ndarray):
        input_structure_ase.pop(-1) # remove fake V
    
    distorted_structure = aaa.get_structure(input_structure_ase)
    distorted_atoms =  [(i[1], i[2]) for i in nearest ] # element and site number
    bdm_distorted_defect = {"distorted_structure": distorted_structure,
                            "defect_index": atom_index or None,
                            "number_distorted_neighbours": num_nearest_neighbours,
                            "distorted_atoms": distorted_atoms }
    if verbose:
        # print(f"\n Distortion factor {distortion_factor}")
        # print("Number of distorted neighbours:", num_nearest_neighbours)
        # print("Input Structure Formula:", structure.formula)
        distorted = [ (round(i[0],2), i[1], i[2]) for i in distorted ] ; nearest = [ (round(i[0],2), i[1], i[2]) for i in nearest ] # round numbers 
        print("     Defect Atom Index/frac. coords:", atom_index or frac_coords )
        print("     Original Neigbhour Distances:", nearest)
        print("     Distorted Neighbour Distances:", distorted) ; print() 

    return bdm_distorted_defect 


def rattle(
        structure, 
           stdev=0.25
           ):
    """Given a pymnatgen structure, it applies a random distortion to the coordinates of all atoms. 
    Random distortion chosen from a gaussian with a standard deviation of stdev.
    Args:
        structure : pymatgen structure
        stdev (float): 
            standard dev of the gaussian used for rattle (in A)
            (default: 0.25)
    Returns:
        rattled structure"""
    aaa = AseAtomsAdaptor()
    ase_struct = aaa.get_atoms(structure)
    ase_struct.rattle(stdev=stdev)
    rattled_structure = aaa.get_structure(ase_struct)
    
    return rattled_structure

def localized_rattle(
        structure, 
        defect_coords,
        stdev=0.25,
        ):
    """Given a pymnatgen structure, it applies a random distortion to the coordinates of the atoms in a radius 5 A from defect atom. 
    Random distortion chosen from a gaussian with a standard deviation of stdev.
    Args:
        structure : 
            pymatgen structure
        defect_coords (np.array):
            cartesian coordinates of defect
        stdev (float): 
            standard dev of the gaussian used for rattle (in A)
            (default: 0.25)
    Returns:
        rattled structure"""
    from pymatgen.core.structure import Structure
    from pymatgen.core.sites import Site
    
    aaa = AseAtomsAdaptor()
    structure_copy = structure.copy()

    # Classify sites in 2 lists: inside or outside 5 A sphere
    sites_inside_cutoff, sites_outside_cutoff = [], []
    for site in structure_copy:
        distance, image = site.distance_and_image_from_frac_coords(defect_coords)[:2]
        if distance < 5:
            sites_inside_cutoff.append(site)
        else:
            sites_outside_cutoff.append(site)
    
    # Apply rattle to sites within 5 A sphere
    structure_inside_cutoff = structure_copy.from_sites(sites_inside_cutoff)
    ase_struct = aaa.get_atoms(structure_inside_cutoff)
    ase_struct.rattle(stdev=stdev)
    rattled_structure = aaa.get_structure(ase_struct)
    
    # Add the sites outside the 5 A sphere to the rattled structure
    [ rattled_structure.append(site_outside_cutoff.specie, site_outside_cutoff.frac_coords) for site_outside_cutoff in sites_outside_cutoff ]
    
    return rattled_structure

def update_struct_defect_dict(defect_dict,
                              structure, 
                              poscar_comment: str):
    """Given a structure and poscar comment updates the dictionary of folders (generated with prepare_vasp_defect_inputs()) with the given values.
    Args: 
        defect_dict: as generated with doped prepare_vasp_defect_inputs()
        structure: defect structure in pymatgen format
        poscar_comment: comment to include in POSCAR
    Returns:
        single defect dict in doped format, ready for submission"""
    defect_dict_copy = deepcopy(defect_dict)
    defect_dict_copy["Defect Structure"] = structure
    defect_dict_copy['POSCAR Comment'] = poscar_comment
    
    return defect_dict_copy

def calc_number_electrons(defect_dict: dict, 
                          valences_oxi_states: dict):
    """ Given the defect_dict and valences_oxi_states, calculates the number of extra/missing e- of defect based on oxidation states.
    Args:
        defect_dict (dict): 
            Defect dictionary in doped.pycdt.core.defectsmaker.ChargedDefectsStructures() format,
        valences_oxi_states (dict): 
            Dictionary with valences and oxidation states of the atoms in the material 
            (e.g. {"valences":   {"Cd": 2, "Te": 6},
                   "oxi_states": {"Cd": +2, "Te": -2} 
                   } 
             ).
    Returns: 
        number of extra/missing e-. Negative number if defect has extra e-, positive if missing e-.
        """
    #valences = valences_oxi_states['valences']
    oxi_states = valences_oxi_states['oxi_states']
    
    oxi_states['Vac']=0 ; #valences['Vac'] = 0
    
    # determine number of extra/missing e- based on defect type and oxidation states
    if defect_dict['defect_type'] == 'vacancy':
        site_specie = str(defect_dict['site_specie'])
        substituting_specie = 'Vac'
        #num_electrons = - ( oxi_states[substituting_specie] - oxi_states[site_specie] ) 
                
    elif defect_dict['defect_type'] == 'interstitial':
        substituting_specie = str(defect_dict['site_specie'])
        site_specie = 'Vac' # Consider an interstitial as the interstitial atom substituting a vacant position
        #num_electrons = - ( oxi_states[substituting_specie] - oxi_states[site_specie] ) 
    
    elif defect_dict['defect_type'] == 'antisite':
        site_specie = str(defect_dict['site_specie'])
        substituting_specie = defect_dict['substituting_specie']
        #num_electrons = valences[substituting_specie] - valences[site_specie] 
        
    elif defect_dict['defect_type'] == 'substitution':
        site_specie = str(defect_dict['site_specie'])
        substituting_specie = str(defect_dict['substitution_specie'])
        #num_electrons = valences[substituting_specie] - valences[site_specie] 
        
    num_electrons = - ( oxi_states[substituting_specie] - oxi_states[site_specie] ) 
    #print(f"Number of missing/extra electrons of defect {defect['name']}: ", num_electrons)   
    return int(num_electrons)

def calc_number_neighbours(num_electrons: int):
    """Calculates the number of neighbours to distort by considering
    the number of extra/missing electrons.
    Args:
        num_electrons (int): number of missing/extra electrons in defect
    Returns:
        number of neighbours to distort"""
    if num_electrons < -4 or num_electrons > 4 : 
        # if number of missing/extra e- higher than 4, then distort 8-num_electrons
        num_neighbours = abs(8 - abs(num_electrons) )
    elif -4 < num_electrons < 4:
        num_neighbours = abs(num_electrons)
    elif abs(num_electrons) == 4:
        num_neighbours = abs(num_electrons)
        
    return abs(num_neighbours)

def apply_rattle_BDM(defect_dict: dict, 
                     num_nearest_neighbours: int, 
                     distortion_factor: float, 
                     std_dev: float = 0.25,
                     distorted_element: str = None,
                     verbose: bool = False,
                     ):
    """ Applies rattle + BDM to the ideal defect structure. It calls BDM with either:
            - fractional coordinates (for vacancies) or 
            - atom site (other defect types).
        Args:
            defect_dict (dict): 
                defect dictionary in doped format
            num_nearest_neighbours (int):
                number of defect neighbours to distort
            distortion_factor (float): 
                distortion factor applied to defect neighbours
            std_dev (float): 
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
    if defect_dict['defect_type'] == 'vacancy': # for vacancies, we need to use fractional coordinates (no atom site!)
        bdm_distorted_defect  = bdm(
                    defect_dict["supercell"]["structure"],
                    num_nearest_neighbours,
                    distortion_factor,
                    frac_coords = defect_dict["bulk_supercell_site"].frac_coords,
                    distorted_element = distorted_element,
                    verbose = verbose,
                )
        bdm_distorted_defect["distorted_structure"] = rattle(bdm_distorted_defect["distorted_structure"] , 
                                                             std_dev )  
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
        bdm_distorted_defect["distorted_structure"] = rattle(bdm_distorted_defect["distorted_structure"], 
                                                             std_dev) 
    return bdm_distorted_defect

def apply_distortions(defect_dict: dict, 
                      num_nearest_neighbours: int, 
                      bdm_distortions: list,
                      std_dev: float = 0.25,
                      distorted_element: str = None,
                      verbose: bool = False,
                      ):
    """ Applies rattle (+ BDM) to defect structure depending on the number of neighbours.
    0 neighbours : only rattle; any other case: rattle + BDM.
    For the rattle + BDM distortion, it creates a dictionary with the distorted structures by bdm_distortions. 
    Returns a dictionary that maps distotion to umperturbed and distorted structures
    For rattling only (no electron change), returns a dictionary with rattled and non-distorted structure.
    Args:
        defect_dict (dict): 
            Defect dictionary as generated with doped.
        num_nearest_neighbours (int): 
            Number of nearest neighbours to distort
        bdm_distortions (list):
            List of distortions to apply to nearest neighbours. (e.g. [-0.5, 0.5])
        std_dev (float): 
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
                                                    std_dev = std_dev,
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
                                     std_dev) 
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
        
def create_vasp_input(defect_name: str, 
                      charged_defect: dict, 
                      incar_settings: dict,        
                      potcar_settings: dict = None,
                      bdm_type: str= "BDM"):
    """Creates folders for storing BDM files. 
    Args:
        defect_name (str): 
            folder name
        bdm_type (str): 
            Type of distortion method. Either 'BDM' (standard method) or 
            'champion' (if the ground-state distortion found for other charge state is tried for the remaining charge states).
                (default: 'BDM')
        incar_settings (dict): 
            incar settings to update doped default ones
        """
    create_folder(defect_name) # create folder for defect
    create_folder(defect_name+"/"+bdm_type) # either folder (named BDM or champion) where BDM distortions will be written.
    for key, defect_dict in charged_defect.items(): # for each distortion, create its folder  
        potcar_settings_deepcopy = deepcopy(potcar_settings) # vasp_gam_files empties the potcar_settings dict, so make a deepcopy
        vasp_input.vasp_gam_files(defect_dict, 
                                  input_dir = f"{defect_name}/{bdm_type}/{defect_name}_{key}", 
                                  incar_settings = incar_settings,
                                  potcar_settings = potcar_settings_deepcopy
                                  ) 

def apply_RBDM_defect_dict(defects_dict: dict, 
                      valences_oxi_states: dict, 
                      incar_settings: dict = default_incar_settings,                 
                      dict_number_electrons_user : dict = None,
                      bdm_increment: float = 0.1,
                      bdm_distortions: list = None,
                      std_dev: float = 0.25,
                      distorted_elements: dict = None,
                      bdm_type: str= "BDM",
                      potcar_settings: dict = None,
                      write_files: bool = True,
                      verbose: bool = False,
                      ):
    """ Applies rattle+BDM to all defects in your input defect dictionary (in doped format).
    It creates the input files for the vasp_gam relaxations of all RBDM distortions.
    It also creates a dictionary with an entry for every defect, which in turn links to a dictionary with 
    all BDM distortions (and undistorted) for each charge state of the defect (in case wanna check something).
    Args:
        defects_dict (dict):
            Defect dictionary as generated with doped ChargedDefectsStructures()
        valences_oxi_states (dict):
            chemical valences and oxidation states of your material, used to determine the number of neighbours to distort
            (e.g {"valences": {"Cd": 2, "Te": 6},
                  "oxi_states": {"Cd": +2, "Te": -2} )
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are.
        dict_number_electrons_user (dict):
            Optional argument to set the number of extra/missing e- for the input defects.
            Dictionary with format { defect_name : number_of_electrons } where number_of_electrons is negative for extra e- and positive for missing e-.
            (default: None)
        bdm_increment (float):
            Distortion increment for BDM. Recommended values: 0.1-0.25
            (default: 0.1)
        bdm_distortions (list):
            List of distortions to apply to nearest neighbours instead of default ones. (e.g. [-0.5, 0.5])
            (default: None)
        std_dev (float):
            Standard deviation of rattle function that will be aplied to defect structures.
            Recommended values: 0.15 or 0.25. 
            (default: 0.25),
        distorted_elements (dict): 
            Allows the user to specify the element to distort for each defect. 
            Dictionary mapping defect name (without charge state) to chemical symbol (e.g {'vac_1_Cd': 'Te'})
            (default: None)
        bdm_type (str) :
            Leave it as default ("BDM") for standard method. 
            The option "champion" is used when relaxing a defect from the ground state configuration found for other charge states. In this case, only the Unperturbed
            and rattled configurations are relaxed.
            (default: "BDM" (normal BDM is assumed))
        potcar_settings (dict): 
            Dictionary of element-potcar name (e.g { "POTCAR" : {"Ti":"Ti_pv"} } to override default potcar (recommended to check default POTCAR's in doped default_POTCARs.yalm').
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
    if not bdm_distortions:
        bdm_distortions = list(np.around(np.arange(-0.6, 0.601, bdm_increment), decimals=3)) #[i/100 for i in range(-60, 60.1, int(bdm_increment*100) ]
    bdm_metadata = {"BDM_parameters": {"BDM_increment": bdm_increment,
                                       "BDM_distortions": bdm_distortions,
                                       "rattle_std_dev": std_dev},
                    "defects": {},
                    } # dict with BDM parameters, useful for posterior analysis
    
    print(f"Applying BDM... Will rattle with std dev of {std_dev} A \n")
    
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
            else: 
                distorted_element = None
            
            if not dict_number_electrons_user : # if not given, BDM will calculate the number of extra/missing e- of defect
                number_electrons = calc_number_electrons(defect, valences_oxi_states) 
            else:
                number_electrons = dict_number_electrons_user[defect_name]
                
            dict_defects[defect_name] = {}
            bdm_metadata["defects"][defect_name] = {"unique_site" :  list(bulk_supercell_site.frac_coords), 
                                                    "charges": {},
                                                    }
            
            print("\nDefect:", defect_name)
            if number_electrons < 0 :
                print(f"Number of extra electrons in neutral state: {number_electrons}")
            elif number_electrons >= 0 :
                print(f"Number of missing electrons in neutral state: {number_electrons}")
                
            for charge in defect["charges"]:
                
                poscar_comment = vasp_defect_inputs[f"{defect_name}_{charge}"]["POSCAR Comment"]
                charged_defect = {}
    
                # Entry for the unperturbed defect to compare
                charged_defect["Unperturbed_Defect"] = deepcopy(vasp_defect_inputs[f"{defect_name}_{charge}"])
                                                                                
                # Generate perturbed structures
                # Calculate extra/missing e- accounting for the charge state of the defect 
                num_electrons_charged_defect = number_electrons + charge # negative if extra e-, positive if missing e-
                num_nearest_neighbours = calc_number_neighbours(num_electrons_charged_defect) # Number of distorted neighbours for each charge state
                
                
                print(f"\nDefect {defect_name} in charge state: {charge}. Number of distorted neighbours: {num_nearest_neighbours}")
                distorted_structures = apply_distortions(defect, 
                                                         num_nearest_neighbours, 
                                                         bdm_distortions,
                                                         std_dev,
                                                         distorted_element,
                                                         verbose = verbose,
                                                         )
                bdm_metadata["defects"][defect_name]["defect_index"] = distorted_structures["BDM_parameters"]["defect_index"] # store site number of defect
                bdm_metadata["defects"][defect_name]["charges"].update({int(charge): 
                                                                            {"number_neighbours": num_nearest_neighbours,
                                                                             "distorted_atoms" : distorted_structures["BDM_parameters"]["distorted_atoms"],
                                                                             } 
                                                                        } 
                                                                       ) # store BDM parameters used for latter analysis
                
                
                for key_distortion, struct in distorted_structures["Distortions"].items():
                    poscar_comment = key_distortion.split("Distortion")[0] + "_" + vasp_defect_inputs[f"{defect_name}_{charge}"]["POSCAR Comment"] + "__num_neighbours=" + str(num_nearest_neighbours)
                    charged_defect[key_distortion] = update_struct_defect_dict(vasp_defect_inputs[f"{defect_name}_{charge}"],
                                                                               struct,  
                                                                               poscar_comment,
                                                                               )
     
                dict_defects[defect_name][f"{defect_name}_{charge}"] = charged_defect # add charged defect entry to dict
                if write_files :
                    create_vasp_input( f"{defect_name}_{charge}", 
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


################################################################################
# Try distortions that lead to ground-states for the other charge states, in case it's
# also a favourable distortion for them.
# work in progresss
################################################################################

def get_deep_distortions(defect_charges: dict, 
                     bdm_type: str='BDM',
                     stol = 0.2,
                     ):
    """Quick and nasty function to easily spot defects undergoing a deep relaxation.
    Useful for trying the ground-state of a certain charge state for the other charge states.
    Args:
        defect_charges (dict): 
            dictionary matching defect name to a list with its charge states. (e.g {"Int_Sb_0":[0,+1,+2]} )
        base_path (str): 
            path to the directory where your BDM output files are (will need CONTCAR to check structural similarity of the final configurations)
        bdm_type (str):
            Allows to search in the 'BDM' (normal method) or 'champion' folders (when trying the energy-lowering distortion found for other charge state).
   Returns:
        fancy_defects (dict): 
            dictionary of defects for which BDM found a deep distortion (missed with normal relaxation).
            The charge state with the distortion associated to the greatest E drop is selected 
        """
    fancy_defects = {} #dict of defects undergoing deep distortions
    sm = StructureMatcher(ltol=0.2, stol=stol)
    for defect in defect_charges.keys():
        print("\n",defect)
        for charge in defect_charges[defect]:
            defect_name = "{}_{}".format(defect, str(charge)) #defect + "_" + str(charge)
            file_energies = "{}{}/{}/{}.txt".format(base_path, defect_name, bdm_type ,defect_name ) 
            dict_energies, energy_diff, gs_distortion = sort_data(file_energies)
            
            if float(energy_diff) < -0.1 : #if a significant E drop occured , then store this fancy defect
                print("Deep distortion found for ", defect_name)    
                if  gs_distortion != "rattle":
                    bdm_distortion = str(round(gs_distortion * 100, 1)) #change distortion format to the one used in file name (eg from 0.1 to 10.0)
                    if bdm_distortion == "0.0":
                        bdm_distortion = "-0.0"
                    file_path="{}{}/{}/{}_{}%_BDM_Distortion/vasp_gam/CONTCAR".format(base_path, defect_name, bdm_type ,defect_name, bdm_distortion) 
                else:
                    bdm_distortion = "only_rattled" # file naming format used for rattle
                    file_path="{}{}/{}/{}_{}/vasp_gam/CONTCAR".format(base_path, defect_name, bdm_type ,defect_name, bdm_distortion) 
                try:
                    gs_struct = grab_contcar(file_path) # get the final structure of the E lowering distortion
                    if gs_struct  == "Not converged":
                        print(f"Problem grabbing gs structure for {bdm_distortion} of {defect_name}")
                except FileNotFoundError:
                    print("NO CONTCAR for ground-state distortion")
                    break
                if defect in fancy_defects.keys(): #check if defect already in dict (other charge state lead to a lower E structure)
                    
                    gs_struct_in_dict = fancy_defects[defect]["structure"]                        
                  
                    if energy_diff < fancy_defects[defect]["energy_diff"]: #if E drop is greater (more negative), then update the dict with the lowest E distortion
                        print("Charge {} lead to greatest E lowering distortion".format(charge))
                        fancy_defects[defect].update(
                            {"structure": gs_struct, "BDM_distortion": gs_distortion,"energy_diff": energy_diff, "charges":[charge]}
                            )   
                            
                elif defect not in fancy_defects.keys(): # if defect not in dict, add it
                    print("New defect! Adding {} with charge {} to dict".format(defect, charge))
                    fancy_defects[defect] = {"charges" : [charge], "structure": gs_struct, "energy_diff": energy_diff, "BDM_distortion": gs_distortion}
        
        #let's check that the gs structure wasn`t found already by BDM for the other charge states    
        if defect in fancy_defects.keys(): # if the defect lead to an E lowering distortion
            for charge in defect_charges[defect]: # for all charge states of the defect
                if charge not in fancy_defects[defect]["charges"]: #if gs struct wasn't found already for that charge state
                    defect_name = "{}_{}".format(defect, str(charge)) #defect + "_" + str(charge)
                    gs_struct_in_dict = fancy_defects[defect]["structure"] 
                    if compare_gs_struct_to_BDM_structs( gs_struct_in_dict, defect_name, base_path, stol = stol  ) : 
                        # structure found in BDM calcs for this charge state. Add it to the list to avoid redundant work
                        fancy_defects[defect]["charges"].append(charge)
            #print("Ground-state structure found for {}_{} has been also found for the charge states: {}".format(defect, fancy_defects[defect]["charges"][0], fancy_defects[defect]["charges"] ))
    return fancy_defects  

def compare_gs_struct_to_BDM_structs(gs_contcar, 
                                     defect_name: str, 
                                     base_path: str,
                                     stol: float = 0.2,
                                     ):
    """Compares the ground-state structure found for a certain charge state with all BDM structures found for other charge states
    to avoid trying the ground-state distortion when it has already been found.
    Args: 
        gs_contcar (structure): 
            structure of ground-state distortion
        defect_name (str): 
            name of defect including charge state (e.g. "vac_1_Sb_0")
        base_path: (str): 
            path to the directory where your BDM defect output is (will need CONTCAR)
    Returns:
        True if energy lowering distortion was found for the considered charge state, False otherwise
    """
    sm = StructureMatcher(ltol=0.2, stol=stol)
    defect_structures = get_structures(defect_name, base_path )
    for key, structure in defect_structures.items():
        if gs_contcar == "Not converged" :
            print("gs_contcar not converged")
            break
        elif structure != "Not converged" :
            try:
                if sm.fit(gs_contcar, structure):
                    return True #as soon as you find the structure, return True
            except AttributeError:
                print("Fucking error grabbing structure")
        else:
            print("{} Structure not converged".format(key))
    return False # structure not found for this charge state
            
def create_dict_deep_distortion_old(defect_dict: dict, 
                                fancy_defects: dict,
                                ):
    """Imports the ground-state distortion found for a certain charge state in order to
    try it for the other charge states.
    Args:
        defect_dict (dict): 
            defect dict in doped format
        fancy_defects (dict): 
            dict containing the defects for which we found E lowering distortions
        """
    dict_deep_distortion  = {}
    defect_dict_copy = defect_dict.copy()
    for defect_type in fancy_defects.keys(): # for each defect type (vac, as , int)
        
        dict_deep_distortion[defect_type]  = import_deep_distortion_by_type(defect_dict_copy[defect_type],
                                                                            fancy_defects[defect_type]) #defects for which we'll try the deep distortion found for one of the charge states      
    return  dict_deep_distortion

def import_deep_distortion_by_type(defect_list: list,
                                   fancy_defects: dict,
                                   ):
    """Imports the ground-state distortion found for a certain charge state in order to
    try it for the other charge states.
    Args:
        defect_list (list): 
            defect dict in doped format of same type (vacancies, antisites or interstitials)
        fancy_defects (dict): 
            dict containing the defects for which we found E lowering distortions (vacancies, or antisites or interstitials)
         """
    list_deep_distortion = []
    for i in defect_list: 
                if i['name'] in fancy_defects.keys(): # if defect underwent a deep distortion
                    defect_name = i['name']
                    print(defect_name)
                    i['supercell']['structure'] = fancy_defects[defect_name]['structure'] #structure of E lowering distortion
                    print("Using the distortion found for charge state(s) {} with BDM distortion {}".format(fancy_defects[defect_name]['charges'],
                                                                                                            fancy_defects[defect_name]["BDM_distortion"] ) )
                    #remove the charge state of the E lowering distortion distortion
                    if len(fancy_defects[i['name']]['charges']) > 1:
                        print("Intial charge states of defect:", i['charges'], "Will remove the ones where the distortion was found...")
                        [i['charges'].remove(charge) for charge in fancy_defects[defect_name]['charges']]
                        print("Trying distortion for charge states:", i['charges'])
                    else:   
                        i['charges'].remove(fancy_defects[defect_name]['charges'][0])
                    if i['charges']: #if list of charges to try deep distortion not empty, then add defect to the list
                        list_deep_distortion.append(i)
    return list_deep_distortion

