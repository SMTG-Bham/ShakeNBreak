"""
Quick and _dirty_ collection of useful functions to analyse BDM defect output

@author: Irea Mosquera
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import json
from IPython.display import display
from copy import deepcopy

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import CrystalNN
sm = StructureMatcher(ltol=0.2, stol=0.1) #The stol default (0.3) is a bit large for defect structural comparison
crystalNN = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False, search_cutoff=5)

from ase.io.vasp import read_vasp

#############################################################################################

def open_file(
    myfile: str
    )-> list:
    """ Open file and split lines"""
    if os.path.isfile(myfile):
        with open(myfile) as ff:
            read_file= ff.read()
            list_file = read_file.splitlines()
        return list_file  
    else:
        print(f"Path {myfile} does not exist")

def organize_data(
    list_file: list
    ) -> dict:
    """
    Creates dictionary maping distortion to final E. 
    Args:
        list_file (list): 
            list of lines in BDM output summary file (which includes BDM distortion and its final E)
    """
    energies_dict = {'distortions': {} }   
    for i in range(len(list_file)//2):
        i = 2*i
        if 'rattle' in list_file[i]:
            key = 'rattled'
            energies_dict['distortions'][key] = float(list_file[i+1])
        else:
            if 'Unperturbed' in list_file[i]:
                energies_dict['Unperturbed'] =  float(list_file[i+1])
            else:
                key = list_file[i].split("_BDM")[0].split("%")[0]
                key = float(key.split('_')[-1]) /100  # from % to decimal
                if key == -0.0:
                    key = 0.0
                energies_dict['distortions'][key] = float(list_file[i+1])
    sorted_dict = {'distortions': {} , 
                   'Unperturbed': energies_dict['Unperturbed'] }   
    for key in sorted(energies_dict['distortions'].keys()): # Order dict items by key (from -0.6 to 0 to +0.6)
        sorted_dict['distortions'][key] = energies_dict['distortions'][key]
    return sorted_dict

def get_gs_distortion(
    dict_energies: dict
    ):
    """
    Calculates energy difference between Unperturbed structure and most favourable distortion.
    Returns energy drop of the ground-state relative to Unperturbed (in eV) and the BDM distortion that lead to ground-state.
    Args:
        dict_energies (dict): 
            Dictionary matching distortion to final energy, as produced by organize_data()
    Returns:
        (energy_difference, BDM_ground_state_distortion)
        """
    if len(dict_energies['distortions']) == 1:
        energy_diff = dict_energies['distortions']['rattled'] - dict_energies['Unperturbed']
        if energy_diff < 0 :
            gs_distortion = 'rattled' #just rattle (no BDM)
        else:
            gs_distortion = "Unperturbed"
    else:
        lowest_E_RBDM = min(dict_energies['distortions'].values()) #lowest E obtained with RBDM
        energy_diff = lowest_E_RBDM - dict_energies['Unperturbed']
        if lowest_E_RBDM < dict_energies['Unperturbed'] : #if energy lower that with Unperturbed
            gs_distortion = list(dict_energies['distortions'].keys())[list(dict_energies['distortions'].values()).index( lowest_E_RBDM )] #BDM distortion that lead to ground-state
        else:
            gs_distortion = "Unperturbed"
       
    return energy_diff, gs_distortion
        
def sort_data(
    file_energies: str
    ):
    """ 
    Organices BDM results in a dictionary, gets energy drop (of ground-state relative to Unperturbed, in eV) 
    and the BDM distortion that lead to ground-state.
    Args:
        file_energies : txt file with BDM distortions and final energies (in eV)
    Returns:
        dict_energies: 
            dictionary with BDM distortion and final energy
        energy_diff: 
            energy difference between minimum energy obtained with BDM and Unperturbed relaxation
        gs_distortion: 
            distortion that lead to the minimum energy structure
    """
    dict_energies = organize_data( open_file(file_energies) )
    energy_diff, gs_distortion = get_gs_distortion(dict_energies)
    if energy_diff < -0.1 :
        defect_name = file_energies.split("/")[-1].split(".txt")[0]
        print("{} : E diff. between minimum found with {} RBDM and unperturbed:".format(defect_name, gs_distortion), "{:+.2f}".format(energy_diff), "eV \n")
    return dict_energies, energy_diff, gs_distortion


#############################################################################################

def grab_contcar(
    file_path: str
    ) -> Structure:
    """
    Read pmg structure from CONTCAR path
    Args:
        file_path (str):
            path file to CONTCAR
    Returns:
        Structure"""
    abs_path = file_path
    abs_path_formated = abs_path.replace('\\','/') 
    #assert os.path.isfile(abs_path_formated)
    if not os.path.isfile(abs_path_formated):
        print("CONTCAR file doesn't exist. Check path & relaxation")
    try :
        mystruct = Structure.from_file(abs_path_formated)
    except ( FileNotFoundError or IndexError or ValueError ):
        mystruct = "Not converged"
        print(f"Path to structure {abs_path_formated}")
    except:
        print("Problem grabbing structure from CONTCAR.")
        print(f"Path to structure {abs_path_formated}")
        mystruct = "Not converged"
    return mystruct

def analyse_structure_from_site(
    name: str, 
    structure: Structure, 
    site_num: int = None,
    vac_site: list = None,
    ):
    """ 
    Analyse coordination environment and bond distances to nearest neighbours of defect 
    Args:  
        name (str): 
            name of defect
        structure (str): 
            pmg structure to analyse
        site_num (int): 
            number of defect site in structure, starting from 1
        vac_site (list): 
            for vacancies, the fractional coordinates of the vacanct site.
            (default: None)
    """
    #get defect site
    struct = deepcopy(structure)
    if site_num:
        isite = site_num  - 1 # Starts counting from zero!
    elif vac_site :
        struct.append('U', vac_site) # Have to add a fake element
        isite = len(struct.sites) - 1 # Starts counting from zero!
    
    print("==> ", name + " structural analysis "," <==")    
    print("Analysing site", struct[isite].specie, struct[isite].frac_coords)
    coordination = crystalNN.get_local_order_parameters(struct, isite)
    if coordination != None:
        list_coord = []
        for coord, value in coordination.items():
            mydict={}
            mydict['Coordination']=coord
            mydict['Factor'] = value
            list_coord.append(mydict)
        print("Local order parameters (i.e. resemblence to given structural motif): ")
        display(pd.DataFrame(list_coord))
    # Bond Lengths?
    bond_lengths = []
    for i in crystalNN.get_nn_info(struct,isite):
        bond_lengths.append({'Element': i['site'].specie.as_dict()['element'],
        'Distance': f"{i['site'].distance(struct[isite]):.3f}"})
    df = pd.DataFrame(bond_lengths)
    print("\nBond-lengths (in A) to nearest neighbours: ")
    #print(df, "\n")
    display(df)
    print()
    #return crystalNN.get_local_order_parameters(struct,isite)
    #return bond_lengths

def analyse_structure(
    defect_name: str, 
    structure: Structure, 
    output_path: str,
    ):
    """
    Analyses the local distortion of a defect from its pmg Structure. 
    Requires access to BDM_metadata file generated with BDM to read info about defect site.
    If lacking this, can use `analyse_structure_from_site`.
    Args:
        defect_name (str): 
            name of defect (with charge) (i.e vac_1_Cd_0),
        structure: 
            defect structure to analyse
        output_path (str):
            path where output and BDM_metadata.json are located
            """
    charge = defect_name.rsplit("_")[-1]
    defect_name_without_charge = defect_name.rsplit("_",1)[0]
    
    # Read site from BDM_metadata.json
    with open(f"{output_path}/BDM_metadata.json") as json_file:
        BDM_metadata = json.load(json_file)
    
    defect_site = BDM_metadata["defects"][defect_name_without_charge]["defect_index"] # starts counting from 1 
    if not defect_site : # for vacancies, get fractional coords
        defect_frac_coords = BDM_metadata["defects"][defect_name_without_charge]["unique_site"]
        return analyse_structure_from_site(defect_name, structure, vac_site = defect_frac_coords)
    
    return analyse_structure_from_site(defect_name, structure, site_num = defect_site)

def compare_structures(
    defect_dict: dict,
    defect_energies: dict,
    compare_to: str = "Unperturbed", 
    norm_structure = None, 
    stol: float = 0.5,
    units: str = "eV",
    ) -> pd.DataFrame:
    """ Compares BDM final structures with either Unperturbed or an external structure and calculated the root mean squared displacement and maximum distance between paired sites between them.
    Args:
        defect_dict (dict):
            dictionary maping BDM distortion to structure
        defect_energies (dict):
            dict maping BDM distortion to final E (eV)
        compare_to (str): 
            name of reference/norm structure used for comparison (recommended to compared with Unperturbed)
            (default: "Unperturbed")
        norm_structure: 
            structure used as reference/norm structure. 
            This allows to compare final BDM structure with an external structure (not obtained with BDM distortions).
            (default: None)
        units (str):
            Either eV or meV used when reporting energies
        stol (float):
            site tolerance used for structural comparison
            (default: 0.5 A)
       """
    print(f"Comparing structures to {compare_to}...")
    
    rms_list = []
    if norm_structure: #if we wanna give it a external structure (not obtained with BDM distortions)
        norm_struct = norm_structure
    else : #else we get reference structure from defect dictionary 
        norm_struct = defect_dict[compare_to]
    assert norm_struct

    distortion_list = list(defect_energies['distortions'].keys())
    distortion_list.append("Unperturbed")
    for distortion in distortion_list:
        if distortion == "Unperturbed":
            rel_energy = defect_energies[distortion]
        else :
            try:
                rel_energy = defect_energies['distortions'][distortion]
            except KeyError: # if relaxation didn't converge for this BDM distortion, store it as NotANumber
                rel_energy = float("nan")
        struct = defect_dict[distortion]
        if struct and struct != "Not converged" and norm_struct and norm_struct != "Not converged":                          
            sm = StructureMatcher(ltol=0.3, stol=stol, angle_tol=5, primitive_cell=False, scale=True) #higher stol for calculating rms
            try:
                rms_displacement = round(sm.get_rms_dist(norm_struct, struct)[0], 3)
                rms_dist_sites = round(sm.get_rms_dist(norm_struct, struct)[1], 3) # select rms displacement normalized by (Vol / nsites) ** (1/3)
            except TypeError: # lattices didn't match
                rms_displacement = None
                rms_dist_sites = None
            rms_list.append([distortion, rms_displacement, rms_dist_sites, round(rel_energy, 2) ] )
    display(pd.DataFrame(rms_list, columns=["BDM Dist.", "rms", "max. dist (A)", f"Rel. E ({units})"]) )
    return pd.DataFrame(rms_list, columns=["BDM Dist.", "rms", "max. dist (A)", f"Rel. E ({units})"])
        
############################################################################
    
def get_structures(
    defect_name: str, 
    output_path: str, 
    bdm_increment: float=0.1,
    bdm_distortions: list = None,
    bdm_type="BDM",
    ) -> dict:
    """Imports all the structures found with BDM and stores them in a dictionary matching BDM distortion to final structure.
    Args:
        defect_name (str) : 
            name of defect (e.g "vac_1_Sb_0")
        output_path (str) : 
            path where material folder is
        bdm_increment (float):
            Distortion increment for BDM. 
            (default: 0.1)
        bdm_distortions (list):
            List of distortions applied to nearest neighbours instead of default ones. (e.g. [-0.5, 0.5])
            (default: None)
        bdm_type (str): 
            BDM or champion
            (default: BDM)
    Returns:
        dictionary mathing BDM distortion to final structure"""
    defect_structures = {}
    try:
        # Read BDM_parameters from BDM_metadata.json
        with open(f"{output_path}/BDM_metadata.json") as json_file:
            bdm_parameters = json.load(json_file)['BDM_parameters']
            bdm_distortions = bdm_parameters['BDM_distortions']
            bdm_distortions = [i*100 for i in bdm_distortions]
    except: # if there's not a BDM metadata file       
        if bdm_distortions: 
            bdm_distortions = [i*100 for i in bdm_distortions]
        else:
            bdm_distortions = range(-60, 70, bdm_increment*100) # if user didn't specify BDM distortions
        
    rattle_dir_path = output_path  + "/"+ defect_name + "/" + bdm_type + "/" + defect_name +"_" + "only_rattled"
    if os.path.isdir(rattle_dir_path): #check if rattle folder exists (if so, it means we only applied rattle (no BDM as 0 change in electrons), 
                                       # hence grab the rattle & Unperturbed, not BDM distortions)
        try:
            path= rattle_dir_path + "/vasp_gam/CONTCAR"
            defect_structures['rattle'] = grab_contcar(path)
        except:
            print("Problems in get_structures")
            defect_structures['rattle'] = "Not converged"
    else:
        for i in bdm_distortions:
            key = i / 100 #key used in dictionary. Using the same format as the one in dictionary that matches distortion to final energy
            i = '{:.1f}'.format(i)
            if i == "0.0": 
                i = "-0.0" #this is the format used in defect file name
            path = output_path  + "/"+ defect_name + "/" + bdm_type + "/" + defect_name +"_" + str(i) + "%_BDM_Distortion/vasp_gam/CONTCAR"
            try :
                defect_structures[key] = grab_contcar(path)
            except FileNotFoundError or IndexError or ValueError:
                print("Error grabbing structure.")
                print("Your defect path is: ", path)
                defect_structures[key] = "Not converged"
            except:
                print("Problem in get_structures")
                print("Your defect path is: ", path)
                defect_structures[key] = "Not converged"
    try:
        defect_structures["Unperturbed"] = grab_contcar(output_path + "/"+ defect_name + "/" + bdm_type + "/" + defect_name +"_" +  "Unperturbed_Defect" + "/vasp_gam/CONTCAR")
    except FileNotFoundError:
        print("Your defect path is: ", path)
        defect_structures[key] = "Not converged"
    return defect_structures

def get_energies(
    defect_name: str, 
    output_path: str, 
    bdm_type: str ="BDM",
    units: str = "eV",
    ) -> dict:
    """Imports final energies for each BDM distortion and stores them in a dictionary matching BDM distortion to final E (eV).
    Args:
        defect_name (str) : 
            name of defect (e.g "vac_1_Sb_0")
        output_path (str) : 
            path where material folder is
        bdm_type (str): 
            BDM or champion
    Returns:
        dictionary mathing BDM distortion to final E"""
    dict_energies = {}
    energy_file_path=f"{output_path}/{defect_name}/{bdm_type}/{defect_name}.txt"
    dict_energies = sort_data(energy_file_path)[0]
    for distortion, energy in dict_energies["distortions"].items():
        dict_energies["distortions"][distortion] = energy - dict_energies["Unperturbed"]
    dict_energies["Unperturbed"] = 0.0
    if units == "meV":
        for key in dict_energies["distortions"].keys():
            dict_energies["distortions"][key] = dict_energies["distortions"][key] *1000

    return dict_energies

def calculate_rms_dist(
    defect_struct: dict, 
    rms: int = 1,
    ) -> dict:
    """Function to calculate the root mean squared displacement between each distorted structures and Unperturbed one.
    Args:
        defect_struct (dict) : 
            dictionary matching distortion to pymatgen structure
        rms (int) : 
            whether to calculate root mean squared displacement (rms=0) or maximum distance between paired sites (rms=1) 
            (default : 1)
    Returns:
        dict_rms (dict): 
            dict matching distortion to rms displacement """
    dict_rms = {}
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=5, primitive_cell=False, scale=True)
    for distortion in list(defect_struct.keys()):
        if defect_struct[distortion] != "Not converged":
            try:
                dict_rms[distortion] = sm.get_rms_dist( defect_struct["Unperturbed"], 
                                                       defect_struct[distortion])[rms] 
            except TypeError: 
                dict_rms[distortion] = None # algorithm couldn't match lattices. Set rms to None
        else:
            dict_rms[distortion] = "Not converged" #not converged structure.
    
    if sum(value == None for value in dict_rms.values()) > 5 : # If rms couldn't be calculated for more than 5 distortions, then return None
        return None
    return dict_rms
