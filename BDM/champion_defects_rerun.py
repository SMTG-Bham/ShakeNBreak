# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:25:01 2021

@author: Irea
"""

import numpy as np
import os
from ase.io.vasp import read_vasp, write_vasp
from pymatgen.io.ase import AseAtomsAdaptor, Structure
from pymatgen.core.structure import Structure
from doped import vasp_input
from BDM.analyse_defects import *

default_incar_settings={"ADDGRID": False, "ALGO": 'Normal', "EDIFFG": -0.01,
                        "IBRION": 2, "ISPIN": 2, "POTIM": 0.2,
                       "LVHAR": False, "LSUBROT": False, "LREAL": 'Auto', "LWAVE" : False}

def read_defects_directories(defect_path=None):
    """Reads all defect folders in current directory and stores defect names and charge states in dictionary."""
    if defect_path:
        path=defect_path
    else:
        path='./defects'
    list_subdirectories = [i for i in os.listdir(path=path) if ("as" in i) or ("vac" in i) or ("Int" in i)]
    list_name_charge = [i.rsplit('_', 1) for i in list_subdirectories] #split by last "_" (separate defect name from charge state)
    defect_charges_dict = {}
    for i in list_name_charge:
        if i[0] in defect_charges_dict.keys():
            if i[1] not in defect_charges_dict[i[0]]: # if charge not in value
                defect_charges_dict[i[0]].append(i[1])
        else:
            defect_charges_dict[i[0]] = [ i[1] ]
    return defect_charges_dict

def compare_champion_BDM(defect_name, 
                         base_path,
                         energy_difference=-0.1):
    """Checks if an energy lowering distortion was found by importing the gs of another charge state"""
    dict_energies_BDM, e_drop_BDM , gs_dist_BDM = sort_data("{}{}/BDM/{}.txt".format(base_path, defect_name, defect_name ) )    
    dict_energies_champ, e_drop_champ, gs_dist_champ = sort_data("{}{}/champion/{}.txt".format(base_path, defect_name, defect_name ) )               
    
    # Check what distortion lead to the lowest E structure: Unperturbed, BDM or just rattled?
    if type(gs_dist_BDM) == float or gs_dist_BDM == 'rattled': 
        min_energy_BDM = dict_energies_BDM["distortions"][gs_dist_BDM]
    else:
        min_energy_BDM = dict_energies_BDM["Unperturbed"]
    
    if type(gs_dist_champ) == float or gs_dist_champ == 'rattled':
        min_energy_champ = dict_energies_champ["distortions"][gs_dist_champ]
    else:
        min_energy_champ = dict_energies_champ["Unperturbed"]
    
    energy_diff = min_energy_champ - min_energy_BDM ## or should it be min_energy_champ - dict_energies_BDM["Unperturbed"] ?
    #  if lower E structure found by importing the gs of another charge state
    if energy_diff < energy_difference : 
        energy_diff = min_energy_champ - dict_energies_BDM["Unperturbed"] # for latter comparison, set the E relative to Unperturbed structure (not to the minimum energy found with BDM)
        return True, energy_diff
    return False, energy_diff

def get_champion_defects(defects, 
                         base_path, 
                         energy_difference=-0.1):
    """Get defect names and charge states for which BDM found a ground-state missed by Unperturbed relaxation.
    Args:
        defects (dict): defect dictionary mapping defect name to its charge states.
        base_path (str): path where the defect output is
        energy_difference (float) : minimum energy difference of defect distortion relative to Unperturbed to select the distortions that will be further analysed """
    all_defects_structs = {}
    for defect in defects:
        for charge in defects[defect]:
            defect_name = f"{defect}_{charge}" #defect + "_" + str(charge)
            
            new_struct_found = False #check where a lower E distortion was found by importing the gs of another charge state   
            # check if the gs found for another charge state was tried for the defect (this means initial BDM test didnt find a E lowering distortion)
            
            if os.path.isfile(f"{base_path}{defect_name}/champion/{defect_name}.txt"): # if champion folder exists
                BDM_type = "champion" 
                file_energies = f"{base_path}{defect_name}/{BDM_type}/{defect_name}.txt" 
                dict_energies, E_diff, gs_distortion = sort_data(file_energies) # between imported structure and imported structure + BDM
                new_struct_found, energy_diff = compare_champion_BDM(defect_name, base_path, energy_difference) # check if lower E structure found by importing the ground-state found for another charge state
                if not new_struct_found: 
                # if by importing the ground-state structure of another charge state, we didn't find a favourable distortion, check if initial BDM did
                    BDM_type = "BDM"
                    file_energies = f"{base_path}{defect_name}/{BDM_type}/{defect_name}.txt"
                    dict_energies, energy_diff, gs_distortion = sort_data(file_energies)
                    if float(energy_diff) < energy_difference  :
                        new_struct_found = True
                   
            elif os.path.isfile(f"{base_path}{defect_name}/BDM/{defect_name}.txt") :
                BDM_type = "BDM"
                file_energies = f"{base_path}{defect_name}/{BDM_type}/{defect_name}.txt"
                dict_energies, energy_diff, gs_distortion = sort_data(file_energies)
                if float(energy_diff) < energy_difference  :
                    new_struct_found = True
                
            #if a significant E drop occured then further analyse this fancy defect
            if  new_struct_found : 
                # transform to format used in file names
                if type(gs_distortion) == float :
                    if gs_distortion == 0.0 and BDM_type != "champion" :
                        gs_distortion = -0.0
                    gs_distortion = (str(100* gs_distortion)).format("{:.1f}")+ "%_BDM_Distortion"
                if gs_distortion == "rattled":
                    gs_distortion = "only_rattled"  
                # Grab CONTCAR and tranform to Structure
                try :
                    file_contcar_unperturbed = f"{base_path}{defect_name}/{BDM_type}/{defect_name}_Unperturbed_Defect/vasp_gam/CONTCAR"
                    file_contcar_distorted = f"{base_path}{defect_name}/{BDM_type}/{defect_name}_{gs_distortion}/vasp_gam/CONTCAR"
                    if BDM_type == "champion" : # then get the BDM unperturbed (not the imported structure found for other charge state) 
                        file_contcar_unperturbed = f"{base_path}{defect_name}/BDM/{defect_name}_Unperturbed_Defect/vasp_gam/CONTCAR"
                    
                    # tranform CONTCAR to Structure and store in dictionary
                    struct_unperturbed = grab_contcar(file_contcar_unperturbed) #transform to pmg Structure format
                    struct_distorted = grab_contcar(file_contcar_distorted)
                    defect_dict = {"Unperturbed": struct_unperturbed, 
                                   "Distorted": struct_distorted,
                                   "E_drop": energy_diff
                                   }
                    all_defects_structs[defect_name] = defect_dict 
                except:
                    print("Problem grabbing CONTCARs")
                                 
    return all_defects_structs