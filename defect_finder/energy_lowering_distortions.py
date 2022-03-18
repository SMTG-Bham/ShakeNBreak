""" 
Functions to apply the energy lowering dictortion found for a certain charge state of a defect to 
the other charge states.
In progress
"""
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

def get_deep_distortions(
    defect_charges: dict, 
    bdm_type: str='BDM',
    stol = 0.2,
    ) -> dict:
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

def compare_gs_struct_to_bdm_structs(
    gs_contcar: Structure, 
    defect_name: str, 
    base_path: str,
    stol: float = 0.2,
    ) -> bool:
    """
    Compares the ground-state structure found for a certain charge state with all BDM structures found for other charge states
    to avoid trying the ground-state distortion when it has already been found.
    Args: 
        gs_contcar (Structure): 
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
            
def import_deep_distortion_by_type(
    defect_list: list,
    fancy_defects: dict,
    ) -> list:
    """
    Imports the ground-state distortion found for a certain charge state in order to \
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