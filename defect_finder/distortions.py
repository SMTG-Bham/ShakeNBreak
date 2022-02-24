"""
Code with defetc distortion functions (bdm, rattle and local_rattle)

"""
from typing import Optional

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import MonteCarloRattleTransformation

def bdm(
    structure: Structure,
    num_nearest_neighbours: int,
    distortion_factor: float,
    atom_index: int = None, # starting from 1
    frac_coords: np.ndarray = None, # if it's a vacancy, use frac coords instead of site index (as a numpy array)
    verbose: bool = False,
    distorted_element: Optional[str] = None,
) -> dict:
    """Applies a bond distortion factor to num_nearest_neighbours of defect (indicated with atom_index (antisites/interstitials) or frac coordinates (vacancies))
    Args:
        structure (Structure): 
            defect structure in pymatgen format
        num_nearest_neighbours (int): 
            number of defect neighbours to apply distortion
        distortion factor (float): 
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
        dictionary with distorted defect structure and parameters used in the distortion"""
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
    # Create dictionary with distortion info & distorted structure
    bdm_distorted_defect = {
        "distorted_structure": distorted_structure,
        "defect_index": atom_index or None,
        "number_distorted_neighbours": num_nearest_neighbours,
        "distorted_atoms": distorted_atoms 
        }
    if verbose:
        distorted = [ (round(i[0],2), i[1], i[2]) for i in distorted ] ; nearest = [ (round(i[0],2), i[1], i[2]) for i in nearest ] # round numbers 
        print("     Defect Atom Index/frac. coords:", atom_index or frac_coords )
        print("     Original Neigbhour Distances:", nearest)
        print("     Distorted Neighbour Distances:", distorted) ; print() 

    return bdm_distorted_defect 

def rattle(
        structure: Structure, 
        stdev: float = 0.25
        ) -> Structure:
    """
    Given a pymnatgen structure, it applies a random distortion to the coordinates of all atoms. 
    Random distortion chosen from a gaussian with a standard deviation of stdev.
    Args:
        structure (Structure): pymatgen structure
        stdev (float): 
            standard dev of the gaussian used for rattle (in A)
            (default: 0.25)
    Returns:
        rattled structure
    """
    aaa = AseAtomsAdaptor()
    ase_struct = aaa.get_atoms(structure)
    ase_struct.rattle(stdev=stdev)
    rattled_structure = aaa.get_structure(ase_struct)
    
    return rattled_structure

def localized_rattle(
        structure: Structure, 
        defect_coords: np.array,
        stdev: float = 0.25,
        ):
    """
    Given a pymnatgen structure, it applies a random distortion to the coordinates of the atoms in a radius 5 A from defect atom. 
    Random distortion chosen from a gaussian with a standard deviation of stdev.
    Args:
        structure : 
            Structure
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