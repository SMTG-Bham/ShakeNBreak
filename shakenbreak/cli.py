"""
ShakeNBreak command-line-interface (CLI)
"""
import warnings
import pickle
import click
import numpy as np
from monty.serialization import loadfn

from pymatgen.core.structure import Structure
from monty.json import MontyDecoder

from shakenbreak.input import Distortions


def identify_defect(defect_structure, bulk_structure,
                    defect_coords=None, defect_index=None):
    """
    By comparing the defect and bulk structures, identify the defect present and its site in
    the supercell, and generate a pymatgen defect object from this.
    """
    natoms_defect = len(defect_structure)
    natoms_bulk = len(bulk_structure)
    if natoms_defect == natoms_bulk - 1:
        defect_type = "vacancy"
    elif natoms_defect == natoms_bulk + 1:
        defect_type = "interstitial"
    elif natoms_defect == natoms_bulk:
        defect_type = "substitution"
    else:
        raise ValueError(
            f"Could not identify defect type from number of atoms in defect ({natoms_defect}) "
            f"and bulk ({natoms_bulk}) structures. "
            "ShakeNBreak CLI is currently only built for point defects, please contact the "
            "developers if you would like to use the method for complex defects"
        )

    if defect_coords is not None:
        if defect_index is None:
            site_displacement_tol = 0.1  # distance tolerance for site matching to identify defect
            while site_displacement_tol < 0.8:  # loop over distance tolerances
                if defect_type == "vacancy":
                    possible_defects = sorted(
                        bulk_structure.get_sites_in_sphere(defect_coords, site_displacement_tol,
                                                           include_index=True),
                        key=lambda x: x[1],
                    )
                    searched = "bulk"
                else:
                    possible_defects = sorted(
                        defect_structure.get_sites_in_sphere(defect_coords, site_displacement_tol,
                                                             include_index=True),
                        key=lambda x: x[1],
                    )
                    searched = "defect"

                if len(possible_defects) == 1:
                    defect_index = possible_defects[0][2]
                    break

                site_displacement_tol += 0.1

            if defect_index is None:
                warnings.warn(
                    f"Coordinates {defect_coords} were specified for (auto-determined) "
                    f"{defect_type} "
                    f"defect, but could not find it in {searched} structure "
                    f"(found {len(possible_defects)} possible defect sites). "
                    "Will attempt auto site-matching instead."
                )

        else:  # both defect_coords and defect_index given
            warnings.warn(
                "Both defect_coords and defect_index were provided. Only one is needed, so "
                "just defect_index will be used to determine the defect site")

    if defect_index is None:
        site_displacement_tol = 0.1  # distance tolerance for site matching to identify defect
        while site_displacement_tol < 0.8:  # loop over distance tolerances
            bulk_sites = [site.frac_coords for site in bulk_structure]
            defect_sites = [site.frac_coords for site in defect_structure]
            dist_matrix = defect_structure.lattice.get_all_distances(bulk_sites, defect_sites)
            min_dist_with_index = [
                [min(dist_matrix[bulk_index]), int(bulk_index),
                 int(dist_matrix[bulk_index].argmin())]
                for bulk_index in range(len(dist_matrix))
            ]  # list of [min dist, bulk ind, defect ind]

            site_matching_indices = []
            possible_defects = []
            if defect_type in ["vacancy", "interstitial"]:
                for mindist, bulk_index, def_struc_index in min_dist_with_index:
                    if mindist < site_displacement_tol:
                        site_matching_indices.append([bulk_index, def_struc_index])
                    elif defect_type == "vacancy":
                        possible_defects.append([bulk_index, bulk_sites[bulk_index][:]])

                if defect_type == "interstitial":
                    possible_defects = [
                        [ind, fc[:]]
                        for ind, fc in enumerate(defect_sites)
                        if ind not in np.array(site_matching_indices)[:, 1]
                    ]

            elif defect_type == "substitution":
                for mindist, bulk_index, def_struc_index in min_dist_with_index:
                    species_match = (
                            bulk_structure[bulk_index].specie
                            == defect_structure[def_struc_index].specie
                    )
                    if mindist < site_displacement_tol and species_match:
                        site_matching_indices.append([bulk_index, def_struc_index])

                    elif not species_match:
                        possible_defects.append([def_struc_index, defect_sites[def_struc_index][:]])

            if len(set(np.array(site_matching_indices)[:, 0])) != len(
                    set(np.array(site_matching_indices)[:, 1])
            ):
                raise ValueError(
                    "Error occurred in site_matching routine. Double counting of site matching "
                    f"occurred: {site_matching_indices}\nAbandoning structure parsing."
                )

            if len(possible_defects) == 1:
                defect_index = possible_defects[0][0]
                break

            site_displacement_tol += 0.1

        if defect_index is None:
            raise ValueError(
                "Defect coordinates could not be identified from site-matching. "
                f"Found {len(possible_defects)} possible defect sites – check bulk and defect "
                "structures correspond to the same supercell"
            )

    if defect_type == "vacancy":
        defect_site = bulk_structure[defect_index]
    else:
        defect_site = defect_structure[defect_index]

    for_monty_defect = {
        "@module": "pymatgen.analysis.defects.core",
        "@class": defect_type.capitalize(),
        "structure": bulk_structure,
        "defect_site": defect_site,
    }

    defect = MontyDecoder().process_decoded(for_monty_defect)
    return defect


## CLI Commands:
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group('snb', context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
def snb():
    """
    ShakeNBreak: Defect structure-searching
    """


@snb.command(name="generate", context_settings=CONTEXT_SETTINGS,
             no_args_is_help=True)
@click.option("defect", "-d", help="Path to defect structure", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("bulk", "-b", help="Path to bulk structure", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("charge", "-c", help="Defect charge state", default=None, type=int)
@click.option("min_charge", "--min",
              help="Minimum defect charge state for which to generate distortions", default=None,
              type=int)
@click.option("max_charge", "--max",
              help="Maximum defect charge state for which to generate distortions", default=None,
              type=int)
@click.option("defect_index", "--idx",
              help="Index of defect site in defect structure, in case auto site-matching fails",
              default=None, type=int)
@click.option("defect_coords", "--def-coords",
              help="Fractional coordinates of defect site in defect structure, in case auto "
                   "site-matching fails",
              default=None)
@click.option("--config", help="Config file for advanced distortion settings", default=None)
@click.option("--verbose", "-v", help="Print information about identified defects and generated "
                                      "distortions", default=False, is_flag=True)
def generate(defect, bulk, charge, min_charge, max_charge, defect_index, defect_coords,
             config, verbose):
    """
    Generate the trial distortions for structure-searching for a given defect.
    """
    defect_struc = Structure.from_file(defect)
    bulk_struc = Structure.from_file(bulk)

    defect_object = identify_defect(defect_structure=defect_struc, bulk_structure=bulk_struc,
                                    defect_index=defect_index, defect_coords=defect_coords)
    if verbose:
        click.echo(
            f"Auto site-matching identified {defect} to be "
            f"type {defect_object.as_dict()['@class']} "
            f"with site {defect_object.site}")

    if charge is not None:
        charges = [charge, ]

    elif max_charge is not None or min_charge is not None:
        if max_charge is None or min_charge is None:
            raise ValueError("If using min/max defect charge, both options must be set!")

        charge_lims = [min_charge, max_charge]
        charges = list(range(min(charge_lims),
                             max(charge_lims) + 1))  # just in case user mixes min and max
        # because of different signs ("+1 to -3" etc)

    else:
        warnings.warn("No charge (range) set for defect, assuming default range of +/-2")
        charges = list(range(-2, +3))

    single_defect_dict = {"name": defect_object.name,
                          "bulk_supercell_site": defect_object.site,
                          "defect_type": defect_object.as_dict()["@class"].lower(),
                          "site_multiplicity": defect_object.multiplicity,
                          "supercell": {"size": [1, 1, 1],
                                        "structure": defect_object.generate_defect_structure()},
                          "charges": charges
                          }

    if "Substitution" in str(type(defect_object)):
        # get bulk_site
        poss_deflist = sorted(defect_object.bulk_structure.get_sites_in_sphere(
            defect_object.site.coords, 0.01, include_index=True), key=lambda x: x[1])
        if not poss_deflist:
            raise ValueError(
                "Error in defect object generation; could not find substitution "
                f"site inside bulk structure for {defect_object.name}")
        defindex = poss_deflist[0][2]
        sub_site_in_bulk = defect_object.bulk_structure[defindex]  # bulk site of substitution

        single_defect_dict["unique_site"] = sub_site_in_bulk
        single_defect_dict["site_specie"] = sub_site_in_bulk.specie.symbol
        single_defect_dict["substitution_specie"] = defect_object.site.specie.symbol

    else:
        single_defect_dict["unique_site"] = defect_object.site
        single_defect_dict["site_specie"] = defect_object.site.specie.symbol

    if single_defect_dict["defect_type"] == "vacancy":
        defects_dict = {"vacancies": [single_defect_dict, ]}
    elif single_defect_dict["defect_type"] == "interstitial":
        defects_dict = {"interstitials": [single_defect_dict, ]}
    elif single_defect_dict["defect_type"] == "substitution":
        defects_dict = {"substitutions": [single_defect_dict, ]}

    if config is not None:
        user_settings = loadfn(config)

    Dist = Distortions(defects_dict, **user_settings)
    distorted_defects_dict, distortion_metadata = Dist.write_vasp_files(verbose=verbose)
    with open("./parsed_defects_dict.pickle", "wb") as fp:
        pickle.dump(defects_dict, fp)

# generate-all command where we loop over each directory and create our full defect_dict,
# save to pickle and run through Distortions
# for this will need to use folders as filenames so we know which goes where
# – this ok if folders aren't typical defect names?
