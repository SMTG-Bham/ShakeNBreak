import itertools
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.core import Spin
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")


def hole_finder(vasprun_defect, bands_below=3, bands_above=3):
    warnings.warn("""Python is 0-indexed, VASP in 1-indexed. All k-points and
    bands shown as an output from this script are 1-indexed, i.e. VASP-friendly.""")

    warnings.warn("""This is only designed to work for cases where you have a single hole...
    For multiple holes, or electrons, use the big table to work it out yourself!""")

    defect_data = BSVasprun(filename=vasprun_defect)

    print("Assigning eigenvalue spin channels...")

    eigen_up = defect_data.eigenvalues[Spin.up]
    eigen_down = defect_data.eigenvalues[Spin.down]

    for k in range(len(eigen_up)):
        for b in range(len(eigen_up[k])):
            if eigen_up[k][b][0] == defect_data.eigenvalue_band_properties[2] or eigen_down[k][b][0] == \
                    defect_data.eigenvalue_band_properties[2]:
                print(f"VBM band number: {b + 1}, k-point: {k + 1}")
                vbm_data = [b, k]

    print("\nLoading eigenvalue table...\n")

    eigenvals = [
        [
            b + 1,
            k + 1,
            eigen_up[k][b][0],
            eigen_down[k][b][0],
            eigen_up[k][b][1],
            eigen_down[k][b][1],
        ]
        for b, k in itertools.product(
            range(vbm_data[0] - bands_below, vbm_data[0] + bands_above),
            range(len(eigen_up)),
        )
    ]
    df = pd.DataFrame(eigenvals,
                      columns=["band", "k-point", "up-eigenval", "down-eigenval", "up-occupancy",
                               "down-occupancy"])

    print(df)

    print("======================================================================================\n")

    print(
        "Locating hole... Disregard if you are concerned with any other situation than a defect with a SINGLE HOLE\n")
    print(
        "Information on EINT min and max can be ignored if you are using pymatgen to make your PARCHG file")

    hole_spin_down = df[(df["up-occupancy"] == 1.0) & (df["down-occupancy"] == 0.0)]
    hole_spin_up = df[(df["up-occupancy"] == 0.0) & (df["down-occupancy"] == 1.0)]

    if hole_spin_up.empty == True and hole_spin_down.empty == False:
        print("Hole is in the spin down channel:\n", hole_spin_down)
        print("Recommended min and max for EINT in PARCHG calculation:",
              round(hole_spin_down["down-eigenval"].min(), 2),
              round(hole_spin_down["down-eigenval"].max(), 2))
    elif hole_spin_down.empty == True and hole_spin_up.empty == False:
        print("Hole is in the spin up channel:\n", hole_spin_up)
        print("Recommended min and max for EINT in PARCHG calculation:",
              round(hole_spin_up["up-eigenval"].min(), 2),
              round(hole_spin_up["up-eigenval"].max(), 2))
    elif hole_spin_up.empty == True and hole_spin_down.empty == True:
        print("No holes found!")

if __name__ == "__main__":
    hole_finder("vasprun.xml")  # insert your vasprun.xml file here
