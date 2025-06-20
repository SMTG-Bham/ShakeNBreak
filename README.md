[![Build status](https://github.com/SMTG-Bham/ShakeNBreak/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/ShakeNBreak/actions)
[![Documentation Status](https://readthedocs.org/projects/shakenbreak/badge/?version=latest&style=flat)](https://shakenbreak.readthedocs.io/en/latest/)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.04817/status.svg)](https://doi.org/10.21105/joss.04817)
[![PyPI](https://img.shields.io/pypi/v/shakenbreak)](https://pypi.org/project/shakenbreak)
[![Conda](https://img.shields.io/conda/pn/conda-forge/shakenbreak?label=conda)](https://anaconda.org/conda-forge/shakenbreak)
[![Downloads](https://img.shields.io/pypi/dm/shakenbreak)](https://shakenbreak.readthedocs.io/en/latest/)
[![npj](https://img.shields.io/badge/npj%20Comput%20Mater%20-Mosquera--Lois%2C%20I.%2C%20Kavanagh%2C%20S.R.%2C%20Walsh%2C%20A.%20%26%20Scanlon%2C%20D.O.%20--%202023-9cf)](https://www.nature.com/articles/s41524-023-00973-1)

# `ShakeNBreak` (`SnB`)
<a href="https://shakenbreak.readthedocs.io/en/latest/"><img align="right" width="400" src="https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/toc.png"></a> `ShakeNBreak` is a defect structure-searching method employing chemically-guided bond distortions to
locate ground-state and metastable structures of point defects in solid materials. [Docs here!](https://shakenbreak.readthedocs.io/en/latest/)

Main features include:
1. Defect structure generation:
   * Automatic generation of distorted structures for input defects
   * Optionally, input file generation for geometry optimisation with several codes (`VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`)
2. Analysis:
   * Parsing of geometry relaxation results
   * Plotting of final energies versus distortion to demonstrate what energy-lowering reconstructions have been identified
   * Coordination & bonding analysis to investigate the physico-chemical factors driving an energy-lowering distortion
   * Magnetisation analysis (currently only supported for `VASP`)

The code currently supports `VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`. Code contributions to support additional solid-state packages are welcome.

![ShakeNBreak Summary](https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

### Literature
We kindly ask that you cite the code and theory/method paper if you use `ShakeNBreak` in your work.

- **Preview**: Mosquera-Lois, I.; Kavanagh, S. R. [In Search of Hidden Defects](https://doi.org/10.1016/j.matt.2021.06.003), _Matter_ 4 (8), 2602-2605, **2021**
- **Code**: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. [ShakeNBreak: Navigating the defect configurational landscape](https://doi.org/10.21105/joss.04817), _Journal of Open Source Software_ 7 (80), 4817, **2022**
- **Theory/Method**: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. [Identifying the Ground State Structures of Defects in Solids](https://doi.org/10.1038/s41524-023-00973-1), _npj Comput Mater_ 9, 25 **2023**
- **News & Views**: Mannodi-Kanakkithodi, A. [The Devil is in the Defects](https://doi.org/10.1038/s41567-023-02049-9), _Nature Physics_ **2023** ([Free-to-read link](https://t.co/EetpnRgjzh))
- **YouTube Overview (10 mins)**: [ShakeNBreak: Symmetry-Breaking and Reconstruction at Defects in Solids](https://www.youtube.com/watch?v=aqXlyLofLSU&ab_channel=Se%C3%A1nR.Kavanagh)
- **YouTube Seminar (35 mins)**: [Seminar: Predicting the Atomic Structures of Defects](https://www.youtube.com/watch?v=u7CdhI_1S18&ab_channel=Se%C3%A1nR.Kavanagh)

## Installation
`ShakeNBreak` can be installed using `conda`:
```bash
conda install -c conda-forge shakenbreak
```
or `pip`:
```bash
pip install shakenbreak
```

See the [Installation docs](https://shakenbreak.readthedocs.io/en/latest/Installation.html) if you encounter any issues (e.g. known issue with `phonopy` `CMake` build).

If using `VASP`, in order for `ShakeNBreak` to automatically generate the pseudopotential input files (`POTCAR`s), your local `VASP` pseudopotential directory must be set in the `pymatgen` configuration file `$HOME/.pmgrc.yaml` as follows:
```bash
PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
```
   Within your `VASP` pseudopotential top directory, you should have a folder named `POT_GGA_PAW_PBE`
   which contains the `POTCAR.X(.gz)` files (in this case for PBE `POTCAR`s). Please refer to the [`doped` Installation docs](https://doped.readthedocs.io/en/latest/Installation.html) if you have
   difficulty with this.

The font Montserrat ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL)) will be installed with the package, and will be used by default for plotting.


## Usage

### Python API
`ShakeNBreak` can be used through a Python API, as exemplified in the [SnB Python API tutorial](https://shakenbreak.readthedocs.io/en/latest/ShakeNBreak_Example_Workflow.html), with more info available on the [docs](https://shakenbreak.readthedocs.io).

### Command line interface
Alternatively, the code can be used via the command line:
![ShakeNBreak CLI](https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/SnB_CLI.gif)

The functions provided include:
* [`snb-generate`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-generate): Generate distorted structures for a given defect
* [`snb-generate_all`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-generate-all): Generate distorted structures for all defects present in the specified/current directory
* [`snb-run`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-run): Submit geometry relaxations to the HPC scheduler
* [`snb-parse`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-parse): Parse the results of the geometry relaxations and write them to a `yaml` file
* [`snb-analyse`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-analyse): Generate `csv` files with energies and structural differences between the final configurations
* [`snb-plot`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-plot): Generate plots of energy vs distortion, with the option to include a colorbar to quantify structural differences
* [`snb-regenerate`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-regenerate): Identify defect species undergoing energy-lowering distortions and test these distortions for the other charge states of the defect
* [`snb-groundstate`](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-groundstate): Save the ground state structures to a ``Groundstate`` directory for continuation runs

More information about each function and its inputs/outputs are available from the [CLI section of the docs](https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#commands) or using `-h` help option (e.g. `snb -h`).

We recommend at least looking through the [tutorials](https://shakenbreak.readthedocs.io/en/latest/Tutorials.html) when first starting to use `ShakeNBreak`, to familiarise yourself with the full functionality and workflow.
You may also find the
[YouTube Overview (10 mins)](https://www.youtube.com/watch?v=aqXlyLofLSU&ab_channel=Se%C3%A1nR.Kavanagh),
[YouTube Seminar (35 mins)](https://www.youtube.com/watch?v=u7CdhI_1S18&ab_channel=Se%C3%A1nR.Kavanagh)
and/or papers listed in the [Literature](#literature) section above useful.

## Studies using `ShakeNBreak`

- C. López et al. **_Chalcogen Vacancies Rule Charge Recombination in Pnictogen Chalcohalide Solar-Cell Absorbers_** [_arXiv_](https://arxiv.org/abs/2504.18089) 2025
- K. Ogawa et al. **_Defect Tolerance via External Passivation in the Photocatalyst SrTiO<sub>3</sub>:Al_** [_ChemRxiv_](https://doi.org/10.26434/chemrxiv-2025-j44qd) 2025
- Y. Fu & H. Lohan et al. **_Factors Enabling Delocalized Charge-Carriers in Pnictogen-Based
Solar Absorbers: In-depth Investigation into CuSbSe<sub>2</sub>_** [_Nature Communications_](https://doi.org/10.1038/s41467-024-55254-2) 2025
- S. R. Kavanagh **_Identifying Split Vacancies with Foundation Models and Electrostatics_** [_arXiv_](https://doi.org/10.48550/arXiv.2412.19330) 2025
- Y. Liu **_Small hole polarons in yellow phase δ-CsPbI<sub>3</sub>_** [_Physical Review Materials_](https://doi.org/10.1103/yr22-9j6r) 2025
- S. R. Kavanagh et al. **_Intrinsic point defect tolerance in selenium for indoor and tandem photovoltaics_** [_Energy & Environmental Science_](https://doi.org/10.1039/D4EE04647A) 2025
- J. Huang et al. **_Manganese in β-Ga<sub>2</sub>O<sub>3</sub>: a deep acceptor with a large nonradiative electron capture cross-section_** [_Journal of Physics D: Applied Physics_](https://doi.org/10.1088/1361-6463/adca42) 2025
- J. Hu et al. **_Enabling ionic transport in Li<sub>3</sub>AlP<sub>2</sub> the roles of defects and disorder_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D4TA04347B) 2025
- X. Zhao et al. **_Trace Yb doping-induced cationic vacancy clusters enhance thermoelectrics in p-type PbTe_** [_Applied Physics Letters_](https://doi.org/10.1063/5.0249058) 2025
- Z. Cai & C. Ma **_Origin of oxygen partial pressure-dependent conductivity in SrTiO<sub>3</sub>_** [_Applied Physics Letters_](doi.org/10.1063/5.0245820) 2025
- R. Desai et al. **_Exploring the Defect Landscape and Dopability of Chalcogenide Perovskite BaZrS<sub>3</sub>_** [_Journal of Physical Chemistry C_](https://doi.org/10.1021/acs.jpcc.5c01597) 2025
- C. Kaewmeechai, J. Strand & A. Shluger **_Structure and Migration Mechanisms of Oxygen Interstitial Defects in β-Ga<sub>2</sub>O<sub>3</sub>_** [_Physica Status Solidi B_](https://onlinelibrary.wiley.com/doi/10.1002/pssb.202400652) 2025 <!-- though didn't properly cite SnB or doped code papers... -->
- W. Gierlotka et al. **_Thermodynamics of point defects in the AlSb phase and its influence on phase equilibrium_** [_Computational Materials Science_](https://doi.org/10.1016/j.commatsci.2025.113934) 2025 <!-- though didn't cite SnB code paper... -->
- W. D. Neilson et al. **_Oxygen Potential, Uranium Diffusion, and Defect Chemistry in UO<sub>2±x</sub>: A Density Functional Theory Study_** [_Journal of Physical Chemistry C_](https://doi.org/10.1021/acs.jpcc.4c06580) 2024
- X. Wang et al. **_Sulfur Vacancies Limit the Open-circuit Voltage of Sb<sub>2</sub>S<sub>3</sub> Solar Cells_** [_ACS Energy Letters_](https://doi.org/10.1021/acsenergylett.4c02722) 2024
- Z. Yuan & G. Hautier **_First-principles study of defects and doping limits in CaO_** [_Applied Physics Letters_](https://doi.org/10.1063/5.0211707) 2024
- B. E. Murdock et al. **_Li-Site Defects Induce Formation of Li-Rich Impurity Phases: Implications for Charge Distribution and Performance of LiNi<sub>0.5-x</sub>M<sub>x</sub>Mn<sub>1.5</sub>O<sub>4</sub> Cathodes (M = Fe and Mg; x = 0.05–0.2)_** [_Advanced Materials_](https://doi.org/10.1002/adma.202400343) 2024
- A. G. Squires et al. **_Oxygen dimerization as a defect-driven process in bulk LiNiO2<sub>2</sub>_** [_ACS Energy Letters_](https://pubs.acs.org/doi/10.1021/acsenergylett.4c01307) 2024
- X. Wang et al. **_Upper efficiency limit of Sb<sub>2</sub>Se<sub>3</sub> solar cells_** [_Joule_](https://doi.org/10.1016/j.joule.2024.05.004) 2024
- I. Mosquera-Lois et al. **_Machine-learning structural reconstructions for accelerated point defect calculations_** [_npj Computational Materials_](https://doi.org/10.1038/s41524-024-01303-9) 2024
- S. R. Kavanagh et al. **_doped: Python toolkit for robust and repeatable charged defect supercell calculations_** [_Journal of Open Source Software_](https://doi.org/10.21105/joss.06433) 2024
- K. Li et al. **_Computational Prediction of an Antimony-based n-type Transparent Conducting Oxide: F-doped Sb<sub>2</sub>O<sub>5</sub>_** [_Chemistry of Materials_](https://doi.org/10.1021/acs.chemmater.3c03257) 2024
- S. Hachmioune et al. **_Exploring the Thermoelectric Potential of MgB4: Electronic Band Structure, Transport Properties, and Defect Chemistry_** [_Chemistry of Materials_](https://doi.org/10.1021/acs.chemmater.4c00584) 2024
- X. Wang et al. **_Four-electron negative-U vacancy defects in antimony selenide_** [_Physical Review B_](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134102) 2023
- Y. Kumagai et al. **_Alkali Mono-Pnictides: A New Class of Photovoltaic Materials by Element Mutation_** [_PRX Energy_](http://dx.doi.org/10.1103/PRXEnergy.2.043002) 2023
- A. T. J. Nicolson et al. **_Cu<sub>2</sub>SiSe<sub>3</sub> as a promising solar absorber: harnessing cation dissimilarity to avoid killer antisites_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA02429F) 2023
- J. Willis, K. B. Spooner, D. O. Scanlon **_On the possibility of p-type doping in barium stannate_** [_Applied Physics Letters_](https://doi.org/10.1063/5.0170552) 2023
- J. Cen et al. **_Cation disorder dominates the defect chemistry of high-voltage LiMn<sub>1.5</sub>Ni<sub>0.5</sub>O<sub>4</sub> (LMNO) spinel cathodes_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA00532A) 2023
- J. Willis & R. Claes et al. **_Limits to Hole Mobility and Doping in Copper Iodide_** [_Chemistry of Materials_](https://doi.org/10.1021/acs.chemmater.3c01628) 2023
- I. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon **_Identifying the ground state structures of point defects in solids_** [_npj Computational Materials_](https://www.nature.com/articles/s41524-023-00973-1) 2023
- B. Peng et al. **_Advancing understanding of structural, electronic, and magnetic properties in 3d-transition-metal TM-doped α-Ga₂O₃ (TM = V, Cr, Mn, and Fe)_** [_Journal of Applied Physics_](https://doi.org/10.1063/5.0173544) 2023
- Y. T. Huang & S. R. Kavanagh et al. **_Strong absorption and ultrafast localisation in NaBiS<sub>2</sub> nanocrystals with slow charge-carrier recombination_** [_Nature Communications_](https://www.nature.com/articles/s41467-022-32669-3) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt **_Impact of metastable defect structures on carrier recombination in solar cells_** [_Faraday Discussions_](https://doi.org/10.1039/D2FD00043A) 2022
- Y-S. Choi et al. **_Intrinsic Defects and Their Role in the Phase Transition of Na-Ion Anode Na<sub>2</sub>Ti<sub>3</sub>O<sub>7</sub>_** [_ACS Applied Energy Materials_](https://doi.org/10.1021/acsaem.2c03466) 2022 (Early version)
- S. R. Kavanagh, D. O. Scanlon, A. Walsh **_Rapid Recombination by Cadmium Vacancies in CdTe_** [_ACS Energy Letters_](https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380) 2021
- C. J. Krajewska et al. **_Enhanced visible light absorption in layered Cs<sub>3</sub>Bi<sub>2</sub>Br<sub>9</sub> through mixed-valence Sn(II)/Sn(IV) doping_** [_Chemical Science_](https://doi.org/10.1039/D1SC03775G) 2021 (Early version)
- (News & Views): A. Mannodi-Kanakkithodi **_The devil is in the defects_** [_Nature Physics_](https://doi.org/10.1038/s41567-023-02049-9) 2023 ([Free-to-read link](https://t.co/EetpnRgjzh))

<!-- Wenzhen paper -->
<!-- Oba book -->
<!-- BiOI -->
<!-- Kumagai collab paper -->
<!-- Sykes Magnetic oxide polarons -->
<!-- Kat YTOS -->

## License and Citation
ShakeNBreak is made available under the MIT License.

If you use it in your research, please cite:
- Code: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. [ShakeNBreak: Navigating the defect configurational landscape](https://doi.org/10.21105/joss.04817). _Journal of Open Source Software_ 7 (80), 4817, **2022**
- Theory/Method: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. [Identifying the Ground State Structures of Defects in Solids](https://doi.org/10.1038/s41524-023-00973-1) _npj Comput Mater_ 9, 25 **2023**

You may also find this Preview paper useful, which discusses the general problem of defect structure prediction:
- Mosquera-Lois, I.; Kavanagh, S. R. [In Search of Hidden Defects](https://doi.org/10.1016/j.matt.2021.06.003). _Matter_ 4 (8), 2602-2605, **2021**

`BibTeX` entries for these papers are provided in the [`CITATIONS.md`](CITATIONS.md) file.

## Code Compatibility
`ShakeNBreak` is built to natively function using [`doped`](https://doped.readthedocs.io) / [`pymatgen`](https://materialsproject.github.io/pymatgen-analysis-defects/) `Defect` objects and be compatible with the most recent version of `pymatgen`.
If you are receiving `pymatgen`-related errors when using `ShakeNBreak`, you may need to update `pymatgen` and/or `ShakeNBreak`, which can be done with:
```bash
pip install -U pymatgen shakenbreak
```

`ShakeNBreak` is compatible with a variety of inputs (to then generate the trial distorted structures), including  [`doped`](https://doped.readthedocs.io) / [`pymatgen`](https://materialsproject.github.io/pymatgen-analysis-defects/) `Defect` objects, `pymatgen` `Structure` objects or structure files
(e.g. `POSCAR`s for `VASP`). As such, it should be compatible with any defect code
(such as [`doped`](https://doped.readthedocs.io), [`pydefect`](https://github.com/kumagai-group/pydefect),
[`PyCDT`](https://github.com/mbkumar/pycdt), [`PyLada`](https://github.com/pylada/pylada-defects),
[`DASP`](http://hzwtech.com/files/software/DASP/htmlEnglish/index.html), [`Spinney`](https://gitlab.com/Marrigoni/spinney/-/tree/master),
[`DefAP`](https://github.com/DefAP/defap), [`PyDEF`](https://github.com/PyDEF2/PyDEF-2.0)...) or manual defect supercell generation.
Please let us know if you have any issues with compatibility, or if you would like to see any additional features added to `ShakeNBreak` to make it more compatible with your code.

## Acknowledgements
`ShakeNBreak` has benefitted from feedback from many members of the Walsh and Scanlon research groups who have used / are using it in their work, including Adair Nicolson, Xinwei Wang, Katarina Brlec, Joe Willis, Zhenzhu Li, Jiayi Cen, Lavan Ganeshkumar, Daniel Sykes, Luisa Herring-Rodriguez, Alex Squires, Sabrine Hachmioune and Chris Savory.

## Contributing

### Bugs reports, feature requests and questions
Please use the [Issue Tracker](https://github.com/SMTG-Bham/ShakeNBreak/issues) to report bugs or request new features.

Contributions to extend this package are very welcome! Please use the
["Fork and Pull"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
workflow to do so and follow the [PEP8](https://peps.python.org/pep-0008/) style guidelines.

See the [Contributing Documentation](https://shakenbreak.readthedocs.io/en/latest/Contributing.html) for detailed instructions.
