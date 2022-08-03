# `shakenbreak`
`ShakeNBreak` is a defect structure-searching method employing chemically-guided bond distortions to locate ground-state and metastable structures of point defects in solid materials.

Main features include:
1. Defect structure generation:
   * Automatised generation of distorted structures for all input defects
   * Optionally, the input files to run geometry optimisations with several codes (`VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`) can be generated and organised into separate folders
2. Analysis:
   * Parsing of the geometry relaxation results
   * Plotting of final energies versus distortion to demonstrate what energy-lowering reconstructions have been identified
   * Coordination & bonding analysis to investigate the physico-chemical factors driving a distortion
   * Magnetisation analysis (currently only supported for `VASP`)

The code currently supports `VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`. Code contributions to support additional solid-state packages are welcome.

## Installation
ShakeNBreak can be installed using `pip`:
```bash
  pip install --user shakenbreak
```
### Developer installation
For development work, ShakeNBreak can also be installed from a copy of the source directory:
1. Download `ShakeNBreak` source code using the command:
```bash
  git clone https://github.com/SMTG-UCL/ShakeNBreak
```
2. Navigate to root directory:
```bash
  cd ShakeNBreak
```
3. Install the code, using the command:
```bash
  pip install -e .
```
   This command tries to obtain the required packages and their dependencies and install them automatically.

4. If using `VASP` (and not set), set the `VASP` pseudopotential directory in `$HOME/.pmgrc.yaml` as follows:
```bash
  PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
```
   Within your `VASP` pseudopotential top directory, you should have a folder named `POT_GGA_PAW_PBE` which contains the `POTCAR.X(.gz)` files (in this case for PBE `POTCAR`s).

## Usage

### Python API
ShakeNBreak can be used through a python API, as exemplified in the jupyter notebook `ShakeNBreak_Example_Workflow.ipynb`. This tutorial can also be run interactively using [Binder](https://mybinder.org/v2/gh/SMTG-UCL/ShakeNBreak/HEAD?urlpath=https%3A%2F%2Fgithub.com%2FSMTG-UCL%2FShakeNBreak%2Fblob%2Fdevelop%2FShakeNBreak_Example_Workflow.ipynb).

### Command line interface
Alternatively, the code can be used via the command line. The scripts provided include:
* `snb-generate`: Generate distorted structures for a given defect
* `snb-generate_all`: Generate distorted structures for all defects present int the specified/current directory
* `snb-parse`: Parse the results of the geometry relaxations and write them to a file
* `snb-analyse`: Generate `csv` files with energies and structural differences between the final configurations
* `snb-plot`: Generate plots of energy vs distortion, with the option to include a colorbar to quantify structural differences
* `snb-regenerate`: Identify defect species undergoing energy-lowering distortions and test these distortions for the other charge states of the defect

## License
ShakeNBreak is made available under the MIT License.

## Requirements
`ShakeNBreak` is compatible with Python 3.8 & 3.9 and requires the following open-source python packages:
* [Pymatgen](https://pymatgen.org/)
* [Ase](https://wiki.fysik.dtu.dk/ase/)
* [Hiphive](https://hiphive.materialsmodeling.org/)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Monty](https://pythonhosted.org/monty/index.html)
* [Click](https://click.palletsprojects.com/en/8.1.x/)

## Contributing

### Bugs reports, feature requests and questions
Please use the [Issue Tracker](https://github.com/SMTG-UCL/ShakeNBreak/issues) to report bugs or request new features.
Contributions to extend this package are welcome! Please use the ["Fork and Pull"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow to do so and follow the [PEP8](https://peps.python.org/pep-0008/) style guidelines.

### Tests
Unit tests are in the `tests` directory and can be run from the top directory using [unittest](https://docs.python.org/3/library/unittest.html).
Automatic testing is run on the master and develop branches using Github Actions.
