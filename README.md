# `shakenbreak`
`ShakeNBreak` is a defect structure-searching method employing chemically-guided bond distortions to locate ground-state and metastable structures of point defects in solid materials.

Main feautures include:
1. Defect structure generation:
   * Automatised generation of distorted structures for all input defects.
   * Optionally, the input files for several codes (`VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`) can be generated and organised into separate folders.
2. Analysis:
   * Parsing of the geometry relaxation results.
   * Plotting of final energies versus distortion to demonstrate what energy-lowering reconstructions have been identified.
   * Coordination & bonding analysis to investigate the physico-chemical factors driving a distortion.
   * Magnetisation analysis (currently only supported for `VASP`).

The code currently supports `VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`. Code contributions to support additional solid-state packages are welcome.

## Installation
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

## Examples
The notebook `ShakeNBreak_Example_Workflow.ipynb` demonstrates how to use `ShakeNBreak` from a python API.

## License
ShakeNBreak is made available under the MIT License.

## Requirements
`ShakeNBreak` is compatible with Python 3.8 & 3.9 and requires the following open-source python packages:
* [Pymatgen](https://pymatgen.org/)
* [Ase](https://wiki.fysik.dtu.dk/ase/)
* [Hiphive](https://hiphive.materialsmodeling.org/)
* [Doped](https://github.com/SMTG-UCL/doped)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Monty](https://pythonhosted.org/monty/index.html)
* [Click](https://click.palletsprojects.com/en/8.1.x/)

## Contributing

### Bugs reports, feature requests and questions
Please use the [Issue Tracker](https://github.com/SMTG-UCL/ShakeNBreak/issues) to report bugs or request new features.
Contributions to extend this package are welcome! Please use the ["Fork and Pull"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow to do so and follow the [`PEP8`](https://peps.python.org/pep-0008/) style guidelines.

### Tests
Unit tests are in the `tests` directory and can be run from the top directory using [unittest](https://docs.python.org/3/library/unittest.html).
Automatic testing is run on the master and develop branches using Github Actions.
