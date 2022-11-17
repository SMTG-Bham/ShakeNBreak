[![Build status](https://github.com/SMTG-UCL/ShakeNBreak/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/SMTG-UCL/ShakeNBreak/actions)
[![Documentation Status](https://readthedocs.org/projects/shakenbreak/badge/?version=latest&style=flat)](https://shakenbreak.readthedocs.io/en/latest/)
[![arXiv](https://img.shields.io/badge/arXiv-2207.09862-b31b1b.svg)](https://arxiv.org/abs/2207.09862)
[![PyPI](https://img.shields.io/pypi/v/shakenbreak)](https://pypi.org/project/shakenbreak)
[![status](https://joss.theoj.org/papers/6545bcc1a0439b16360ace684ac5aa25/status.svg)](https://joss.theoj.org/papers/6545bcc1a0439b16360ace684ac5aa25)
[![Downloads](https://img.shields.io/pypi/dm/shakenbreak)](https://shakenbreak.readthedocs.io/en/latest/)
<!--- add JOSS DOI badge here when ready, and published arxiv. Also update pypi package [![DOI]...-->

# `ShakeNBreak` (`SnB`)
<img align="right" width="400" src="https://raw.githubusercontent.com/SMTG-UCL/ShakeNBreak/main/docs/toc.png"> `ShakeNBreak` is a defect structure-searching method employing chemically-guided bond distortions to locate ground-state and metastable structures of point defects in solid materials.

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

![](docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

## Installation
ShakeNBreak can be installed using `pip`:
```bash
  pip install --user shakenbreak
```

If using `VASP`, in order for `ShakeNBreak` to automatically generate the pseudopotential input files (`POTCAR`s), your local `VASP` pseudopotential directory must be set in the `pymatgen` configuration file `$HOME/.pmgrc.yaml` as follows:
```bash
  PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
```
   Within your `VASP` pseudopotential top directory, you should have a folder named `POT_GGA_PAW_PBE` which contains the `POTCAR.X(.gz)` files (in this case for PBE `POTCAR`s). More details given [here](https://pymatgen.org/installation.html#potcar-setup).

The font Montserrat ([Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL)) will be installed with the package, and will be used by default for plotting.

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


## Usage

### Python API
`ShakeNBreak` can be used through a Python API, as exemplified in the [SnB Python API tutorial](https://shakenbreak.readthedocs.io/en/latest/ShakeNBreak_Example_Workflow.html), with more info available on the [docs](https://readthedocs.org/projects/shakenbreak).

### Command line interface
Alternatively, the code can be used via the command line:
![ShakeNBreak CLI](docs/SnB_CLI.gif)

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

## Code Compatibility
`ShakeNBreak` is built to natively function using `pymatgen` `Defect` objects ([docs available here](https://materialsproject.github.io/pymatgen-analysis-defects/)) and be compatible with the most recent version of `pymatgen`.
If you are receiving `pymatgen`-related errors when using `ShakeNBreak`, you may need to update `pymatgen` and/or `ShakeNBreak`, which can be done with:
```bash
pip install --upgrade pymatgen shakenbreak
```

`ShakeNBreak` can take `pymatgen` `Defect` objects as input (to then generate the trial distorted structures),
**_but also_** can take in `pymatgen` `Structure` objects, `doped` defect dictionaries or structure files
(e.g. `POSCAR`s for `VASP`) as inputs. As such, it should be compatible with any defect code
(such as [`doped`](https://github.com/SMTG-UCL/doped), [`pydefect`](https://github.com/kumagai-group/pydefect),
[`PyCDT`](https://github.com/mbkumar/pycdt), [`PyLada`](https://github.com/pylada/pylada-defects),
[`DASP`](http://hzwtech.com/files/software/DASP/htmlEnglish/index.html), [`Spinney`](https://gitlab.com/Marrigoni/spinney/-/tree/master),
[`DefAP`](https://github.com/DefAP/defap), [`PyDEF`](https://github.com/PyDEF2/PyDEF-2.0)...) that generates these files.
Please let us know if you have any issues with compatibility, or if you would like to see any additional features added to `ShakeNBreak` to make it more compatible with your code.

## Contributing

### Bugs reports, feature requests and questions
Please use the [Issue Tracker](https://github.com/SMTG-UCL/ShakeNBreak/issues) to report bugs or request new features.

Contributions to extend this package are very welcome! Please use the
["Fork and Pull"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
workflow to do so and follow the [PEP8](https://peps.python.org/pep-0008/) style guidelines.

See the [Contributing Documentation](https://shakenbreak.readthedocs.io/en/latest/Contributing.html) for detailed instructions.

### Tests
Unit tests are in the `tests` directory and can be run from the top directory using [unittest](https://docs.python.org/3/library/unittest.html).
Automatic testing is run on the master and develop branches using Github Actions. Please run tests and add new tests for any new features whenever submitting pull requests.

## Acknowledgements
`ShakeNBreak` has benefitted from feedback from many members of the Walsh and Scanlon research groups who have used / are using it in their work, including Adair Nicolson, Xinwei Wang, Katarina Brlec, Joe Willis, Zhenzhu Li, Jiayi Cen, Lavan Ganeshkumar, Daniel Sykes, Luisa Herring-Rodriguez, Alex Squires, Sabrine Hachmiouane and Chris Savory.
Code to identify defect species from input supercell structures was written based on the implementation in [PyCDT](https://doi.org/10.1016/j.cpc.2018.01.004) by Broberg et al.

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
