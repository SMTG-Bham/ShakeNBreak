# `shakenbreak`
`ShakeNBreak` is a defect structure-searching method employing chemically-guided bond distortions to locate ground-state and metastable structures of point defects in solid materials.

Main feautures include:
1. Defect structure generation:
   * Automatised generation of distorted structures for all input defects. 
   * Optionally, the input files for several codes (`VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`) can be generated and organised into separate folders.
2. Analysis:
   * Automatised parsing of the geometry relaxation results.
   * Plotting of final energies versus distortion to demonstrate what energy-lowering reconstructions have been identified.
   * Coordination & bonding analysis.
   * Magnetisation analysis (currently only supported for `VASP`).

The code currently supports `VASP`, `CP2K`, `Quantum-Espresso`, `CASTEP` & `FHI-aims`. Code contributions to support additional solid-state packages are welcome.

## Requirements
`ShakeNBreak` is compatible with Python 3.8 & 3.9 and requires `ase`, `pymatgen`, `hiphive` and `doped` (and their dependencies).

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