Change Log
==========

v22.11.1
--------

Main changes:

- Update rattling procedure; :code:`stdev` be automatically set to 10% bulk bond length and :code:`seed` alternated for different
  distortions (set to 100*distortion_factor) to avoid rare 'stuck rattle' occurrences.
- Refactor :code:`pickle` usages to :code:`JSON` serialisation to be more robust to package (i.e. pymatgen) updates.
- Update :code:`snb-regenerate` to be more robust, can be continually rerun without generating duplicate calculations.
- Update :code:`snb-run` to consider calculations with >50 ionic steps and <2 meV energy change as converged.
- Minor changes, efficiency improvements and bug fixes.


v22.10.14
--------

Just bumping version number to test updated GH Actions pip-install-test workflow.

v22.10.13
--------

Main changes:

- Updated defect name handling to work for all conventions
- More robust `snb-generate` and plotting behaviour
- Add CLI summary GIF to docs and README
- Updated `snb-run` behaviour to catch high-energies and forces error to improve efficiency
- Many miscellaneous tests and fixes
- Docs updates

v22.9.21
--------

Main changes:

- Fonts now included in `package_data` so can be installed with `pip` from `PyPI`
- Refactoring `distortion_plots` plot saving to saving to defect directories, and preventing overwriting of previous plots
- Miscellaneous tests and fixes
- Add summary GIF to docs and README
- Handling for partial oxidation state input
- Setting `EDIFFG = -0.01` and `local_rattle = False` as default


v22.9.2
--------

Main changes:

- Update CLI commands (snb-parse, analyse, plot and groundstate can all now be run with no arguments within a defect folder)
- Update custom font
- Update groundstate() tests
- Update plotting


v22.9.1
--------

Main changes:

- Test for pip install
- Automatic release and upload to pypi
- Add ShakeNBreak custom font, and automatise its installation
- Update ShakeNBreak default INCAR for VASP relaxations
- Formatting

v1.0.1
------

Main changes:

- Docs formatting
- Update pymatgen version to v2022.7.25, while refactoring to be compatible with v2022.8.23 takes place.

v1.0
------

Release with full code functionality (CLI and Python), pre JOSS submission.

v0.2
------

Release with final module architecture of the code. Implemented command-line interface
and I/O to codes other than VASP.

v0.1
------

First release with full functionality present, except CLI and I/O to codes other than VASP.


v0.0
------

Initial version of the package.

Added
~~~~~

- Script files:

    - BDM
    - distortions
    - energy_lowering_distortions
    - plot_BDM
    - analyse_defects
    - champion_defects_rerun
