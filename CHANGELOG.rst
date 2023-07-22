Change Log
==========

v3.0.0
----------
- Switch to semantic versioning
- Update rattling functions to handle primitive bulk materials as well as supercells.
- Add check to `snb-run` if there are multiple `OUTCAR`s present with one or less ionic steps, and if
  this is also the case for the current run -> warn the user.
- Small fixes, formatting and docs updates.

v23.06.23
----------
- Add `snb-mag` function, and automatically check the magnetisation from `ISPIN = 2` `OUTCAR` files when continuing
  relaxations with `snb-run` (and change to `ISPIN = 1` if magnetisation is negligible).
- Update handling of minimum distances and oxidation states, to deal with single-atom primitive unit cells and
  systems where `pymatgen` cannot guess the oxidation state (e.g. single-elements, intermetallics etc).
- Docs updates

v23.06.03
----------
- Make parsing of `DefectEntry`s more robust.
- Update dependencies (now supporting `python=3.10` due to `numba` updates)
- Refactor `CITATION.cff` to `CITATIONS.md`
- Update docs, formatting and cleanup.

v23.04.27
----------
- Update `numpy` requirement to `numpy>=1.21.2` to fix `numpy.typing.NDArray` import error.
- Add News & Views free-to-read link to docs

v23.04.26
----------
- Updates to `snb-run` (copy `job` from parent directory if present, switch to `ALGO = All` if poor electronic convergence...)
- Make `format_defect_name()` more robust
- Update docs and `README.md` with published article links
- Formatting and cleanup
- Make oxidation state guessing more efficient (previously was causing bottleneck with large cells)
- Fix oxidation state guessing for rare elements
- Add note to `Tips` docs page about bulk phase transformation behaviour
- Refactor to `json` rather than `pickle`

v23.02.08
----------
- Change `numpy` version requirement in `docs/requirements.txt` to `numpy>=1.21` to work with `numpy.typing.NDArray`.

v23.02.02
----------
- Refactor Distortions() class to take in DefectEntry objects as input, rather than Defect objects, to be
compatible with `pymatgen-analysis-defects`.
- Fix ticks and ticklabels in plots


v23.01.25
--------

- Specify `pandas` version in requirements.txt to equal or higher than 1.1.0
- Refactor `snb-regenerate` to execute when no arguments are specified (rather than showing help message)

v23.01.7
--------

- Add 'Studies using ShakeNBreak' and 'How to Cite' to readme and docs.


v22.12.2
--------

- Add JOSS badge to docs


v22.12.1
--------

- Minor updates to paper.md and paper.bib


v22.11.29
--------

Main changes:
- Add example notebook showing how to generate interstitials and apply SnB to them.
- Fix typo in example notebook and docs.
- Add comment about font installation to Installation guide.
- Update paper.md with suggestions from editor.


v22.11.18
--------

Add docs plots.


v22.11.18
--------

Docs tutorial update.


v22.11.17
--------

Main changes:

- Refactor :code:`Distortions()` to a list or simple-format dict of :code:`Defect` objects as input.
  Same for :code:`Distortions.from_structures()`
- Update defect naming to :code:`{Defect.name}_s{Defect.defect_site_index}` for vacancies/substitutions and
  :code:`{Defect.name}_m{Defect.multiplicity}` for interstitials. Append "a", "b", "c" etc in cases of inequivalent
  defects
- Make :code:`ShakeNBreak` compatible with most recent :code:`pymatgen` and :code:`pymatgen-analysis-defects` packages.
- Update legend format in plots and site index/multiplicity labelling, make default format png.
- Update default charge state setting to match :code:`pymatgen-analysis-defects` oxi state + padding approach.
- A lot of additional warning and error catches.
- Miscellaneous warnings and docs updates.


v22.11.7
--------

Main changes:

- Refactor ShakeNBreak to make it compatible with `pymatgen>=2022.8.23`. Now `Distortions` takes in
  `pymatgen.analysis.defects.core.Defect` objects.
- Add `Distortions.from_dict()` and `Distortions.from_structures()` to generate defect distortions from a
  dictionary of defects (in doped format) or from a list of defect structures, respectively.

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
