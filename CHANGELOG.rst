Change Log
==========

v3.4.0
----------
- Major efficiency updates:
    - Uses ``_scan_sm_stol_till_match`` and turbo-charged ``StructureMatcher`` methods from ``doped``
      ``v3``, speeding up structure matching (e.g. in ``snb-regenerate`` for identifying distinct defect
      geometries) by >~3 orders of magnitude.
    - Uses caching in atomic displacement calculations (for ``"disp"``/``"max_dist"`` metrics)
    - Use efficient Voronoi analyzer from ``doped`` ``v3`` for multiplicity determination.
    - More efficient (and cleaner) plotting with many defects/distortions
- Add ``Dimer`` to default distortions grid output for vacancies, following discussions and testing for
  cation vacancies in oxides.
- Miscellaneous:
    - All ``snb-xxx`` functions now auto-detect if running within a defect folder or in a top-level
      directory (i.e. auto ``--all`` behaviour).
    - Handling of gzipped ``OUTCAR.gz`` files for energy parsing.
    - For (rare) cases of degenerate choices of NNs to distort in terms of distance, but non-degenerate
      `combinations` (e.g. distorting 2 NNs of a square coordination, could be cis or trans choices),
      the choice is made deterministically; choosing the combination with the shortest distances between
      distorted NNs (i.e. the cis choice).
    - Greater verbosity control
    - Some code cleanup and formatting, and robustness updates

v3.3.6
----------
- Add ``py.typed`` to properly detect type hints by @Andrew-S-Rosen
- ``snb-run`` updates to improve efficiency

v3.3.5
----------
- Enforce ``doped>=2.4.4`` requirement.

v3.3.4
----------
- Make oxidation state guessing more efficient.
- Update Quantum Espresso and FHI-aims IO functions to work with new (and old) ASE release.
- Minor updates to ensure compatibility with recent ``pymatgen`` release.
- Allow unrecognised defect names when plotting.

v3.3.3
----------
- Add ``verbose`` option to more parsing/plotting functions for better control of output detail.
- Improve effiency & robustness of oxidation state handling.
- Miscellaneous efficiency (e.g. memory reduction) and robustness updates.
- Improved GitHub Actions test efficiency.

v3.3.2
----------
- Add ``verbose`` options to ``io.parse_energies()`` and ``snb-parse``, also used in ``snb-plot`` and
  ``snb-analyse``, and set to ``False`` by default to reduce verbosity of certain SnB CLI commands.
- Use ``doped`` functions to make oxi-state guessing (and thus defect initialisation) more efficient.
- Miscellaneous efficiency and robustness updates.
- Testing updates.

v3.3.1
----------
- ``distortion_metadata.json`` for each defect now saved to the individual defect folders (as well as the
  combined total distortion metadata in the top level folder) – more likely to be retained by the user
  when ``scp``\ing around etc.
- Minor updates:
    - Refactor ``_format_defect_name`` to ``format_defect_name`` from ``doped`` (now a public function)
    - Update ``snb-run`` to avoid possible 'file exists' warning
    - Update tutorials/notebooks to specify ``vasp_nkred_std`` to streamline workflow
    - Remove unnecessary ``tutorials`` folder with duplicate tutorial notebook (to reduce workload).
    - Add Binder/Colab buttons to run tutorials in the cloud from docs
    - Default verbosity updates (quieten some unnecessary info messages)
    - Make ``distortion_metadata`` overwriting/combining more robust and less (unnecessarily) verbose
- Bugfix of ``snb-run`` from ``v3.3.0``: If max number of electronic steps (``NELM``) threshold was reached
  in an ionic step, it would be falsely recognised as converged (due to ``unconverged`` being in the
  ``OUTCAR``). This would only affect ``snb-run`` behaviour in some cases with ``v3.3.0``, and if so the
  user should be warned anyway with ``Bond_Distortion_X not fully relaxed`` when later running
  ``snb-parse``/``snb-plot``/``snb-groundstate``. Now fixed. To double check, one can update
  ``ShakeNBreak`` and just re-run ``snb-run``, and any affected distortions will be correctly determined as
  unconverged and be re-submitted.

v3.3.0
----------
- Add Dimer distortion as a targeted distortion for dimer reconstructions. It pushes two of the defect NN
  to a distance of 2 Å.
- Add option ``distorted_atoms`` to the ``Distortion`` class to allow users to specify the indexes of the
  atoms to distort.
- Update tests to check the new functionality.
- Update ``get_homoionic_bonds`` to detect homoionic bonds between different cations/anions (rather than
  just bonds between the same element)
- Fix issue with ``snb-generate`` when no defect name was specified (by adding ``unrelaxed=True`` when
  calling ``get_defect_name_from_entry``)
- Update functions that read ``OUTCARs`` to be able to read ``OUTCAR.gz`` files too
- Update energies parsing to still work when all distortions are high energy, but warn
  the user about this (i.e. only ``Unperturbed``)
- Update ``snb-run`` to add early-on detection of distortions that are stuck in high energy basins and
  rename them to "High_Energy" to avoid continuing their relaxation
- Miscellaneous efficiency improvements and bug fixes

v3.2.3
----------
- Ensure the sorted ``pymatgen`` ``Structure`` is created for the VASP input (fixes a rare bug in ``v3.2.1``
  and ``v3.2.2`` where for certain structures the order of elements in the POSCAR was not properly sorted,
  which is usually fine, but messed with the ``ROPT`` ``INCAR`` setting).
- Plotting format updates (make legend frame more transparent to make datapoints behind it easier to see).
- Update tests
- Update docs (note about handling AFM systems)

v3.2.2
----------
- Consolidate ``SnB``/``doped`` ``INCAR`` defaults and remove redundant settings.
- Ensure backwards compatiblity in defect folder name handling.
- Fix bug in ``get_site_magnetizations``.

v3.2.1
----------
- Update CLI config handling.
- Remove ``shakenbreak.vasp`` module and use ``doped`` VASP file writing functions directly.
- Add INCAR/KPOINTS/POTCAR file writing tests. ``test_local.py`` now deleted as these tests are now
  automatically run in ``test_input.py``/``test_cli.py`` if ``POTCAR``\s available.

v3.2.0
----------
- Following the major release of ``doped`` ``v2.0``, now compatible with the new ``pymatgen``
  defects code (``pymatgen>2022.7.25``), this update:
    - Allows input of ``doped`` ``DefectsGenerator`` object to ``Distortions``
    - Updates the tutorials to reflect the current recommended workflow of generating defects
      with ``doped`` and then applying ``ShakeNBreak``, no longer requiring separate virtual environments 🎉

v3.1.0
----------
- Update dependencies, as ``hiphive=1.2`` has been released, making ``ShakeNBreak`` compatible with
  ``python=3.11`` 🎉

v3.0.0
----------
- Switch to semantic versioning
- Update rattling functions to handle primitive bulk materials as well as supercells.
- Add check to ``snb-run`` if there are multiple ``OUTCAR``\s present with one or less ionic steps, and if
  this is also the case for the current run -> warn the user.
- Small fixes, formatting and docs updates.

v23.06.23
----------
- Add ``snb-mag`` function, and automatically check the magnetisation from ``ISPIN = 2`` ``OUTCAR`` files when continuing
  relaxations with ``snb-run`` (and change to ``ISPIN = 1`` if magnetisation is negligible).
- Update handling of minimum distances and oxidation states, to deal with single-atom primitive unit cells and
  systems where ``pymatgen`` cannot guess the oxidation state (e.g. single-elements, intermetallics etc).
- Docs updates

v23.06.03
----------
- Make parsing of ``DefectEntry``\s more robust.
- Update dependencies (now supporting ``python=3.10`` due to ``numba`` updates)
- Refactor ``CITATION.cff`` to ``CITATIONS.md``
- Update docs, formatting and cleanup.

v23.04.27
----------
- Update ``numpy`` requirement to ``numpy>=1.21.2`` to fix ``numpy.typing.NDArray`` import error.
- Add News & Views free-to-read link to docs

v23.04.26
----------
- Updates to ``snb-run`` (copy ``job`` from parent directory if present, switch to ``ALGO = All`` if poor electronic convergence...)
- Make ``format_defect_name()`` more robust
- Update docs and ``README.md`` with published article links
- Formatting and cleanup
- Make oxidation state guessing more efficient (previously was causing bottleneck with large cells)
- Fix oxidation state guessing for rare elements
- Add note to ``Tips`` docs page about bulk phase transformation behaviour
- Refactor to ``json`` rather than ``pickle``

v23.02.08
----------
- Change ``numpy`` version requirement in ``docs/requirements.txt`` to ``numpy>=1.21`` to work with ``numpy.typing.NDArray``.

v23.02.02
----------
- Refactor Distortions() class to take in DefectEntry objects as input, rather than Defect objects, to be
  compatible with ``pymatgen-analysis-defects``.
- Fix ticks and ticklabels in plots


v23.01.25
--------

- Specify ``pandas`` version in requirements.txt to equal or higher than 1.1.0
- Refactor ``snb-regenerate`` to execute when no arguments are specified (rather than showing help message)

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

- Refactor ``Distortions()`` to a list or simple-format dict of ``Defect`` objects as input.
  Same for ``Distortions.from_structures()``
- Update defect naming to ``{Defect.name}_s{Defect.defect_site_index}`` for vacancies/substitutions and
  ``{Defect.name}_m{Defect.multiplicity}`` for interstitials. Append "a", "b", "c" etc in cases of inequivalent
  defects
- Make ``ShakeNBreak`` compatible with most recent ``pymatgen`` and ``pymatgen-analysis-defects`` packages.
- Update legend format in plots and site index/multiplicity labelling, make default format png.
- Update default charge state setting to match ``pymatgen-analysis-defects`` oxi state + padding approach.
- A lot of additional warning and error catches.
- Miscellaneous warnings and docs updates.


v22.11.7
--------

- Refactor ShakeNBreak to make it compatible with ``pymatgen>=2022.8.23``. Now ``Distortions`` takes in
  ``pymatgen.analysis.defects.core.Defect`` objects.
- Add ``Distortions.from_dict()`` and ``Distortions.from_structures()`` to generate defect distortions from a
  dictionary of defects (in doped format) or from a list of defect structures, respectively.

v22.11.1
--------

- Update rattling procedure; ``stdev`` be automatically set to 10% bulk bond length and ``seed`` alternated for different
  distortions (set to 100*distortion_factor) to avoid rare 'stuck rattle' occurrences.
- Refactor ``pickle`` usages to ``JSON`` serialisation to be more robust to package (i.e. pymatgen) updates.
- Update ``snb-regenerate`` to be more robust, can be continually rerun without generating duplicate calculations.
- Update ``snb-run`` to consider calculations with >50 ionic steps and <2 meV energy change as converged.
- Minor changes, efficiency improvements and bug fixes.


v22.10.14
--------

Just bumping version number to test updated GH Actions ``pip-install-test`` workflow.

v22.10.13
--------

- Updated defect name handling to work for all conventions
- More robust ``snb-generate`` and plotting behaviour
- Add CLI summary GIF to docs and README
- Updated ``snb-run`` behaviour to catch high-energies and forces error to improve efficiency
- Many miscellaneous tests and fixes
- Docs updates

v22.9.21
--------

- Fonts now included in ``package_data`` so can be installed with ``pip`` from ``PyPI``
- Refactoring ``distortion_plots`` plot saving to saving to defect directories, and preventing overwriting of previous plots
- Miscellaneous tests and fixes
- Add summary GIF to docs and README
- Handling for partial oxidation state input
- Setting ``EDIFFG = -0.01`` and ``local_rattle = False`` as default


v22.9.2
--------

- Update CLI commands (snb-parse, analyse, plot and groundstate can all now be run with no arguments within a defect folder)
- Update custom font
- Update groundstate() tests
- Update plotting


v22.9.1
--------

- Test for pip install
- Automatic release and upload to pypi
- Add ShakeNBreak custom font, and automatise its installation
- Update ShakeNBreak default INCAR for VASP relaxations
- Formatting

v1.0.1
------

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
