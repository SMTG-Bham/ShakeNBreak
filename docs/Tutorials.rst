Tutorials
===========================================================

Python API
----------
Usage of ShakeNBreak from a python API is exemplified in a `jupyter notebook <https://github.com/SMTG-UCL/ShakeNBreak/tree/main/tutorials>`_.
This tutorial can also be run interactively using
`Binder <https://mybinder.org/v2/gh/SMTG-UCL/ShakeNBreak/HEAD?urlpath=https%3A%2F%2Fgithub.com%2FSMTG-UCL%2FShakeNBreak%2Fblob%2Fdevelop%2Ftutorials%2FShakeNBreak_Example_Workflow.ipyn>`_.

Command line interface (CLI)
----------------------------
Additionally, the core functionality of the code can be accessed through the command line, with the following commands:

1. Generation of distorted structures and/or their relaxation input files:
    * for a specific defect: ``snb-generate``
    * for all defects present in a directory: ``snb-generate_all``
2. Submitting the geometry optimisations:
    ``snb-run``
3. Parsing of the geometry optimisation results:
    ``snb-parse``
4. Analysis of the energies and structural differences between the relaxed configurations:
    ``snb-analyse``
5. Plotting of energy vs distortions to identify what energy lowering reconstructions have been found:
    ``snb-plot``
6. Identification of defect species undergoing energy-lowering distortions and test these distortions for the other charge states of the defect
    ``snb-regenerate``

All commands are documented in the `Python API section <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html>`_,
and their use is exemplified in the following pages:

.. toctree::
   :maxdepth: 2

   Generation
   Analysis

Getting help
~~~~~~~~~~~~~~~~~~~~~~
Beyond the `documentation <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html>`_,
we can get a full list of accepted flags and their description using the :mod:`--help` or :mod:`-h` flag, e.g.:

.. code-block:: bash

    $ snb-generate --help

    > Usage: snb-generate [OPTIONS]
    Generate the trial distortions for structure-searching for a given defect.
    Options:
    -d, --defect FILE               Path to defect structure  [required]
    -b, --bulk FILE                 Path to bulk structure  [required]
    -c, --charge INTEGER            Defect charge state
    --min-charge, --min INTEGER     Minimum defect charge state for which to
    --max-charge, --max INTEGER     Maximum defect charge state for which to
                                    generate distortions
    --defect-index, --idx INTEGER   Index of defect site in defect structure, in
                                    case auto site-matching fails
    --defect-coords, --def-coords <FLOAT FLOAT FLOAT>...
                                    Fractional coordinates of defect site in
                                    defect structure, in case auto site-matching
                                    fails. In the form 'x y z' (3 arguments)
    --code TEXT                     Code to generate relaxation input files for.
                                    Options: 'VASP', 'CP2K', 'espresso',
                                    'CASTEP', 'FHI-aims'. Defaults to 'VASP'
    -n, --name TEXT                 Defect name for folder and metadata
                                    generation. Defaults to pymatgen standard:
                                    '{Defect Type}_mult{Supercell Multiplicity}'
    --config FILE                   Config file for advanced distortion
                                    settings. See example
                                    in/input_files/example_generate_config.yaml
    -v, --verbose                   Print information about identified defects
                                    and generated distortions
    -h, --help                      Show this help message and exit

