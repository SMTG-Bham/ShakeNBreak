.. _tutorials:

Tutorials
===========================================================

Python API
----------
Usage of ShakeNBreak from a Python API is exemplified in the
`SnB Python API Tutorial <https://shakenbreak.readthedocs.io/en/latest/ShakeNBreak_Example_Workflow.html>`_, which includes:

.. toctree::
   :maxdepth: 2

   ShakeNBreak_Example_Workflow


Command line interface (CLI)
----------------------------
Additionally, the core functionality of the code can be accessed through the command line, with the following commands:

1. Generation of distorted structures and/or their relaxation input files:
    * for a specific defect: `snb-generate <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-generate>`_
    * for all defects present in a directory: `snb-generate_all <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-generate-all>`_
2. Submitting the geometry optimisations to the HPC scheduler:
    `snb-run <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-run>`_
3. Parsing of the geometry optimisation results:
    `snb-parse <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-parse>`_
4. Analysis of the energies and structural differences between the relaxed configurations:
    `snb-analyse <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-analyse>`_
5. Plotting of energy vs distortions to identify what energy lowering reconstructions have been found:
    `snb-plot <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-plot>`_
6. Identification of defect species undergoing energy-lowering distortions and test these distortions for the other charge states of the defect
    `snb-regenerate <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-regenerate>`_
7. Saving the ground state structures
    `snb-groundstate <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-groundstate>`_

All commands are documented in the :ref:`Python API section <cli_commands>`,
and their use is exemplified in the following pages:

.. toctree::
   :maxdepth: 2

   Generation
   Analysis

Getting help
~~~~~~~~~~~~~~~~~~~~~~
Beyond the :ref:`documentation <cli_commands>`, we can get a full list of accepted flags and
their description using the :mod:`--help` or :mod:`-h` flag, e.g.:

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

