.. _tutorial_generation:

Structure Generation
=====================

For a brief overview of the CLI operation of `ShakeNBreak`, see the summary GIF in the
`CLI section <https://shakenbreak.readthedocs.io/en/latest/index.html#command-line-interface>`_ of the Welcome page.

For a single defect
-------------------
To generate the distorted structures for a specific defect in a range of charge states, we need to specify the structure
for the bulk material (with ``--bulk`` flag) and the defect structure (``--defect``):

.. code:: bash

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR

Here if you don't specify the charge states, :code:`ShakeNBreak` will assume a default charge range of -2 to +2. To
specify the defect charge states, we can use ``--min-charge`` and ``--max-charge`` (to specify a range of charge states)
or ``--charge`` (for a single charge state):

.. code:: bash

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --min-charge -2 --max-charge 0

The code will try to automatically identify the defect site in the structure. If the site is not found,
we'll get a warning and we'll need to specify the defect site with the ``--defect-index`` or ``--defect-coords`` flag:

.. code:: bash

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --min-charge -2 --max-charge 0 --defect-coords 0 0 0 --code VASP

.. NOTE::
    To specify additional distortion parameters, we can use a
    `config.yaml <https://github.com/SMTG-UCL/ShakeNBreak/blob/main/SnB_input_files/example_generate_config.yaml>`_
    file like the one below and use the ``--config`` flag to specify its path (i.e. ``snb-generate --config ./my_config.yaml``).
    A detailed description of all the parameters is available in the Python API section
    (:ref:`shakenbreak.input.Distortions class <api_input>`).

    .. code:: yaml

        # Example config.yaml file for snb-generate

        # General/Distortion section
        oxidation_states:  # If not specified, the code will determine them
            Cd: 2
            Te: -2
        distortion_increment: 0.1  # Bond distortion increment
        distorted_elements:  # (Default = None, distorts nearest neighbours)
            vac_Cd_1:
                Cd  # Distort Cd atoms near the Cd interstitial

        # Rattle section
        stdev: 0.25  # Rattle standard deviation (Default = 10% of auto-determined bulk bond length)
        d_min: 2.25  # Displacements that place atoms closer than d_min are penalised. (Default = 80% of auto-determined bulk bond length)
        active_atoms: None  # Atoms to apply rattle displacement to. (Default = all atoms)
        max_attempts: 5000  # Limit for how many attempted rattle moves are allowed a single atom; if this limit is reached an `Exception` is raised
        max_disp: 2.0  # Rattle moves that yields a displacement larger than max_disp will always be rejected. Rarely occurs, mostly used as a safety net
        local_rattle: False  # If True, rattle displacements will tail-off as we more away from the defect site. Not recommended as typically worsens performance.
        seed: 42  # Seed from which rattle random displacements are generated (Default = 100*distortion_factor, e.g. 40 for -60% distortion, 100 for 0% Distortion/Rattled etc)


.. NOTE::
    By default, :code:`ShakeNBreak` generates input files for the :code:`VASP` code, but this can be controlled with the
    ``--code`` flag. For instance, to use ``CP2K``:

    .. code:: bash

        $ snb-generate --code cp2k --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --defect-coords 0 0 0


.. TIP::
    To display additional information about the generated distortions we can set the ``--verbose`` flag.

For many defects
-------------------

If instead of a single defect, we are interested in studying many of them,
we can use the ``snb-generate_all`` command. This requires us to specify the path
to the top-level directory containing the defect structures/folders with the ``--defects`` flag
(if not set, it will assume that our defects are located in the current directory).

.. code:: bash

    $ snb-generate-all --bulk bulk_structure.cif --defects defects_folder

By default, the code will look for the structure files
(in ``cif`` or ``POSCAR`` format) present in the specified defects directory or in the immediate subdirectories. For example,
the following directory structures will be parsed correctly:

.. code:: bash

    defects_folder/
        |--- defect_1_POSCAR <-- The code expects the format of the structure files to be CIFs or POSCARSs
        |
        |--- defect_2_POSCAR
        |
        |--- defect_n_POSCAR

.. code:: bash

    defects_folder/
        |--- defect_1/
        |       |--- vac_1_Cd.cif
        |
        |--- defect_2/
        |       |--- POSCAR
        |
        |--- defect_n/
                |---structure.cif

.. NOTE::
    To specify the charge state range for each defect, as well as other optional arguments, we can use a
    `config.yaml <https://github.com/SMTG-UCL/ShakeNBreak/blob/main/SnB_input_files/example_generate_all_config.yaml>`_ file
    like the one below. A detailed description of all the parameters is available in the
    Python API section (:ref:`shakenbreak.input.Distortions class <api_input>`).

    .. code:: yaml

        # Example config.yaml file for snb-generate-all

        # Defects section: to specify charge states and defect index/frac coords
        defects:
            vac_1_Cd:  # Name should match your defect structure file/folder
                charges: [0, -1, -2]  # List of charge states
                defect_coords: [0.0, 0.0, 0.0]  # Fractional coords for vacancies!
            Int_Cd_2:
                charges: [0, +1, +2]
                defect_index: -1  # Lattice site of the interstitial

        # Distortion section
        distortion_increment: 0.1 # Increment for distortion range
        distorted_elements:  # (Default = None, distorts nearest neighbours)
            Int_Cd_2:
                Cd # Distort Cd atoms near the Cd interstitial

        # Rattle section
        stdev: 0.25  # Rattle standard deviation (Default = 10% of auto-determined bulk bond length)
        d_min: 2.25  # Displacements that place atoms closer than d_min are penalised. (Default = 80% of auto-determined bulk bond length)
        active_atoms: None  # Atoms to apply rattle displacement to. (Default = all atoms)
        max_attempts: 5000  # Limit for how many attempted rattle moves are allowed a single atom; if this limit is reached an `Exception` is raised
        max_disp: 2.0  # Rattle moves that yields a displacement larger than max_disp will always be rejected. Rarely occurs, mostly used as a safety net
        local_rattle: False  # If True, rattle displacements will tail-off as we more away from the defect site. Not recommended as typically worsens performance.
        seed: 42  # Seed from which rattle random displacements are generated (Default = 100*distortion_factor, e.g. 40 for -60% distortion, 100 for 0% Distortion/Rattled etc)

The ``generate_all`` command will create a folder for each charged defect in the current directory, each containing
distortion folders with the relaxation input files and structures. If using ``VASP``:

.. code:: bash

    ./
    |--- vac_1_Cd_0/
    |       |--- Unperturbed
    |       |        |--- POSCAR
    |       |        |--- KPOINTS
    |       |        |--- INCAR
    |       |        |--- POTCAR
    |       |
    |       |--- Bond_Distortion_-30.0%
    |       |        |--- POSCAR
    |       |        | ...
    |       | ...
    |
    |
    |--- vac_1_Cd_-1/
            |--- Unperturbed
            |        |--- POSCAR
            |        | ...
            | ...

.. TIP::
    See ``snb-generate_all -h`` or `the CLI docs <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.cli.html#snb-generate-all>`_
    for details on the options available for this command.

Submitting the geometry optimisations
=======================================

Once the input files have been generated, we can submit the geometry optimisations
for a single or all defects using the ``snb-run`` command.
To submit all defects present in the current directory:

.. code:: bash

    $ snb-run -a

This assumes the ``SGE`` queuing system (i.e. ``qsub`` = job submission command) for the HPC and a job script name of
``job`` by default, but again can be controlled with the ``--submit-command`` and ``--job-script`` flags
(as well as other options, see ``snb-run -h``). For example, if we are using the ``SLURM`` queuing system and a job
script file name of ``my_job_script.sh``, we would use:

.. code:: bash

    $ snb-run --submit-command sbatch --job-script my_job_script.sh --all


To submit a single defect, we can simply run the command :code:`snb-run` within the defect folder:

.. code:: bash

    $ snb-run
