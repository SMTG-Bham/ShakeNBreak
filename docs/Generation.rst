.. _tutotial_generation:

Structure generation
=====================

For a single defect
-------------------
To generate the distorted structures for a specific defect in a range of charge states, we need to specify
the structure for the bulk material (with ``--bulk`` flag), the defect structure (``--defect``) and the charge
states (either with ``--min-charge`` and ``--max-charge`` to specify a range of charge states or with ``-charge`` 
for a single charge state):

.. code:: bash

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --min-charge -2 --max-charge 0 --code VASP

The code will try to automatically identify the defect site in the structure. If the site is not found,
we'll get a warning and we'll need to specify the defect site with the ``--defect-index`` or ``--defect-coords`` flag:

.. code:: bash

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --min-charge -2 --max-charge 0 --defect-coords 0 0 0 --code VASP

To specify additional distortion parameters, we can use a 
`config.yaml <https://github.com/SMTG-UCL/ShakeNBreak/blob/main/input_files/example_generate_config.yaml>`_ 
file like the one below and use the ``--config`` flag to specify its path (i.e. ``snb-generate --config ./my_config.yaml``).
A detailed description of all the parameters is available in the Python API section (:ref:`shakenbreak.input subsection <api_input>`).

.. code:: yaml

    # Example config.yaml file for snb-generate

    # General/Distortion section
    oxidation_states: # If not specified, the code will determine them
       Cd: 2
       Te: -2
    distortion_increment: 0.1 # Bond distortion increment
    distorted_elements:  # (Default = None, distorts nearest neighbours)
        vac_Cd_1:
            Cd # Distort Cd atoms near the Cd interstitial

    # Rattle section
    stdev: 0.25 # Rattle standard deviation
    d_min: 2.25  # Displacements that place atoms closer than d_min are penalised. (Default = 80% of auto-determined bulk bond length)
    active_atoms: None  # Atoms to apply rattle displacement to. (Default = all atoms)
    max_attempts: 5000
    max_disp: 2.0
    local_rattle: True # If True, rattle displacements will tail-off as we more away from the defect site

To display additional information about the generated distortions we can set the ``--verbose`` flag.

For many defects
-------------------

If instead of a single defect, we are interested in studying many of them,
we can use the ``snb-generate_all`` command. This requires us to specify the path
to the top-level directory containing the defect structures/folders with the ``--defects`` flag
(if not set, it will assume that our defects are located in the current directory).

.. code:: bash

    $ snb-generate-all --bulk bulk_structure.cif --defects defects_folder --code VASP

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

To specify the charge state range for each defect, as well as other optional arguments, we can use a
`config.yaml <https://github.com/SMTG-UCL/ShakeNBreak/blob/main/input_files/example_generate_all_config.yaml>`_ file
like the one below. A detailed description of all the parameters is available in the
Python API section (:ref:`shakenbreak.input subsection <api_input>`).

.. code:: yaml

    # Example config.yaml file for snb-generate-all

    # Defects section: to specify charge states and defect index/frac coords
    defects:
    vac_1_Cd:  # Name should match your defect structure file/folder
        charges: [0, -1, -2]  # List of charge states
        defect_coords: [0.0, 0.0, 0.0]  # Fractional coords for vacancies!
    Int_Cd_2:
        charges: [0, +1, +2]
        defect_index: -1

    # Distortion section
    distortion_increment: 0.1 # Increment for distortion range
    distorted_elements:  # (Default = None, distorts nearest neighbours)
        Int_Cd_2:
            Cd # Distort Cd atoms near the Cd interstitial

    # Rattle section
    stdev: 0.25
    d_min: 2.25  # (Default = 80% of auto-determined bulk bond length)
    active_atoms: None  # (Default = all atoms)
    max_attempts: 5000
    max_disp: 2.0
    local_rattle: True # If True, rattle displacements will tail-off as we more away from the defect site

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

Submitting the geometry optimisations
=======================================

Once the input files have been generated, we can submit the geometry optimisations
for a single or all defects using the ``snb-run`` command.
To submit all defects present in the current directory:

.. code:: bash

    $ snb-run --job-script my_job_script.sh --all

This assumes that our HPC has the ``SGE`` queuing system. If instead it relies on ``SLURM``,
we can use the ``--submit-command`` flag:

.. code:: bash

    $ snb-run --submit-command sbatch --job-script my_job_script.sh --all
