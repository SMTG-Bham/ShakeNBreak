Structure generation
=====================

For a single defect
-------------------
We can generate the distorted structures for a specific defect in a range of charge states using:

.. code::

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --min-charge -2 --max-charge 0 --code VASP

The code will try to automatically identify the defect site in the structure. If the site is not found,
we'll get a warning and we'll need to specify the defect site with the ``--defect-index`` flag
(or ``--defect-coords`` for vacancies - no defect atom!).

.. code::

    $ snb-generate --bulk bulk_structure.cif --defect vac_1_Cd_POSCAR --min-charge -2 --max-charge 0 --defect-coords 0 0 0 --code VASP

To specify additional distortion parameters, we can use a
`config.yaml <https://github.com/SMTG-UCL/ShakeNBreak/blob/main/input_files/example_generate_config.yaml>`_ file like the one
below and use the ``--config`` flag to specify its path (i.e. ``snb-generate --config ./my_config.yaml``). A detailed description
of all the parameters is available in the Python API section (`shakenbreak.input subsection <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.input.html>`_).

.. code::

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


For many defects
-------------------

If instead of a single defect, we are interested in studying many of them,
we can use the ``snb-generate-all`` command. Here we need to specify the path to the directory containing the defect
structures/folders with the ``--defects`` flag:

.. code::

    $ snb-generate-all --bulk bulk_structure.cif --defects defects_folder --code VASP

To specify the charge state range for each defect, as well as other optional arguments, we can use a
`config.yaml <https://github.com/SMTG-UCL/ShakeNBreak/blob/main/input_files/example_generate_all_config.yaml>`_ file
like the one below:

.. code::

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