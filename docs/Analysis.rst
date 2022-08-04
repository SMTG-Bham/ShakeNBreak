Analysis & plotting
=====================

Parsing
----------

To parse the final energies of the geometry optimisations for a specific defect:

.. code::

    snb-parse --code vasp --defect vac_1_Cd_0 --path defects_folder

Where ``defects_folder`` is the path to the top level directory containing the defect directory.

Instead of a single defect, we can parse the results for all defects present
in a given/current directory using:

.. code::

    snb-parse --code vasp --path defects_folder

This generates a ``yaml`` file mapping each distortion to the final energy (in eV)
of the relaxed structure:

.. code::

    distortions:
        -0.6: -187.70
        -0.3: -187.45
        ...
    Unperturbed: -186.70

Analysis
----------
To analyse the structures obtained after the relaxations, we can use ``snb-analyse``.
It will generate ``csv`` files with the final energies and structural similarities
for a given/all defects. To measure structural similarity, we use the sum
of atomic displacements and the maximum distance between matched sites.
For instance, to analyse the results obtained with ``VASP``
for the defect ``vac_1_Cd_0``, we can use:

.. code::

    snb-analyse --defect vac_1_Cd_0 --code vasp --path defects_folder

Plotting
-----------
To quickly identify any energy lowering distortions, we can plot the final energies
of the relaxed structures versus the distortion factor using ``snb-plot``:

.. code::

    snb-plot --defect vac_1_Cd_0 --path defects_folder

which will generate a figure like the one below:

.. image:: ./vac_1_Cd_0.svg
    :width: 400px

We can make these plots more informative by adding a colobar measuring the structural
similarity between the structures:

.. code::

    snb-plot --defect vac_1_Cd_0 --path defects_folder --colorbar

.. image:: ./vac_1_Cd_0_colobar.svg
    :width: 400px