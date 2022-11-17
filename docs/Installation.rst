Installation
=====================

ShakeNBreak can be installed using ``pip``:

.. code:: bash

    pip install --user shakenbreak

If using ``VASP``, in order for ``ShakeNBreak`` to automatically generate the pseudopotential
input files (``POTCARs``), your local ``VASP`` pseudopotential directory must be set in the ``pymatgen``
configuration file ``$HOME/.pmgrc.yaml`` as follows:

.. code:: bash

  PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>

Within your ``VASP`` pseudopotential top directory, you should have a folder named ``POT_GGA_PAW_PBE``
which contains the ``POTCAR.X(.gz)`` files (in this case for PBE ``POTCARs``). More details given
`here <https://pymatgen.org/installation.html#potcar-setup>`_.

.. NOTE::
   The font `Montserrat <https://fonts.google.com/specimen/Montserrat/about>`
   (`Open Font License <https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL>`_)
   will be installed with the package, and will be used by default for plotting. If you prefer to use a different
   font, you can change the font in the ``matplotlib`` style sheet (in ``shakenbreak/shakenbreak.mplstyle``).

Developer's installation (*optional*)
-----------------------------------------

For development work, ``ShakeNBreak`` can also be installed from a copy of the source directory:

1. Download ``ShakeNBreak`` source code using the command:

   .. code:: bash

      git clone https://github.com/SMTG-UCL/ShakeNBreak

2. Navigate to root directory:

   .. code:: bash

      cd ShakeNBreak

3. Install the code with the command:

   .. code:: bash

      pip install -e .

   This command tries to obtain the required packages and their dependencies and install them automatically.