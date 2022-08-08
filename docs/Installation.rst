Installation
=====================

ShakeNBreak can be installed using ``pip``:

.. code:: bash

    pip install --user shakenbreak

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

4. If using ``VASP`` (and not set), set the ``VASP`` pseudopotential directory in ``$HOME/.pmgrc.yaml`` as follows:

.. code:: bash

    PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>

Within your ``VASP`` pseudopotential top directory, you should have a folder named ``POT_GGA_PAW_PBE``
which contains the ``POTCAR.X(.gz)`` files (in this case for PBE ``POTCARs``).