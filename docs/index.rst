.. shakenbreak documentation master file, created by
   sphinx-quickstart on Tue Aug  2 22:08:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://github.com/SMTG-Bham/ShakeNBreak/actions/workflows/test.yml/badge.svg
 :target: https://github.com/SMTG-Bham/ShakeNBreak/actions

.. image:: https://readthedocs.org/projects/shakenbreak/badge/?version=latest&style=flat
 :target: https://shakenbreak.readthedocs.io/en/latest/

.. image:: https://img.shields.io/pypi/v/shakenbreak
 :target: https://pypi.org/project/shakenbreak

.. image:: https://img.shields.io/conda/pn/conda-forge/shakenbreak?label=conda
 :target: https://anaconda.org/conda-forge/shakenbreak

.. image:: https://joss.theoj.org/papers/10.21105/joss.04817/status.svg
   :target: https://doi.org/10.21105/joss.04817

.. image:: https://img.shields.io/pypi/dm/shakenbreak
 :target: https://shakenbreak.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/npj%20Comput%20Mater%20-Mosquera--Lois%2C%20I.%2C%20Kavanagh%2C%20S.R.%2C%20Walsh%2C%20A.%20%26%20Scanlon%2C%20D.O.%20--%202023-9cf
    :target: https://www.nature.com/articles/s41524-023-00973-1
|

Welcome to ShakeNBreak!
=======================================

``ShakeNBreak`` (``SnB``) is a defect structure-searching method employing
chemically-guided bond distortions to locate ground-state and metastable
structures of point defects in solid materials.

Main features include:

1. Defect structure generation:

   * Automatic generation of distorted structures for input defects
   * Optionally, input file generation for geometry optimisation with several codes (``VASP``, ``CP2K``, ``Quantum-Espresso``, ``CASTEP`` & ``FHI-aims``)
2. Analysis:

   * Parsing of geometry relaxation results
   * Plotting of final energies versus distortion to demonstrate what energy-lowering reconstructions have been identified
   * Coordination & bonding analysis to investigate the physico-chemical factors driving an energy-lowering distortion
   * Magnetisation analysis (currently only supported for ``VASP``)

The code currently supports ``VASP``, ``CP2K``, ``Quantum-Espresso``, ``CASTEP`` and ``FHI-aims``.
Code contributions to support additional solid-state packages are welcome!

|
.. image:: SnB_Supercell_Schematic_PES_2sec_Compressed.gif
   :width: 800px
|

Literature
------------------------

- Preview: Mosquera-Lois, I.; Kavanagh, S. R. `In Search of Hidden Defects`_, *Matter* 4 (8), 2602-2605, **2021**
- Code: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. `ShakeNBreak: Navigating the defect configurational landscape`_, *Journal of Open Source Software* 7 (80), 4817, **2022**
- Theory/Method: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. `Identifying the Ground State Structures of Defects in Solids`_, *npj Comput Mater* 9, 25, **2023**
- News & Views: Mannodi-Kanakkithodi, A. `The Devil is in the Defects`_, *Nature Physics* **2023** (`Free-to-read link <https://t.co/EetpnRgjzh>`__)


.. _ShakeNBreak\: Navigating the defect configurational landscape: https://doi.org/10.21105/joss.04817
.. _Journal of Open Source Software: https://doi.org/10.21105/joss.04817
.. _Identifying the Ground State Structures of Defects in Solids: https://www.nature.com/articles/s41524-023-00973-1
.. _In Search of Hidden Defects: https://doi.org/10.1016/j.matt.2021.06.003
.. _The Devil is in the Defects: https://doi.org/10.1038/s41567-023-02049-9

Installation
========================

``ShakeNBreak`` can be installed using ``pip``:

.. code:: bash

  pip install shakenbreak

Alternatively if needed, it can also be installed from ``conda`` with:

.. code:: bash

  conda install -c conda-forge shakenbreak


If using ``VASP``, in order for ``ShakeNBreak`` to automatically generate the pseudopotential
input files (``POTCARs``), your local ``VASP`` pseudopotential directory must be set in the ``pymatgen``
configuration file ``$HOME/.pmgrc.yaml`` as follows:

.. code:: bash

  PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>

Within your ``VASP`` pseudopotential top directory, you should have a folder named ``POT_GGA_PAW_PBE``
which contains the ``POTCAR.X(.gz)`` files (in this case for PBE ``POTCARs``). Please refer to the
`doped Installation docs <https://doped.readthedocs.io/en/latest/Installation.html>`_ if you have
difficulty with this.

Developer installation
------------------------

For development work, ``ShakeNBreak`` can also be installed from a copy of the source directory:

1. Download ``ShakeNBreak`` source code using the command:

   .. code:: bash

      git clone https://github.com/SMTG-Bham/ShakeNBreak

2. Navigate to root directory:

   .. code:: bash

      cd ShakeNBreak

3. Install the code, using the command:

   .. code:: bash

      pip install -e .

   This command tries to obtain the required packages and their dependencies and install them automatically.

Usage
========================

Python API
----------------

``ShakeNBreak`` can be used through a Python API, as exemplified in the `SnB Python API Tutorial <https://shakenbreak.readthedocs.io/en/latest/ShakeNBreak_Example_Workflow.html>`_.

Command line interface
-------------------------

Alternatively, the code can be used via the command line.

|
.. image:: SnB_CLI.gif
   :width: 800px
|

The functions provided include:

* ``snb-generate``: Generate distorted structures for a given defect
* ``snb-generate_all``: Generate distorted structures for all defects present int the specified/current directory
* ``snb-run``: Submit geometry relaxations to the HPC scheduler
* ``snb-parse``: Parse the results of the geometry relaxations and write them to a ``yaml`` file
* ``snb-analyse``: Generate ``csv`` files with energies and structural differences between the final configurations
* ``snb-plot``: Generate plots of energy vs distortion, with the option to include a colorbar to quantify structural differences
* ``snb-regenerate``: Identify defect species undergoing energy-lowering distortions and test these distortions for the other charge states of the defect
* ``snb-groundstate``: Save the ground state structures to a ``Groundstate`` directory for continuation runs

More information about each function and its inputs/outputs are available from the
:ref:`CLI section of the docs <cli_commands>` or using ``-h`` help option (e.g. ``snb -h``).

We recommend at least looking through the :ref:`Tutorials <tutorials>` when first starting to use ``ShakeNBreak``, to
familiarise yourself with the full functionality and workflow.


Code Compatibility
========================

:code:`ShakeNBreak` is built to natively function using :code:`pymatgen` :code:`Defect` objects (`docs available here <https://materialsproject.github.io/pymatgen-analysis-defects/>`_) and be compatible with the most recent version of :code:`pymatgen`. If you are receiving :code:`pymatgen`-related errors when using :code:`ShakeNBreak`, you may need to update :code:`pymatgen` and/or :code:`ShakeNBreak`, which can be done with:

.. code:: bash

   pip install --upgrade pymatgen shakenbreak


:code:`ShakeNBreak` can take :code:`pymatgen` :code:`Defect` objects as input (to then generate the trial distorted
structures), **but also** can take in :code:`pymatgen` :code:`Structure` objects, :code:`doped` defect dictionaries or
structure files (e.g. :code:`POSCAR`\s for :code:`VASP`) as inputs. As such, it should be compatible with any defect code
(such as `doped <https://doped.readthedocs.io/en/latest/index.html>`_, `pydefect <https://github.com/kumagai-group/pydefect>`_,
`PyCDT <https://github.com/mbkumar/pycdt>`_, `PyLada <https://github.com/pylada/pylada-defects>`_,
`DASP <http://hzwtech.com/files/software/DASP/htmlEnglish/index.html>`_, `Spinney <https://gitlab.com/Marrigoni/spinney/-/tree/master>`_,
`DefAP <https://github.com/DefAP/defap>`_, `PyDEF <https://github.com/PyDEF2/PyDEF-2.0>`_...) that generates these files.
Please let us know if you have any issues with compatibility, or if you would like to see any additional features added to :code:`ShakeNBreak` to make it more compatible with your code.

Acknowledgements
========================

``ShakeNBreak`` has benefitted from feedback from many members of the Walsh and Scanlon research groups who have
used / are using it in their work, including Adair Nicolson, Xinwei Wang, Katarina Brlec, Joe Willis,
Zhenzhu Li, Jiayi Cen, Lavan Ganeshkumar, Daniel Sykes, Luisa Herring-Rodriguez, Alex Squires, Sabrine Hachmioune and
Chris Savory.

Contributing
========================

Bugs reports, feature requests and questions
----------------------------------------------

Please use the `Issue Tracker <https://github.com/SMTG-Bham/ShakeNBreak/issues>`_
to report bugs or request new features.

Contributions to extend this package are very welcome! Please use the
`"Fork and Pull" <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_
workflow to do so and follow the `PEP8 <https://peps.python.org/pep-0008/>`_ style guidelines.

See the `Contributing Documentation <https://shakenbreak.readthedocs.io/en/latest/Contributing.html>`_ for detailed instructions.

Tests
----------------------------------------------

Unit tests are in the ``tests`` directory and can be run from the top directory using
`unittest <https://docs.python.org/3/library/unittest.html>`_.
Automatic testing is run on the master and develop branches using Github Actions. Please
run tests and add new tests for any new features whenever submitting pull requests.

Studies using ``ShakeNBreak``
=============================

- B\. E. Murdock et al. **Li-Site Defects Induce Formation of Li-Rich Impurity Phases: Implications for Charge Distribution and Performance of LiNi** :sub:`0.5-x` **M** :sub:`x` **Mn** :sub:`1.5` **O** :sub:`4`  **Cathodes (M = Fe and Mg; x = 0.05–0.2)** `Advanced Materials <https://doi.org/10.1002/adma.202400343>`_ 2024
- A\. G. Squires et al. **Oxygen dimerization as a defect-driven process in bulk LiNiO₂** `ChemRxiv <https://doi.org/10.26434/chemrxiv-2024-lcmkj>`_ 2024
- X\. Wang et al. **Upper efficiency limit of Sb₂Se₃ solar cells** `Joule <https://doi.org/10.1016/j.joule.2024.05.004>`_ 2024
- I\. Mosquera-Lois et al. **Machine-learning structural reconstructions for accelerated point defect calculations** `arXiv <https://doi.org/10.48550/arXiv.2401.12127>`_ 2024
- S\. R. Kavanagh et al. **doped: Python toolkit for robust and repeatable charged defect supercell calculations** `Journal of Open Source Software <https://doi.org/10.21105/joss.06433>`_ 2024
- K\. Li et al. **Computational Prediction of an Antimony-based n-type Transparent Conducting Oxide: F-doped Sb₂O₅** `Chemistry of Materials <https://doi.org/10.1021/acs.chemmater.3c03257>`_ 2024
- X\. Wang et al. **Four-electron negative-U vacancy defects in antimony selenide** `Physical Review B <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134102>`_ 2023
- Y\. Kumagai et al. **Alkali Mono-Pnictides: A New Class of Photovoltaic Materials by Element Mutation** `PRX Energy <http://dx.doi.org/10.1103/PRXEnergy.2.043002>`__ 2023
- J\. Willis, K. B. Spooner, D. O. Scanlon. **On the possibility of p-type doping in barium stannate** `Applied Physics Letters <https://doi.org/10.1063/5.0170552>`__ 2023
- A\. T. J. Nicolson et al. **Cu₂SiSe₃ as a promising solar absorber: harnessing cation dissimilarity to avoid killer antisites** `Journal of Materials Chemistry A <https://doi.org/10.1039/D3TA02429F>`__ 2023
- J\. Cen et al. **Cation disorder dominates the defect chemistry of high-voltage LiMn** :sub:`1.5` **Ni** :sub:`0.5` **O₄ (LMNO) spinel cathodes** `Journal of Materials Chemistry A`_ 2023
- J\. Willis & R. Claes et al. **Limits to Hole Mobility and Doping in Copper Iodide** `Chemistry of Materials <https://doi.org/10.1021/acs.chemmater.3c01628>`__ 2023
- I\. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon **Identifying the ground state structures of point defects in solids** `npj Computational Materials`_ 2023
- B\. Peng et al. **Advancing understanding of structural, electronic, and magnetic properties in 3d-transition-metal TM-doped α-Ga₂O₃ (TM = V, Cr, Mn, and Fe)** `Journal of Applied Physics <https://doi.org/10.1063/5.0173544>`__ 2023
- Y\. T. Huang & S. R. Kavanagh et al. **Strong absorption and ultrafast localisation in NaBiS₂ nanocrystals with slow charge-carrier recombination** `Nature Communications`_ 2022
- S\. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt **Impact of metastable defect structures on carrier recombination in solar cells** `Faraday Discussions`_ 2022
- Y-S\. Choi et al. **Intrinsic Defects and Their Role in the Phase Transition of Na-Ion Anode Na₂Ti₃O₇** `ACS Applied Energy Materials <https://doi.org/10.1021/acsaem.2c03466>`__ 2022 (Early version)
- S\. R. Kavanagh, D. O. Scanlon, A. Walsh **Rapid Recombination by Cadmium Vacancies in CdTe** `ACS Energy Letters <https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380>`__ 2021
- C\. J. Krajewska et al. **Enhanced visible light absorption in layered Cs₃Bi₂Br₉ through mixed-valence Sn(II)/Sn(IV) doping** `Chemical Science`_ 2021 (Early version)
- (News & Views): A. Mannodi-Kanakkithodi **The devil is in the defects** `Nature Physics`_ 2023 (`Free-to-read link <https://t.co/EetpnRgjzh>`__)

.. Se
.. Wenzhen paper
.. Oba book
.. BiOI
.. Kumagai collab paper
.. Lavan LiNiO2
.. Sykes Magnetic oxide polarons
.. Kat YTOS
.. Squires (and mention benchmark test against AIRSS? See Slack message)

.. _Journal of Materials Chemistry A: https://doi.org/10.1039/D3TA00532A
.. _npj Computational Materials: https://www.nature.com/articles/s41524-023-00973-1
.. _Nature Communications: https://www.nature.com/articles/s41467-022-32669-3
.. _Faraday Discussions: https://doi.org/10.1039/D2FD00043A
.. _ACS Energy Letters: https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380
.. _Nature Physics: https://doi.org/10.1038/s41567-023-02049-9
.. _Chemical Science: https://doi.org/10.1039/D1SC03775G

License and Citation
========================

``ShakeNBreak`` is made available under the MIT License.

If you use it in your research, please cite:

- Code: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. `ShakeNBreak: Navigating the defect configurational landscape`_. *Journal of Open Source Software* 7 (80), 4817, **2022**
- Theory/Method: Mosquera-Lois, I. & Kavanagh, S. R.; Walsh, A.; Scanlon, D. O. `Identifying the Ground State Structures of Defects in Solids`_. *npj Comput Mater* 9, 25, **2023**

You may also find this Preview paper useful, which discusses the general problem of defect structure prediction:

- Mosquera-Lois, I.; Kavanagh, S. R. `In Search of Hidden Defects`_. *Matter* 4 (8), 2602-2605, **2021**

``BibTeX`` entries for these papers are provided in the repository `CITATIONS.md <https://github.com/SMTG-Bham/ShakeNBreak/blob/main/CITATIONS.md>`_ file.

.. _ShakeNBreak\: Navigating the defect configurational landscape: https://doi.org/10.21105/joss.04817
.. _Journal of Open Source Software: https://doi.org/10.21105/joss.04817
.. _Identifying the Ground State Structures of Defects in Solids: https://www.nature.com/articles/s41524-023-00973-1
.. _In Search of Hidden Defects: https://doi.org/10.1016/j.matt.2021.06.003


Requirements
========================

``ShakeNBreak`` is compatible with Python 3.9 - 3.12 and requires the following open-source python packages:

* `Pymatgen <https://pymatgen.org/>`_
* `Ase <https://wiki.fysik.dtu.dk/ase/>`_
* `Hiphive <https://hiphive.materialsmodeling.org/>`_
* `Numpy <https://numpy.org/>`_
* `Matplotlib <https://matplotlib.org/>`_
* `Pandas <https://pandas.pydata.org/>`_
* `Seaborn <https://seaborn.pydata.org/>`_
* `Monty <https://pythonhosted.org/monty/index.html>`_
* `Click <https://click.palletsprojects.com/en/8.1.x/>`_

.. toctree::
   :hidden:
   :caption: Usage
   :maxdepth: 4

   Installation
   Python API <modules>
   Tutorials
   Tips

.. toctree::
   :hidden:
   :caption: Information
   :maxdepth: 1

   Code_Compatibility
   Contributing
   changelog_link
   ShakeNBreak on GitHub <https://github.com/SMTG-Bham/ShakeNBreak>