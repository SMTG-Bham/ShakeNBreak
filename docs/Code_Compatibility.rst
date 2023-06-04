Code Compatibility
========================

:code:`ShakeNBreak` is built to natively function using :code:`pymatgen` :code:`Defect` objects
(`docs available here <https://materialsproject.github.io/pymatgen-analysis-defects/>`_) and be compatible with the
most recent version of :code:`pymatgen`. If you are receiving :code:`pymatgen`-related errors when using
:code:`ShakeNBreak`, you may need to update :code:`pymatgen` and/or :code:`ShakeNBreak`, which can be done with:

.. code:: bash

   pip install --upgrade pymatgen shakenbreak


:code:`ShakeNBreak` can take :code:`pymatgen` :code:`DefectEntry` objects as input (to then generate the trial distorted
structures), **but also** can take in :code:`pymatgen` :code:`Structure` objects, :code:`doped` defect dictionaries or
structure files (e.g. :code:`POSCAR`\s for :code:`VASP`) as inputs. As such, it should be compatible with any defect code
(such as `doped <https://github.com/SMTG-UCL/doped>`_, `pydefect <https://github.com/kumagai-group/pydefect>`_,
`PyCDT <https://github.com/mbkumar/pycdt>`_, `PyLada <https://github.com/pylada/pylada-defects>`_,
`DASP <http://hzwtech.com/files/software/DASP/htmlEnglish/index.html>`_, `Spinney <https://gitlab.com/Marrigoni/spinney/-/tree/master>`_,
`DefAP <https://github.com/DefAP/defap>`_, `PyDEF <https://github.com/PyDEF2/PyDEF-2.0>`_...) that generates these files.

Please let us know if you have any issues with compatibility, or if you would like to see any additional features added
to :code:`ShakeNBreak` to make it more compatible with your code.
