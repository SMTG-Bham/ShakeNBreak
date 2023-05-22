.. _api_cli:

shakenbreak.cli module
===========================

Functions
------------
The ``shakenbreak.cli`` module includes functions to identify the defect species from
a given structure, as well as process it into a ``Defect`` object or a defect dictionary:

.. autofunction:: shakenbreak.cli.identify_defect

.. autofunction:: shakenbreak.cli.generate_defect_dict

.. _cli_commands:

Commands
--------------

``ShakeNBreak`` has eight main commands: ``snb-generate``, ``snb-generate_all``, ``snb-run``,
``snb-parse``, ``snb-analyse``, ``snb-plot``, ``snb-regenerate`` and ``snb-groundstate``.
Their functionality and options are described below.

See the `Structure generation <https://shakenbreak.readthedocs.io/en/latest/Generation.html>`_ and
`Analysis & plotting <https://shakenbreak.readthedocs.io/en/latest/Analysis.html>`_ Tutorials pages for an overview of
their usage.


.. click:: shakenbreak.cli:snb
   :prog: snb
   :nested: full