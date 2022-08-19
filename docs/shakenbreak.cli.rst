shakenbreak.cli module
===========================

Functions
------------
The `shakenbreak.cli` module includes functions to identify the defect species from
a given structure, as well as process it into a `Defect` object or a defect dictionary:

.. autofunction:: shakenbreak.cli.identify_defect

.. autofunction:: shakenbreak.cli.generate_defect_dict

Commands
--------------
`ShakeNBreak` has seven main commands: `snb-generate`, `snb-generate_all`, `snb-run`,
`snb-parse`, `snb-analyse`, `snb-plot` and `snb-regenerate`. Their functionality
and options are described below.

.. click:: shakenbreak.cli:snb
   :prog: snb
   :nested: full
