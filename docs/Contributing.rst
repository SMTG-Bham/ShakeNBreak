Contributing
=======================================

Bugs reports, feature requests and questions
---------------------------------------------

Please use the `Issue Tracker <https://github.com/SMTG-UCL/ShakeNBreak/issues>`_ to report bugs or
request new features. Contributions to extend this package are welcome! Please use the
`Fork and Pull <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_
workflow to do so and follow the `PEP8 <https://peps.python.org/pep-0008/>`_ style guidelines.

The easiest way to handle these guidelines is to run the following in the **correct sequence**
on your local machine. First run `black <https://black.readthedocs.io/en/stable/index.html>`_,
as this will automatically reformat the code to ``PEP8`` conventions:

.. code:: bash

    $ black  --line-length 88 --diff --color shakenbreak

Then run `pycodestyle <https://pycodestyle.pycqa.org/en/latest/>`_,
followed by `flake8 <https://flake8.pycqa.org/en/latest/>`_, which will identify any remaining issues.

.. code:: bash

    $ pycodestyle --max-line-length=107 shakenbreak
    $ flake8 --max-line-length 107 --color always --ignore=E121,E123,E126,E226,E24,E704,W503,W504,F401 shakenbreak

- Please use comments, informative docstrings and tests as much as possible.


Tests
-------

Unit tests are in the `tests <https://github.com/SMTG-UCL/ShakeNBreak/tree/main/tests>`_ directory
and can be run from the top directory using ``unittest``. Automatic testing is run on the master and
develop branches using `Github Actions <https://docs.github.com/en/actions>`_. Please
run tests and add new tests for any new features whenever submitting pull requests.
