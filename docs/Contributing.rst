Contributing
=======================================

Bugs reports, feature requests and questions
---------------------------------------------

Please use the `Issue Tracker <https://github.com/SMTG-UCL/ShakeNBreak/issues>`_ to report bugs or
request new features. Contributions to extend this package are welcome! Please use the
`Fork and Pull <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_
workflow to do so and follow the `PEP8 <https://peps.python.org/pep-0008/>`_ style guidelines.

.. TIP::
    The easiest way to handle these guidelines is to use the example pre-commit hook.
    To do this, first install `pre-commit <https://pre-commit.com/>`_ using ``pip`` and then run ``pre-commit install`` to set up the git hook scripts:

    .. code:: bash

        $ pip install pre-commit
        $ pre-commit install

    Now, ``pre-commit`` will run *automatically* the linting tests after each commit.
    Optionally, you can use the following command to *manually* run the hooks against all files:

    .. code:: bash

        $ pre-commit run --all-files

    .. NOTE::
        Alternatively, if you prefer not to use ``pre-commit hooks``, you can manually run the following in the **correct sequence**
        on your local machine. From the ``shakenbreak`` top directory, run `isort <https://pycqa.github.io/isort/>`_ to sort and format your imports, followed by
        `black <https://black.readthedocs.io/en/stable/index.html>`_, which will automatically reformat the code to ``PEP8`` conventions:

        .. code:: bash

            $ isort . --profile black
            $ black  --diff --color shakenbreak

        Then run `pycodestyle <https://pycodestyle.pycqa.org/en/latest/>`_ to check the docstrings,
        followed by `flake8 <https://flake8.pycqa.org/en/latest/>`_, which will identify any remaining issues.

        .. code:: bash

            $ pycodestyle --max-line-length=107 --ignore=E121,E123,E126,E203,E226,E24,E704,W503,W504,F401 shakenbreak
            $ flake8 --max-line-length 107 --color always --ignore=E121,E123,E126,E203,E226,E24,E704,W503,W504,F401 shakenbreak

.. IMPORTANT::
    - Please use comments, informative docstrings and tests as much as possible.

Tests
-------

Unit tests are in the `tests <https://github.com/SMTG-UCL/ShakeNBreak/tree/main/tests>`_ directory
and can be run from the top directory using ``unittest``. Automatic testing is run on the master and
develop branches using `Github Actions <https://docs.github.com/en/actions>`_. Please
run tests and add new tests for any new features whenever submitting pull requests.
