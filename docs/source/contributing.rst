Contributing
============

Contributions to **lvm_tools** are welcome. This guide describes the development workflow and contribution process.

Development Setup
-----------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone git@github.com:TomHilder/lvm_tools.git
   cd lvm_tools
   pip install -e ".[dev]"

Or using uv:

.. code-block:: bash

   uv sync --all-extras

Code Style
----------

This project uses:

- `Ruff <https://github.com/astral-sh/ruff>`_ for linting and formatting
- Line length of 99 characters
- Type hints throughout

Run the linter:

.. code-block:: bash

   ruff check src/
   ruff format src/

Testing
-------

Tests are run with pytest:

.. code-block:: bash

   pytest tests/

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

View the built documentation at ``docs/build/html/index.html``.

Pull Requests
-------------

1. Fork the repository
2. Create a feature branch from ``main``
3. Make your changes
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

Commit Messages
~~~~~~~~~~~~~~~

Use clear, descriptive commit messages:

- ``feat: Add support for multiple wavelength ranges``
- ``fix: Handle NaN values in LSF array``
- ``docs: Update installation instructions``
- ``refactor: Simplify normalisation logic``

Reporting Issues
----------------

Report bugs and request features through `GitHub Issues <https://github.com/TomHilder/lvm_tools/issues>`_.

When reporting bugs, include:

- Python version
- Package versions (``pip freeze``)
- Minimal reproducible example
- Full error traceback

Contact
-------

For questions, contact the maintainer at Thomas.Hilder@monash.edu.
