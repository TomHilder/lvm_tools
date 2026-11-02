# Contributing

Contributions to **lvm_tools** are welcome. This guide describes the development workflow and contribution process.

## Development Setup

Clone the repository and install in development mode:

``` bash
git clone git@github.com:TomHilder/lvm_tools.git
cd lvm_tools
pip install -e ".[dev]"
```

Or using uv:

``` bash
uv sync --all-extras
```

## Code Style

This project uses:

- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length of 99 characters
- Type hints throughout

Run the linter:

``` bash
ruff check src/
ruff format src/
```

## Testing

Tests are run with pytest:

``` bash
pytest tests/
```

## Documentation

The documentation is Markdown in `docs/`, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Preview it locally with live reload:

``` bash
pip install -r docs/requirements.txt
mkdocs serve
```

and open the address it prints. `mkdocs build --strict` checks it as CI does.

## Pull Requests

1.  Fork the repository
2.  Create a feature branch from `main`
3.  Make your changes
4.  Add tests for new functionality
5.  Update documentation as needed
6.  Submit a pull request

### Commit Messages

Use clear, descriptive commit messages:

- `feat: Add support for multiple wavelength ranges`
- `fix: Handle NaN values in LSF array`
- `docs: Update installation instructions`
- `refactor: Simplify normalisation logic`

## Reporting Issues

Report bugs and request features through [GitHub Issues](https://github.com/TomHilder/lvm_tools/issues).

When reporting bugs, include:

- Python version
- Package versions (`pip freeze`)
- Minimal reproducible example
- Full error traceback

## Contact

For questions, contact the maintainer at <Thomas.Hilder@monash.edu>.
