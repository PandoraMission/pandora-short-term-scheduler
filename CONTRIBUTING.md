Contributing
============

Thanks for helping improve the pandora-short-term-scheduler project! A few quick notes to get your development environment ready and to keep contributions consistent.

Local setup
-----------
1. Install dependencies (Poetry is used for packaging):

```bash
poetry install --with dev
```

2. Run the test-suite while developing:

```bash
make pytest
```

Formatting & linting
--------------------
- Reformat code with Black (line length = 79):

```bash
make black
```

- Sort imports with isort:

```bash
make isort
```

- Run the linter:

```bash
make flake8
```

- Ruff may be used to auto-fix many issues:

```bash
poetry run ruff check --fix src tests
```

Pre-commit
----------
We recommend installing pre-commit hooks to catch issues before committing:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Commit and PR workflow
----------------------
- Create a branch, make changes, run the tests and linters locally.
- Open a PR against `main`. CI runs ruff, flake8, and pytest.

Coding style
------------
- Keep functions reasonably small; if a function grows complex, consider refactoring into smaller helpers.
- Use `astropy.time.Time` and return `Time` objects where applicable.

If you're unsure about a change that affects the scheduler algorithm, open a draft PR and add a descriptive note so reviewers can focus on the algorithmic intent.
