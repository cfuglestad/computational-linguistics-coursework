# Contributing

1. Create a focused branch from `main`.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Install hooks with `pre-commit install`.
4. Add or update tests for behavioral changes.
5. Run `ruff format --check .`, `ruff check .`, `mypy src`, and `pytest`.
6. Open a pull request that explains the change, validation, and limitations.

Do not commit credentials, private data, or proprietary datasets.
