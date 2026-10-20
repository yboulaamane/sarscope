# Contributing

Bug reports and focused pull requests are welcome. Include the workflow, package
versions, settings and a small non-confidential example that reproduces the issue.
For scientific changes, distinguish numerical correctness from predictive scores.

## Development setup

From a fresh virtual environment in the repository root:

```bash
python -m pip install -e ".[dev,app]"
python -m ruff check .
python -m ruff format --check .
python -m mypy src/sarscope
python -m pytest -q -m "not network"
```

The app extra is needed by browser-stage and Streamlit tests. Optional docking
integrations need their own dependencies. Live-network tests are separate:

```bash
python -m pytest -m network
```

Do not commit private compound data, credentials or generated report caches.
Preserve source provenance, licences and unsuccessful benchmark outcomes.
