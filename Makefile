.PHONY: install test test-core test-science verify-fields lint fmt typecheck check clean

VENV := .venv
PY   := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev]"

## Everything that should pass right now, stubs and all.
test-core:
	$(PY) -m pytest -q -m "not science and not network"

## The analysis layer's specification suite.
test-science:
	$(PY) -m pytest -q -m "science and not network"

test:
	$(PY) -m pytest -q -m "not network"

## Offline implementation checks; passing is not certification of a user's model.
verify-fields:
	$(PY) -m pytest -q tests/test_field_numerics.py tests/test_field_campaign.py tests/test_field_qsar.py tests/test_field_reproduction.py tests/test_field_reference.py tests/test_app_field_qsar.py tests/test_field_sensitivity.py tests/test_tripos_reference.py tests/test_field_qualification.py tests/test_alignment_worker.py

lint:
	$(VENV)/bin/ruff check .

fmt:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

typecheck:
	$(VENV)/bin/mypy src/sarscope

check: lint typecheck test

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
