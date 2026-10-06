#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Installation
.PHONY: install
install:
	$(PYTHON) -m pip install -e ".[dev]"
	-mypy --install-types --non-interactive ./

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	pyupgrade --exit-zero-even-if-changed --py311-plus **/*.py
	isort --settings-path pyproject.toml ./
	black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) pytest -c pyproject.toml --cov-report=html --cov=unifiedwvalticolocs tests/
	coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	isort --diff --check-only --settings-path pyproject.toml ./
	black --diff --check --config pyproject.toml ./
	flake8 --config setup.cfg unifiedwvalticolocs/ tests/
	darglint --verbosity 2 unifiedwvalticolocs tests

.PHONY: mypy
mypy:
	mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	bandit -ll --recursive unifiedwvalticolocs tests

.PHONY: check-safety-full
check-safety-full:
	safety scan --full-report
	bandit -ll --recursive unifiedwvalticolocs tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	$(PYTHON) -m pip install -U bandit darglint "isort[colors]" mypy pre-commit pydocstyle pylint pytest pyupgrade safety coverage coverage-badge pytest-html pytest-cov black

#* Documentation
.PHONY: docs
docs:
	sphinx-build -b html docs docs/_build/html

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
