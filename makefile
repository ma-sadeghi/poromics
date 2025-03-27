SHELL := /bin/bash
PACKAGE_SLUG := src/poromics

ifdef CI
    PYTHON_VERSION := $(shell python --version | cut -d' ' -f2)
else
    PYTHON_VERSION := $(shell cat .python-version)
endif

PYTHON_SHORT_VERSION := $(shell echo $(PYTHON_VERSION) | grep -o '[0-9].[0-9]*')

ifeq ($(USE_SYSTEM_PYTHON), true)
    PYTHON_EXEC := python
    PYTHON_PACKAGE_PATH := $(shell python -c "import sys; print(sys.path[-1])")
else
    PYTHON_EXEC := . .venv/bin/activate && python
    PYTHON_PACKAGE_PATH := .venv/lib/python$(PYTHON_SHORT_VERSION)/site-packages
endif

PACKAGE_CHECK := $(PYTHON_PACKAGE_PATH)/build

.PHONY: all install venv pip pre-commit chores tests build publish checks fixes

install: venv
	uv sync --all-extras

venv:
	uv venv

pre-commit:
	pre-commit install

chores: fixes

fixes: ruff_fix black_fix dapperdata_fix tomlsort_fix

ruff_fix:
	uv run ruff check . --fix

black_fix:
	uv run ruff format .

dapperdata_fix:
	uv run -m dapperdata.cli pretty . --no-dry-run

tomlsort_fix:
	uv run toml-sort $(shell find . -not -path "./.venv/*" -name "*.toml") -i

tests: install pytest checks

pytest:
	uv run pytest --cov=./${PACKAGE_SLUG} --cov-report=term-missing tests

pytest_loud:
	uv run pytest --log-cli-level=DEBUG --log-cli=true --cov=./${PACKAGE_SLUG} --cov-report=term-missing tests

checks: ruff_check black_check mypy_check dapperdata_check tomlsort_check

ruff_check:
	uv run ruff check

black_check:
	uv run ruff format . --check

mypy_check:
	uv run mypy ${PACKAGE_SLUG}

dapperdata_check:
	uv run -m dapperdata.cli pretty .

tomlsort_check:
	uv run toml-sort $(shell find . -not -path "./.venv/*" -name "*.toml") --check

dependencies: requirements.txt requirements-dev.txt

requirements.txt: $(PACKAGE_CHECK) pyproject.toml
	uv run uv pip compile --upgrade --output-file=requirements.txt pyproject.toml

requirements-dev.txt: $(PACKAGE_CHECK) pyproject.toml
	uv run uv pip compile --upgrade --output-file=requirements-dev.txt --extra=dev pyproject.toml

build: $(PACKAGE_CHECK)
	uv run hatch build

publish: $(PACKAGE_CHECK)
	uv run hatch publish

deploy-docs:
	uv run mkdocs gh-deploy --force

serve-docs:
	uv run mkdocs serve
