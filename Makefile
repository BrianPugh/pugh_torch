.PHONY: clean clean-test clean-pyc clean-build docs help release release-patch release-minor release-major
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

PYTHON := python3

BROWSER := $(PYTHON) -c "$$BROWSER_PYSCRIPT"

RELEASE_PUSH := git push \
	&& git push --tags \
	&& git branch -D stable \
	&& git checkout -b stable \
	&& git push --set-upstream origin stable -f \
	&& git checkout master

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean:  ## clean all build, python, and testing files
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

build: ## run tox / run tests and lint
	tox

lint:
	black pugh_torch
	black experiments

check-lint:
	tox -e lint

gen-docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pugh_torch*.rst
	rm -f docs/modules.rst
	tox -e docs

docs: gen-docs  ## generate Sphinx HTML documentation, including API docs, and serve to browser
	$(BROWSER) docs/_build/html/index.html

test:
	$(PYTHON) -m pytest pugh_torch/tests

test-pdb:
	$(PYTHON) -m pytest pugh_torch/tests -s --pdb

release:
	# To be called after bumping version
	$(RELEASE_PUSH)

release-patch:
	bumpversion patch
	$(RELEASE_PUSH)

release-minor:
	bumpversion minor
	$(RELEASE_PUSH)

release-major:
	bumpversion major
	$(RELEASE_PUSH)

