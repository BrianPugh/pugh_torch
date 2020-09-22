.PHONY: train test test-pdb

PYTHON := python3

train:
	$(PYTHON) train.py

test:
	$(PYTHON) -m pytest tests

test-pdb:
	$(PYTHON) -m pytest tests -s --pdb

lint:
	black .
