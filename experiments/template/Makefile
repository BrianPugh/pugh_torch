.PHONY: train test test-pdb lint clean

PYTHON := python3

train:
	$(PYTHON) train.py

test:
	$(PYTHON) -m pytest tests

test-pdb:
	$(PYTHON) -m pytest tests -s --pdb

lint:
	black .

clean:
	# WARNING: DELETES THE OUTPUTS/ FOLDER
	rm -rf outputs

tensorboard:
	tensorboard --logdir outputs --port 6006 --host 0.0.0.0
