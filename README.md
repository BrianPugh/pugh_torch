# Pugh Torch

[![Build Status](https://github.com/BrianPugh/pugh_torch/workflows/Build%20Master/badge.svg)](https://github.com/BrianPugh/pugh_torch/actions)
[![Documentation](https://github.com/BrianPugh/pugh_torch/workflows/Documentation/badge.svg)](https://BrianPugh.github.io/pugh_torch)
[![Code Coverage](https://codecov.io/gh/BrianPugh/pugh_torch/branch/master/graph/badge.svg)](https://codecov.io/gh/BrianPugh/pugh_torch)
[![Code Coverage](https://img.shields.io/pypi/v/pugh_torch.svg)](https://pypi.org/project/pugh-torch/)

Functions, losses, module blocks to share between experiments.

---

## Package Features
* Additional methods to TensorBoard summary writer for adding normalized images and semantic segmentation images.
* hetero_cross_entropy for cross_entropy loss across heterogeneous datasets
* Convenient dataset downloading/unpacking to `~/.pugh_torch/datasets/`.
    * You can override this via the ENV variable `ROOT_DATASET_PATH`.

## Installation
**Stable Release:** `pip install pugh_torch`<br>
**Development Head:** `pip install git+https://github.com/BrianPugh/pugh_torch.git`

## Experiments
A big part of this repo is a framework to quickly be able to iterate on ideas.

To accomplish this, we provide the following:
* A docker container `brianpugh/pugh_torch` that contains many dependencies
  experimenters would like to use.
    * You can pull the docker image and launch the container via:
        ```
        docker pull brianpugh/pugh_torch
        ./docker_run.sh
        ```
    * This will map `~/.pugh_torch` and the local copy of the git repo 
      into the container. You may change this if you like.
      This will also pass in any available GPUs and set other common
      docker flags for running/training neural nets.
    * This container runs a VNC server, incase you need to perform some visual
      actions, like using `matplotlib.pyplot`
* A unified training driver `experiments/train.py` to run experiments.
    * From the `experiments/` folder, run `python3 train.py template` to begin
      training the default resnet50 architecture on ImageNet.
    * ImageNet cannot be automatically downloaded (see the error raised). To
      get training started with an easier-to-obtain dataset, run:
          ```
          python3 train.py template dataset=cifar100 model=cifar100
          ```
* A template project `experiments/template` that should get you going. The goal
  here is to  provide maximum flexibility while minimizing "project startup
  costs". We leverage the following libraries:
    * [Hydra](https://github.com/facebookresearch/hydra) for managing experiment
      hyperparameters and other configuration. It's a good idea to make your 
      code configurable via this configuration rather than directly tweaking 
      code to make experiments more trackable and reproduceable.
    * [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
      for general project organization and training.
    * `pugh_torch` for various tweaks and helpers that make using the above
      libraries easier for common projects and tasks.

## Documentation
For full package documentation please visit [BrianPugh.github.io/pugh_torch](https://BrianPugh.github.io/pugh_torch).

***Free software: MIT license***

