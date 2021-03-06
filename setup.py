#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("CHANGELOG.md") as f:
    changelog = f.read()


setup_requirements = [
    "pytest-runner>=5.2",
]

requirements = [
    "adabelief-pytorch>=0.0.5",
    "albumentations>=0.4.6",
    "gdown>=3.10.1",
    "h5py>=3.0.0",
    "matplotlib>=3.3.2",
    "numpy>=1.17.1",
    "opencv-python>=4.2.0",
    "packaging>=20.2.4",
    "ranger-adabelief>=0.0.9",
    "scikit-image==0.17.2",
    "tensorboard>=2.2.1",
    "torch>=1.4.0",
    "torchvision>=0.5.0",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8-debugger>=3.2.1",
    "flake8>=3.8.3",
    "pytest-cov>=2.9.0",
    "pytest-helpers-namespace==2019.1.8",
    "pytest-mock>=3.3.1",
    "pytest-raises>=0.11",
    "pytest>=5.4.3",
    "pytorch-lightning>=1.0.0rc4",
]

dev_requirements = [
    "bumpversion>=0.6.0",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r>=0.2.1",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "torchtest==0.5",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

experiment_requirements = [
    "hydra-core==1.0.2",
    "ipdb",
    "open3d>=0.10.0.0",
    "scipy>=1.5.1",
    "tqdm>=4.45.0",
]


extra_requirements = {
    "setup": setup_requirements,
    "test": [
        *setup_requirements,
        *requirements,
        *test_requirements,
    ],
    "dev": [
        *setup_requirements,
        *requirements,
        *test_requirements,
        *dev_requirements,
    ],
    "all": [
        *requirements,
        *setup_requirements,
        *test_requirements,
        *dev_requirements,
        *experiment_requirements,
    ],
}

setup(
    author="Brian Pugh",
    author_email="bnp117@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Functions, losses, and module blocks to share between experiments.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + changelog,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="pugh_torch",
    name="pugh_torch",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="pugh_torch/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/BrianPugh/pugh_torch",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.4.0",
    zip_safe=False,
)
