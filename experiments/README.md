# Intro

This is my pytorch playground. Typically useful, generic pieces of code go into
the package `pugh_torch` while experiment-specific items go into the experiment
subfolder.

Most of my experiments are computer-vision related, so the template is 
geared towards CNN-usage.

# Packages used

This playground uses several packages to keep experiments organized, reproduceable,
and fast.

## PyTorch-Lightning

This is the core training and project layout.

## Hydra

This is the configuration/hyperparameter manager

# Getting started with an experiment

tl;dr copy the `template/` folder into a new directory. This has everything
necessary to be a standalone project.

# Design Rules

These items are here for code consistency and ease-of-debugging.

* If images are involved, the order **will** be RGB.
* If paths are used, they will be pathlib.Path objects until absolutely necessary.
* All networks operate on (B, C, H, W) tensors unless otherwise noted.
* As general-all-purpose portions of code are discovered, add them to the template.
* Usually an existing package does a better job than you can, including this repo.
* When in doubt, cater to the largest audience if it results in less choice 
  for the developer. Its safe to assume the person running this code
  will be running it on a GPU released within the past 6 years. This is already
  a super opinionated repo, so no turning back.
