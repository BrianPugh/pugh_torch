# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - TBD
### Added
* `weight` argument to hetero-cross-entropy loss
* google drive model support
* `adabelief` optimizers
* configurable learning rate schedulers via `get_scheduler`

## [0.4.0] - 2020-10-14
### Added
* pytorch-lightning callbacks (TensorBoardAddSS, TensorBoardAddClassification)
  for add_ss and add_rgb for segmentation and classification tasks, respectively.
* Initial form of a project template to get ideas going quickly.
* ADE20K dataset
* various optimizers and getters
* various activation functions and getters
* LoadStateDictMixin that adds verbosity to model loading and has more laxed
  `strict` shape requirements.
* pretrained resnet models (from torchvision) that utilize LoadStateDictMixin
* Label smoothing losses

## [0.3.1] - 2020-09-21
### Added
* Aliased `ResizeShortest` to `ShortestMaxSize` to be consistent with `albumentations.augmentations.transforms.LongestMaxSize`

### Fixed
* Add missing interpolation attribute in ResizeShortest transform.
* Fixed `ResizeShortest` producing erroenous results when both sides are the same length.

## [0.3.0] - 2020-09-21
### Added
+ Text label adding to TensorBoard Images
+ ResizeShortest augmentation transform
+ Unit Testing utilities
+ basic Datasets API
+ A bunch of useful dependencies added.

## [0.2.0] - 2020-09-15
### Added
* Additional extra_requires in preparation for docker release.

## [0.1.0] - 2020-09-13
### Added
* Initial Release
