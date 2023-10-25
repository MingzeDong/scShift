# scShift

scShift is a mechanistic model that learns batch and biological patterns from atlas-level scRNA-seq data as well as perturbation scRNA-seq data. scShift models genes as functions of latent biological processes, with sparse shifts induced by batch effects and biological perturbations, leveraging recent advances of causal representation learning. scShift is able to reveal unified cell type representations and underlying biological variations for query data in zero-shot manners. scShift is implemented by scvi-tools. Therefore, we refer its model architecture (and the package name) as PertVI.

![fig1github](https://github.com/MingzeDong/scShift/assets/68533876/34ada998-a766-4d30-a41e-dd8e906690b7)<?xml version="1.0" encoding="utf-8"?>

Read our preprint on BioRxiv:

## System requirements
### OS requirements
The scShift (pertvi) package is supported for all OS in principle. The package has been tested on the following systems:
* macOS: Monterey (12.4)
* Linux: Ubantu (20.04.5)
### Dependencies
See `setup.cfg` for details.

## Installation
PertVI requires `python` version 3.7+.  Install directly from pip with:

    pip install pertvi

The installation should take no more than a few minutes on a normal desktop computer.


## Usage

For detailed usage, follow our step-by-step tutorial here:

- [Getting Started with SIMVI](https://github.com/MingzeDong/SIMVI/blob/main/SIMVItutorial.ipynb)

Download the model and data used for the tutorial here:

- [Human MERFISH MTG data](https://drive.google.com/file/d/1i6spfxfEqqczgSHDX0gNImrGkH7Ruy7z/view?usp=sharing)
