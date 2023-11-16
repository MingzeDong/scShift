# scShift

scShift is a framework that learns batch and biological patterns from atlas-level scRNA-seq data as well as perturbation scRNA-seq data. scShift models genes as functions of latent biological processes, with sparse shifts induced by batch effects and biological perturbations, leveraging recent advances of causal representation learning. scShift is able to reveal unified cell type representations and underlying biological variations for query data in zero-shot manners. scShift is implemented by scvi-tools. scShift not only performs variational inference for large scRNA-seq datasets, but covers the data curation, model interpretation, and zero-shot query. Therefore, we refer its model architecture (and the package name) as a different name (PertVI).

![fig1github](https://github.com/MingzeDong/scShift/assets/68533876/06c7c7bd-1bf0-4736-b113-32dd2cd202e9)

Read our preprint on BioRxiv: [Deep identifiable modeling of single-cell atlases enables zero-shot query of cellular states](https://www.biorxiv.org/content/10.1101/2023.11.11.566161v1)

## System requirements
### OS requirements
The scShift (pertvi) package is supported for all OS in principle. The package has been tested on the following systems:
* macOS: Monterey (12.4)
* Linux: Ubantu (20.04.5)
### Dependencies
See `setup.cfg` for details.

## Installation
scShift (PertVI) requires `python` version 3.7+.  Install directly from pip with:

    pip install pertvi

The installation should take no more than a few minutes on a normal desktop computer.


## Usage

For detailed usage, follow our step-by-step tutorial here:

- [Getting Started with scShift](https://github.com/MingzeDong/scShift/blob/main/scShift_tutorial.ipynb)

Download the model and data used for the tutorial here:

- [Model and datasets used in the tutorial](https://drive.google.com/drive/folders/1sx-c1i_Xp1J__XBD1nmCWc39WsbMwys5?usp=share_link)
