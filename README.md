# DarkCLR

Github repository for DarkCLR, a framework for the detection of semi-visible jets.\
This is the reference codebase for:

**Semi-visible jets, energy-based models, and self-supervision**\
arXiv:2312.03067\
Favaro L., Kraemer M., Modak T., Plehn T., Rueschkamp J.

The QCD background and the main ''Aachen'' signal datasets are published on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12801842.svg)](https://doi.org/10.5281/zenodo.12801842) 

Usage:\
`TransformerEncoder.py` includes the transformer class\
`contrastive_losses.py` defines the CLR loss function\
`jet_augmentations.py` contains the physical and anomalous jet augmentations\
We provide an exmaple run script with the set of default parameters in ```run_scripts```.


Related refs.\
Dillon B. et al.\
**Anomalies, Representations, and Self-Supervision**\
arXiv:2301.04660

Dillon B. et al.\
**Symmetries, Safety, and Self-supervision**\
arXiv:2108.04253

Dillon B. et al.\
**A normalized autoencoder for LHC triggers**\
arXiv:2206.14225
