# DarkCLR

Github repository for DarkCLR, a framework for the detection of semi-visible jets.\\
This is the reference codebase for:

**Semi-visible jets, energy-based models, and self-supervision**
arXiv:2312.03067
Favaro L., Kr\"amer M., Modak T., Plehn T., R\"uschkamp J.

Usage:
`TransformerEncoder.py` includes the transformer class\\
`contrastive_losses.py` defines the CLR loss function\\
`jet_augmentations.py` contains the physical and anomalous jet augmentations\\
We provide an exmaple run script with the set of default parameters in ```run_scripts```.


Related refs.
Dillon B. et al.  
**Anomalies, Representations, and Self-Supervision**  
arXiv:2301.04660 

Dillon B. et al.
**Symmetry, Safety, and Self-supervision**
arXiv:2108.04253
