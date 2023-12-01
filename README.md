# Synthetic data, real errors: how (not) to publish and use synthetic data 

This library implements accompanies the paper:
"Synthetic data, real errors: how (not) to publish and use synthetic data", ICML 2023.
https://arxiv.org/abs/2305.09235

The main focus is the proposed **Deep Generative Ensemble (DGE)**, and its comparison to naive synthetic data methods.

## Installation
Create environment and install packages:
```bash
$ conda create -n synthetic_errors python=3.8
$ conda activate synthetic_errors
$ pip install -r requirements.txt
$ pip install .
```
This code uses the generative modelling library of Synthcity (https://github.com/vanderschaarlab/synthcity)

To run the boosting logic, need to install the submodule version of Synthcity:

```bash
conda create -n syn python=3.8
conda activate syn
pip install -r requirements.txt
```

Navigate to the submodule Synthcity repo, then run

```bash
pip install -e .
```

If encounter some version issues or want to use a more updated Synthcity, can do this instead:

```bash
conda create -n syn python=3.8
conda activate syn
pip install -r requirements.txt
pip install synthcity
```

If there's a CUDA error about "symbol cublasLtGetStatusString version libcublasLt.so.11 not defined in file libcublasLt.so.11", it's likely due to pytorch1.13 automaticallying installing CUDA toolkit but you may already have one installed (like Google cloud instance set up), so run:

```bash
pip uninstall nvidia_cublas_cu11
```

## Run experiments

All experiments are provided in the notebook main_experiments.ipynb

## Cite
```
@inproceedings{breugel2023synthetic,
  title={Synthetic data, real errors: how (not) to publish and use synthetic data},
  author={van Breugel, Boris and Qian, Zhaozhi and van der Schaar, Mihaela},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```




