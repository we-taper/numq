# X-Py routines for quantum computation

This project accelerates the simulation of quantum circuits through two backends: NumPy for CPU, and CuPy for GPU (CUDA based).

## Setup CuPy

Prepare CUDA: `conda install -c anaconda cudatoolkit>=11.0`

Depending on CUDA's version, install different packages. Example:

CUDA v11.0: `pip install cupy-cuda110`.
