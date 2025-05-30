# X-Py routines for quantum computation

This project accelerates the simulation of quantum circuits through two backends: NumPy for CPU, and CuPy for GPU (CUDA based).

## Setup

This project is hosted in PyPI, and can be installed by `pip install numq`. Note that [Numba][1] will be installed in that process. Numba is a compiler which accelerates native Python codes through translation into machine code in a "just in time" (JIT) manner. It is used to accelerate the for loops in certain partial trace computations.

[1]: https://numba.pydata.org/

**CuPy**

To use Nvidia GPU to accelerate simulations, [CuPy][2] and CUDA are required. CUDA is the standard GPU acceleration API provided by Nvidia. CuPy is a NumPy-like library which leverages CUDA to implement some common computation routines. This project additional implements custom CUDA kernels for computing partial trace.

I highly recommend installing and managing CUDA through the Conda (or miniconda) tools:
* To install CUDA: `conda install -c anaconda cudatoolkit>=11.0`
* Depending on CUDA's version, install the [corresponding CuPy package][3]. For example, for CUDA v11.0, `pip install cupy-cuda110`.

[2]: https://cupy.dev/
[3]: https://docs.cupy.dev/en/stable/install.html#installing-cupy

## Usage

This project can be used in a flexible functional interface. The core routines are centrally declared and can be directly imported from `numq`. These core routines, when applied to data of different types, will choose the appropriate implementation to run. For example, if a quantum state is stored in a CuPy array on GPU, to apply an isometry to it, one needs to: first import the routine `from numq import apply_isometry_to_density_matrices`, then apply it to the array directly `apply_isometry_to_density_matrices(YOUR_ISOMETRY, YOUR_WAVEFUNCTION)`.

Example: 

```python
from numq import make_density_matrix, apply_isometry_to_density_matrices

wf = # your wavefunction
rho = make_density_matrix(wf)
u = # your isometry
rho_after_u = apply_isometry_to_density_matrices(u, rho)
```

Detailed documentations for core routines are available at https://we-taper.github.io/numq/.
