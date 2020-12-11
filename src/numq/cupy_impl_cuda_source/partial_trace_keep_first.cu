#include <cupy/complex.cuh>

/*
Please see the code and doc for
    mlec.q_toolkit.numpy_impl.partial_trace_wf_keep_first_numpy
for explanation of how this algorithm works.

Also, the return will be C-ordered/row-majored/row-by-row.

Parameters:
- m, n_idx see the numpy code
- iwf, iwf_conj, rho: obvious. Note that rho must be initialised to 0.
- rho_max_length: this would be rho.shape[0] * rho.shape[1]. This is
  used to make this algorithm return once we exceeds the boundary of rho.
*/
extern "C" __global__ void partial_trace_keep_first(
    const complex<float> *iwf, const complex<float> *iwf_conj,
    complex<float> *rho,
    int m, int m_idx, int n_idx
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // i is row idx
    int j = blockIdx.y * blockDim.y + threadIdx.y; // j is column idx
    if ((i >= n_idx) || (j >= n_idx)) {return ;}

    int rho_idx = i * n_idx + j;
    int i_shift = i << m;
    int j_shift = j << m;

    for (int k = 0; k < m_idx; k ++) {
        rho[rho_idx] += iwf[i_shift + k] * iwf_conj[j_shift + k];
    }
}