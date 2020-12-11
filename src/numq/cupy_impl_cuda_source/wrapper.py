import os
from pathlib import Path

import cupy
from cupy import complex64

__all__ = ['partial_trace_wf_keep_first_cuda', 'default_dtype']
_here = Path(os.path.dirname(__file__))
_name = 'partial_trace_keep_first.cu'
partial_trace_wf_keep_first_cuda = cupy.RawKernel(
    code=(_here / _name).read_text(),
    name=_name.split('.')[0])

default_dtype = complex64
