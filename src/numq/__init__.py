import importlib as _importlib

from ._base_function import *
from ._register import get_implementation_module as _get_mod


def load_module(*types):
    for type_ in types:
        _load_module(type_)


def _load_module(name):
    try:
        mod_name = _get_mod(name)
        _importlib.import_module(mod_name)
    except KeyError:
        raise ValueError(f'Unregistered module {name}')

__all__ = [_ for _ in locals() if not _.startswith('')]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
