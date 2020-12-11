_registered_mods = dict()


def register(name, module):
    """Register a module name, whose implementation is inside the module."""
    if name in _registered_mods:
        raise KeyError(f"Module name {name} has already been registered.")
    from importlib.util import find_spec

    if find_spec(module) is None:
        raise ModuleNotFoundError(f"Module named {module} cannot be found.")

    _registered_mods[name] = module


register("numpy", "numq.numpy_impl")
register("cupy", "numq.cupy_impl")
register("tf", "numq.tf_impl")


def get_implementation_module(name):
    return _registered_mods[name]
