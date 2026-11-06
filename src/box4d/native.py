"""Optional CPU stages. Missing builds use NumPy; stale builds require rebuilding."""

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def _load():
    try:
        return import_module("box4d._geometry_native")
    except ModuleNotFoundError as error:
        if error.name != "box4d._geometry_native":
            raise
        return None


def geometry_kernel(backend="auto"):
    if backend == "numpy":
        return None
    module = _load()
    if module is None and backend == "cpp":
        raise RuntimeError(
            "C++ geometry kernel is not built. Run bash src/build_box4d_native.sh "
            "with BOX4D_PYTHON set to your runtime Python."
        )
    if module is not None and getattr(module, "api_version", None) != 4:
        raise RuntimeError(
            "C++ extension version does not match this source. Run bash src/build_box4d_native.sh "
            "with BOX4D_PYTHON set to your runtime Python."
        )
    return module


def geometry_backend(backend="auto"):
    return "cpp" if geometry_kernel(backend) is not None else "numpy"
