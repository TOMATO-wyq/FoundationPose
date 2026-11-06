#pragma once

#include <pybind11/pybind11.h>

// Image kernels share the geometry extension so selecting a backend is atomic.
void bind_image_ops(pybind11::module_& module);
