#pragma once

#include <pybind11/pybind11.h>

// Whole face/line stages share one Python boundary rather than one call per edge.
void bind_face_ops(pybind11::module_& module);
