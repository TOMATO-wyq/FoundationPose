// CPU 像素循环。阈值和像素归属与 NumPy 参考路径完全一致。
#include "image_ops.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using HSV = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using Depth = py::array_t<float, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<std::int8_t> classify_hsv(
    const HSV& hsv, int min_saturation, int min_value,
    const std::vector<std::array<int, 2>>& red_ranges, const std::array<int, 2>& blue_range) {
    if (hsv.ndim() < 1 || hsv.shape(hsv.ndim() - 1) != 3)
        throw std::invalid_argument("expected HSV array with final dimension 3");
    std::vector<py::ssize_t> shape(hsv.shape(), hsv.shape() + hsv.ndim() - 1);
    py::array_t<std::int8_t> labels(shape);
    const auto* pixels = hsv.data();
    auto* result = labels.mutable_data();
    const auto count = labels.size();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < count; ++i) {
            const auto hue = pixels[3 * i];
            const bool colored = pixels[3 * i + 1] >= min_saturation && pixels[3 * i + 2] >= min_value;
            bool red = false;
            for (const auto& interval : red_ranges)
                red = red || (hue >= interval[0] && hue <= interval[1]);
            const bool blue = hue >= blue_range[0] && hue <= blue_range[1];
            // 用户将红/蓝阈值设为重叠时，交集仍然是未知颜色。
            result[i] = colored && red && !blue ? 1 : colored && blue && !red ? -1 : 0;
        }
    }
    return labels;
}

py::array_t<std::uint8_t> depth_edges(const Depth& depth, double min_depth, double max_depth,
                                    double threshold) {
    if (depth.ndim() != 2) throw std::invalid_argument("expected depth[H,W]");
    const auto height = depth.shape(0), width = depth.shape(1);
    py::array_t<std::uint8_t> edges({height, width});
    const float* z = depth.data();
    auto* result = edges.mutable_data();
    // NumPy compares a float32 array with these scalar thresholds in float32.
    const float minimum = static_cast<float>(min_depth), maximum = static_cast<float>(max_depth);
    const float jump = static_cast<float>(threshold);
    {
        py::gil_scoped_release release;
        auto valid = [minimum, maximum](float value) {
            return std::isfinite(value) && value >= minimum && value <= maximum;
        };
        for (py::ssize_t y = 0; y < height; ++y) {
            for (py::ssize_t x = 0; x < width; ++x) {
                const auto i = y * width + x;
                const float current = z[i];
                bool edge = false;
                if (valid(current)) {
                    // 每个跳变只标右侧/下侧的像素；不跨越缺失深度造边。
                    if (x > 0 && valid(z[i - 1])) edge = std::abs(current - z[i - 1]) > jump;
                    if (y > 0 && valid(z[i - width])) edge = edge || std::abs(current - z[i - width]) > jump;
                }
                result[i] = edge ? 255 : 0;
            }
        }
    }
    return edges;
}

py::array_t<std::uint8_t> dense_plane_mask(const Depth& depth, const Depth& rays,
                                         const DoubleArray& normal, double offset, double min_depth,
                                         double max_depth, double threshold) {
    if (depth.ndim() != 2 || rays.ndim() != 3 || rays.shape(2) != 3 ||
        rays.shape(0) != depth.shape(0) || rays.shape(1) != depth.shape(1) ||
        normal.ndim() != 1 || normal.shape(0) != 3)
        throw std::invalid_argument("expected depth[H,W], rays[H,W,3] and normal[3]");
    const auto height = depth.shape(0), width = depth.shape(1);
    py::array_t<std::uint8_t> mask({height, width});
    const auto* depth_data = depth.data();
    const auto* ray_data = rays.data();
    const auto* n = normal.data();
    auto* result = mask.mutable_data();
    const auto count = depth.size();
    const float minimum = static_cast<float>(min_depth), maximum = static_cast<float>(max_depth);
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < count; ++i) {
            const float z = depth_data[i];
            if (!std::isfinite(z) || z < minimum || z > maximum) {
                result[i] = 0;
                continue;
            }
            // 先在 float32 中生成 XYZ，再按原 einsum 的 double 精度累计。
            const float px = ray_data[3 * i] * z;
            const float py = ray_data[3 * i + 1] * z;
            const float pz = ray_data[3 * i + 2] * z;
            const double x = static_cast<double>(px) * n[0];
            const double y = static_cast<double>(py) * n[1];
            const double zz = static_cast<double>(pz) * n[2];
            // NumPy 的三项 einsum 累计次序是 (x + z) + y，阈值为严格小于。
            const double residual = std::abs(((x + zz) + y) + offset);
            result[i] = residual < threshold ? 255 : 0;
        }
    }
    return mask;
}
}  // namespace

void bind_image_ops(py::module_& module) {
    module.def("classify_hsv", &classify_hsv, py::arg("hsv"), py::arg("min_saturation"),
               py::arg("min_value"), py::arg("red_ranges"), py::arg("blue_range"));
    module.def("depth_edges", &depth_edges, py::arg("depth"), py::arg("min_depth"),
               py::arg("max_depth"), py::arg("threshold"));
    module.def("dense_plane_mask", &dense_plane_mask, py::arg("depth"), py::arg("rays"),
               py::arg("normal"), py::arg("d"), py::arg("min_depth"), py::arg("max_depth"),
               py::arg("threshold"));
}
