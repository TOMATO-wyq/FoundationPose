// 小矩阵几何一次调用内完成，减少候选循环中的 NumPy 临时数组。
#include "core_ops.hpp"

#include <pybind11/numpy.h>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> order_quad(const Array& points) {
    if (points.ndim() != 2 || points.shape(0) != 4 || points.shape(1) != 2)
        throw std::invalid_argument("expected points[4,2]");
    py::array_t<double> result({4, 2});
    const auto* p = points.data();
    auto* out = result.mutable_data();
    {
        py::gil_scoped_release release;
        std::array<int, 4> order{0, 1, 2, 3};
        std::array<double, 4> angles{};
        double x = 0, y = 0;
        for (int i = 0; i < 4; ++i) {
            if (!std::isfinite(p[2*i]) || !std::isfinite(p[2*i+1]))
                throw std::invalid_argument("invalid quadrilateral");
            x += p[2*i]; y += p[2*i+1];
            for (int j = 0; j < i; ++j)
                if (p[2*i] == p[2*j] && p[2*i+1] == p[2*j+1])
                    throw std::invalid_argument("invalid quadrilateral");
        }
        for (int i = 0; i < 4; ++i) angles[i] = std::atan2(p[2*i+1]-y/4, p[2*i]-x/4);
        std::stable_sort(order.begin(), order.end(), [&angles](int a, int b) { return angles[a] < angles[b]; });
        int first = 0;
        for (int i = 1; i < 4; ++i)
            if (p[2*order[i]] + p[2*order[i]+1] < p[2*order[first]] + p[2*order[first]+1]) first = i;
        std::vector<cv::Point2f> contour;
        contour.reserve(4);
        for (int i = 0; i < 4; ++i) {
            const int source = order[(first+i)%4];
            out[2*i] = p[2*source]; out[2*i+1] = p[2*source+1];
            contour.emplace_back(static_cast<float>(out[2*i]), static_cast<float>(out[2*i+1]));
        }
        if (!cv::isContourConvex(contour) || cv::contourArea(contour) < 1)
            throw std::invalid_argument("nonconvex or degenerate quadrilateral");
    }
    return result;
}

py::array_t<double> rigid_fit(const Array& source, const Array& target) {
    if (source.ndim() != 2 || target.ndim() != 2 || source.shape(1) != 3 || target.shape(1) != 3 ||
        source.shape(0) != target.shape(0) || source.shape(0) < 1)
        throw std::invalid_argument("expected matching source[N,3] and target[N,3]");
    py::array_t<double> result({4, 4});
    auto* out = result.mutable_data();
    const auto count = source.shape(0);
    const auto* a = source.data(); const auto* b = target.data();
    {
        py::gil_scoped_release release;
        Eigen::Vector3d ac = Eigen::Vector3d::Zero(), bc = Eigen::Vector3d::Zero();
        for (py::ssize_t i = 0; i < count; ++i) {
            const Eigen::Map<const Eigen::Vector3d> pa(a+3*i), pb(b+3*i);
            if (!pa.allFinite() || !pb.allFinite()) throw std::invalid_argument("nonfinite rigid fit input");
            ac += pa; bc += pb;
        }
        ac /= count; bc /= count;
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (py::ssize_t i = 0; i < count; ++i)
            covariance.noalias() += (Eigen::Map<const Eigen::Vector3d>(a+3*i)-ac) *
                                   (Eigen::Map<const Eigen::Vector3d>(b+3*i)-bc).transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d v = svd.matrixV();
        Eigen::Matrix3d rotation = v * svd.matrixU().transpose();
        if (rotation.determinant() < 0) {
            v.col(2) *= -1;
            rotation.noalias() = v * svd.matrixU().transpose();
        }
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.topLeftCorner<3,3>() = rotation;
        pose.topRightCorner<3,1>() = bc-rotation*ac;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) out[4*y+x] = pose(y,x);
    }
    return result;
}
}  // namespace

void bind_core_ops(py::module_& module) {
    module.def("order_quad", &order_quad, py::arg("points"));
    module.def("rigid_fit", &rigid_fit, py::arg("source"), py::arg("target"));
}
