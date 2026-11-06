// CPU 数值内核。Python 负责流程，C++ 负责点循环；不使用 CUDA/OpenMP 线程池。
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "image_ops.hpp"
#include "face_ops.hpp"
#include "core_ops.hpp"
#include "refinement_ops.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using Image = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Vec3 = Eigen::Vector3d;

py::tuple fit_plane(const Array& points, const Indices& ids, double threshold, int minimum) {
    // 只在跨语言入口检查数组契约；退化数据仍属于正常的几何拒绝。
    if (points.ndim() != 2 || points.shape(1) != 3 || ids.ndim() != 2 || ids.shape(1) != 3)
        throw std::invalid_argument("expected points[N,3] and sample indices[T,3]");
    const auto count = points.shape(0);
    if (minimum < 3 || count < minimum || !std::isfinite(threshold) || threshold <= 0)
        throw std::invalid_argument("insufficient valid depth samples or invalid plane threshold");
    const double* data = points.data();
    const auto* samples = ids.data();
    Vec3 normal, center;
    double offset = 0, rms = 0, support = 0;
    {
        py::gil_scoped_release release;
        std::vector<Vec3> p;
        p.reserve(count);
        for (py::ssize_t i = 0; i < count; ++i) {
            Vec3 point(data[3*i], data[3*i+1], data[3*i+2]);
            if (!point.allFinite() || point.z() <= 0)
                throw std::invalid_argument("native plane input must contain valid positive depth");
            p.push_back(point);
        }
        // Python 传入同一组固定随机采样，避免换实现时同时更换 RANSAC 假设。
        std::vector<unsigned char> mask(count);
        py::ssize_t best_count = -1;
        for (py::ssize_t trial = 0; trial < ids.shape(0); ++trial) {
            auto a = samples[3*trial], b = samples[3*trial+1], c = samples[3*trial+2];
            if (a < 0 || b < 0 || c < 0 || a >= count || b >= count || c >= count)
                throw std::invalid_argument("plane sample index outside point array");
            Vec3 n = (p[b]-p[a]).cross(p[c]-p[a]);
            double length = n.norm();
            if (length <= 1e-9) continue;
            n /= length;
            double d = -n.dot(p[a]);
            py::ssize_t inliers = 0;
            for (const auto& point : p) inliers += std::abs(point.dot(n)+d) < threshold;
            if (inliers > best_count) {
                best_count = inliers;
                for (py::ssize_t i = 0; i < count; ++i)
                    mask[i] = std::abs(p[i].dot(n)+d) < threshold;
            }
        }
        if (best_count < 0) throw std::invalid_argument("degenerate plane samples");
        // 与 NumPy 参考实现一致：三轮内点协方差重估，不增加新的平整度门槛。
        for (int iteration = 0; iteration < 3; ++iteration) {
            center.setZero();
            py::ssize_t selected = 0;
            for (py::ssize_t i = 0; i < count; ++i) {
                if (mask[i]) { center += p[i]; ++selected; }
            }
            if (selected < minimum) throw std::invalid_argument("plane has insufficient inliers");
            center /= static_cast<double>(selected);
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (py::ssize_t i = 0; i < count; ++i) {
                if (mask[i]) {
                    Vec3 q = p[i]-center;
                    covariance.noalias() += q*q.transpose();
                }
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance/static_cast<double>(selected));
            if (eigen.eigenvalues()[1] < 1e-9) throw std::invalid_argument("collinear plane samples");
            normal = eigen.eigenvectors().col(0);
            offset = -normal.dot(center);
            for (py::ssize_t i = 0; i < count; ++i)
                mask[i] = std::abs(p[i].dot(normal)+offset) < threshold;
        }
        if (normal.dot(center) > 0) { normal = -normal; offset = -offset; }
        py::ssize_t selected = 0;
        double squares = 0;
        for (py::ssize_t i = 0; i < count; ++i) {
            if (mask[i]) {
                double r = p[i].dot(normal)+offset;
                squares += r*r;
                ++selected;
            }
        }
        rms = std::sqrt(squares/static_cast<double>(selected));
        support = static_cast<double>(selected)/static_cast<double>(count);
    }
    py::array_t<double> result(3);
    std::copy(normal.data(), normal.data()+3, result.mutable_data());
    return py::make_tuple(result, offset, rms, support);
}

py::tuple edge_support(const Array& quad, const Image& distance) {
    if (quad.ndim() != 2 || quad.shape(0) != 4 || quad.shape(1) != 2 || distance.ndim() != 2)
        throw std::invalid_argument("expected quad[4,2] and distance[H,W]");
    const double* q = quad.data();
    for (int i = 0; i < 8; ++i)
        if (!std::isfinite(q[i])) throw std::invalid_argument("quad must be finite");
    const auto width = distance.shape(1), height = distance.shape(0);
    const float* pixels = distance.data();
    std::array<double, 4> scores{};
    {
        py::gil_scoped_release release;
        for (int side = 0; side < 4; ++side) {
            int next = (side+1)%4;
            double x = q[2*side], y = q[2*side+1];
            double dx = q[2*next]-x, dy = q[2*next+1]-y;
            int count = static_cast<int>(std::clamp(std::hypot(dx,dy)/3., 12., 60.));
            int inside = 0, supported = 0;
            for (int i = 0; i < count; ++i) {
                double t = (i == count-1) ? .92 : .08+i*(.92-.08)/(count-1);
                double u = std::nearbyint(x+t*dx), v = std::nearbyint(y+t*dy);
                if (u < 0 || v < 0 || u >= width || v >= height) continue;
                ++inside;
                supported += pixels[static_cast<py::ssize_t>(v)*width+static_cast<py::ssize_t>(u)] <= 3.5f;
            }
            scores[side] = inside ? static_cast<double>(supported)/inside : 0.;
        }
    }
    py::list sides;
    double total = 0;
    for (double score : scores) { sides.append(score); total += score; }
    return py::make_tuple(total/4., sides);
}

PYBIND11_MODULE(_geometry_native, module) {
    module.doc() = "Box4D CPU image and face geometry (Eigen/OpenCV, no CUDA)";
    module.attr("api_version") = 4;
    // cv2 wheel 与此扩展可能链接两份 OpenCV；各自都必须限制 CPU 线程。
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
    module.attr("opencv_version") = CV_VERSION;
    module.def("configure_opencv", [](int threads) { cv::setNumThreads(threads); });
    module.def("opencv_num_threads", []() { return cv::getNumThreads(); });
    module.def("fit_plane", &fit_plane, py::arg("points"), py::arg("sample_indices"),
               py::arg("threshold"), py::arg("min_points"));
    module.def("edge_support", &edge_support, py::arg("quad"), py::arg("distance"));
    bind_image_ops(module);
    bind_face_ops(module);
    bind_core_ops(module);
    bind_refinement_ops(module);
}
