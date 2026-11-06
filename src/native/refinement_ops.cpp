// 4D 联合优化的一轮残差与解析雅可比；SciPy 仍负责有界稳健求解。
#include "refinement_ops.hpp"

#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Vec3 = Eigen::Vector3d;

struct FaceData {
    std::array<Vec3, 4> objects;
    std::array<cv::Point2d, 4> images;
    std::vector<Vec3> samples;
    Vec3 normal, center;
    double factor;
};

class RefinementEvaluator {
public:
    RefinementEvaluator(const py::sequence& prepared, const Array& upright,
                        const Array& camera, const Array& distortion) {
        if (upright.ndim() != 2 || upright.shape(0) != 3 || upright.shape(1) != 3 ||
            camera.ndim() != 2 || camera.shape(0) != 3 || camera.shape(1) != 3)
            throw std::invalid_argument("expected upright[3,3] and camera[3,3]");
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                camera_from_upright_(x,y) = upright.data()[3*y+x];
                camera_(y,x) = camera.data()[3*y+x];
            }
        }
        if (!camera_from_upright_.allFinite())
            throw std::invalid_argument("upright matrix must be finite");
        if (distortion.size()) {
            distortion_ = cv::Mat(1, static_cast<int>(distortion.size()), CV_64F);
            std::copy(distortion.data(), distortion.data()+distortion.size(), distortion_.ptr<double>());
        }
        faces_.reserve(prepared.size());
        // Copy immutable face data once. No Python buffers or Python calls are used in evaluate's loop.
        for (py::handle item : prepared) {
            const auto fields = py::cast<py::tuple>(item);
            if (fields.size() != 6) throw std::invalid_argument("expected six prepared face fields");
            const auto object = py::cast<Array>(fields[0]);
            const auto image = py::cast<Array>(fields[1]);
            const auto samples = py::cast<Array>(fields[2]);
            const auto normal = py::cast<Array>(fields[3]);
            const auto center = py::cast<Array>(fields[4]);
            if (object.ndim() != 2 || object.shape(0) != 4 || object.shape(1) != 3 ||
                image.ndim() != 2 || image.shape(0) != 4 || image.shape(1) != 2 ||
                samples.ndim() != 2 || samples.shape(1) != 3 || normal.size() != 3 || center.size() != 3)
                throw std::invalid_argument("expected object[4,3], image[4,2], samples[N,3], normal[3], center[3]");
            FaceData face;
            for (int i = 0; i < 4; ++i) {
                face.objects[i] = Eigen::Map<const Vec3>(object.data()+3*i);
                face.images[i] = cv::Point2d(image.data()[2*i], image.data()[2*i+1]);
            }
            face.normal = Eigen::Map<const Vec3>(normal.data());
            face.center = Eigen::Map<const Vec3>(center.data());
            face.factor = py::cast<double>(fields[5]);
            face.samples.reserve(samples.shape(0));
            for (py::ssize_t i = 0; i < samples.shape(0); ++i)
                face.samples.emplace_back(Eigen::Map<const Vec3>(samples.data()+3*i));
            length_ += 8+static_cast<py::ssize_t>(face.samples.size());
            faces_.push_back(std::move(face));
        }
    }

    py::tuple evaluate(const Array& parameters) const {
        if (parameters.ndim() != 1 || parameters.size() != 4)
            throw std::invalid_argument("expected four upright pose parameters");
        py::array_t<double> residual(length_), jacobian({length_, py::ssize_t(4)});
        auto* out = residual.mutable_data();
        auto* jac = jacobian.mutable_data();
        const auto* x = parameters.data();
        {
            py::gil_scoped_release release;
            const double c = std::cos(x[3]), s = std::sin(x[3]);
            Eigen::Matrix3d yaw, derivative;
            yaw << c, -s, 0, s, c, 0, 0, 0, 1;
            derivative << -s, -c, 0, c, -s, 0, 0, 0, 0;
            const Eigen::Matrix3d rotation = camera_from_upright_*yaw;
            const Eigen::Matrix3d d_rotation = camera_from_upright_*derivative;
            const Vec3 translation = camera_from_upright_*Eigen::Map<const Vec3>(x);
            py::ssize_t row = 0;
            for (const auto& face : faces_) {
                std::vector<cv::Point3d> points;
                points.reserve(4);
                bool behind = false;
                for (const auto& object : face.objects) {
                    const Vec3 point = rotation*object+translation;
                    if (point.z() <= 0) {
                        behind = true;
                        break;
                    }
                    points.emplace_back(point.x(), point.y(), point.z());
                }
                if (behind) {
                    // Exactly the same large residual and zero derivative as the reference branch.
                    std::fill(out, out+length_, 1e6);
                    std::fill(jac, jac+4*length_, 0.);
                    break;
                }
                std::vector<cv::Point2d> projected;
                cv::Mat project_jacobian;
                cv::projectPoints(points, cv::Vec3d(0,0,0), cv::Vec3d(0,0,0), camera_, distortion_,
                                  projected, project_jacobian);
                for (int i = 0; i < 4; ++i) {
                    out[row+2*i] = projected[i].x-face.images[i].x;
                    out[row+2*i+1] = projected[i].y-face.images[i].y;
                    const Vec3 yaw_derivative = d_rotation*face.objects[i];
                    for (int coordinate = 0; coordinate < 2; ++coordinate) {
                        const auto* projection = project_jacobian.ptr<double>(2*i+coordinate)+3;
                        for (int parameter = 0; parameter < 3; ++parameter) {
                            jac[4*(row+2*i+coordinate)+parameter] =
                                projection[0]*camera_from_upright_(0,parameter) +
                                projection[1]*camera_from_upright_(1,parameter) +
                                projection[2]*camera_from_upright_(2,parameter);
                        }
                        jac[4*(row+2*i+coordinate)+3] =
                            projection[0]*yaw_derivative.x()+projection[1]*yaw_derivative.y()+
                            projection[2]*yaw_derivative.z();
                    }
                }
                row += 8;
                const Vec3 normal = rotation*face.normal;
                const Vec3 center = rotation*face.center+translation;
                const Eigen::RowVector3d depth_translation = -normal.transpose()*camera_from_upright_*face.factor;
                const Vec3 d_normal = d_rotation*face.normal;
                const double center_derivative = normal.dot(d_rotation*face.center);
                for (const auto& sample : face.samples) {
                    const Vec3 difference = sample-center;
                    out[row] = difference.dot(normal)*face.factor;
                    for (int parameter = 0; parameter < 3; ++parameter)
                        jac[4*row+parameter] = depth_translation[parameter];
                    jac[4*row+3] = (difference.dot(d_normal)-center_derivative)*face.factor;
                    ++row;
                }
            }
        }
        return py::make_tuple(residual, jacobian);
    }

private:
    Eigen::Matrix3d camera_from_upright_;
    cv::Matx33d camera_;
    cv::Mat distortion_;
    std::vector<FaceData> faces_;
    py::ssize_t length_ = 0;
};
}  // namespace

void bind_refinement_ops(py::module_& module) {
    py::class_<RefinementEvaluator>(module, "RefinementEvaluator")
        .def(py::init<const py::sequence&, const Array&, const Array&, const Array&>(),
             py::arg("prepared_data"), py::arg("u_from_c"), py::arg("camera_k"), py::arg("camera_d"))
        .def("evaluate", &RefinementEvaluator::evaluate, py::arg("parameters"));
}
