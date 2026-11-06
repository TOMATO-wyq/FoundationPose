// Real PCL algorithms, exposed to the existing Python RGB-D pipeline.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/pcl_config.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace py = pybind11;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Input = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Clock = std::chrono::steady_clock;

struct PlaneResult {
  Eigen::Vector4f coefficients;
  double rms = 0.;
  std::vector<int> indices;
};

static PlaneResult describe(const Cloud& cloud, const pcl::ModelCoefficients& model,
                            const std::vector<int>& indices) {
  PlaneResult result;
  result.coefficients = Eigen::Vector4f(model.values[0], model.values[1], model.values[2], model.values[3]);
  const float length = result.coefficients.head<3>().norm();
  if (!std::isfinite(length) || length < 1e-8f || indices.empty())
    throw std::runtime_error("PCL returned a degenerate plane");
  result.coefficients /= length;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  double squared = 0.;
  for (int index : indices) {
    const auto& point = cloud[index];
    center += point.getVector3fMap();
    const double residual = result.coefficients.head<3>().dot(point.getVector3fMap()) + result.coefficients[3];
    squared += residual * residual;
  }
  center /= static_cast<float>(indices.size());
  if (result.coefficients.head<3>().dot(center) > 0.f)
    result.coefficients *= -1.f;
  result.rms = std::sqrt(squared / indices.size());
  result.indices = indices;
  return result;
}

static py::dict segment(Input xyz, int min_inliers, double distance,
                        double angular_degrees, double smoothing, double max_curvature,
                        double depth_change_factor, bool covariance, bool capture_normals) {
  const auto view = xyz.request();
  if (view.ndim != 3 || view.shape[2] != 3 || view.shape[0] < 8 || view.shape[1] < 8)
    throw std::invalid_argument("organized XYZ must have shape H,W,3 with H,W >=8");
  if (min_inliers < 3 || distance <= 0 || angular_degrees <= 0 || angular_degrees >= 90 ||
      smoothing < 3 || max_curvature <= 0 || depth_change_factor <= 0)
    throw std::invalid_argument("invalid PCL segmentation options");
  const py::ssize_t height = view.shape[0], width = view.shape[1];
  Cloud::Ptr cloud(new Cloud);
  cloud->width = static_cast<std::uint32_t>(width);
  cloud->height = static_cast<std::uint32_t>(height);
  cloud->is_dense = false;
  cloud->resize(height * width);
  const float* input = static_cast<const float*>(view.ptr);
  int valid_points = 0;
  for (std::size_t i = 0; i < cloud->size(); ++i) {
    auto& point = (*cloud)[i];
    point.x = input[3*i]; point.y = input[3*i+1]; point.z = input[3*i+2];
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) && point.z > 0)
      ++valid_points;
    else
      point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
  }
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  std::vector<PlaneResult> planes;
  double normal_ms = 0., segmentation_ms = 0.;
  {
    py::gil_scoped_release release;
    if (valid_points >= min_inliers) {
      auto begin = Clock::now();
      pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> estimator;
      estimator.setNormalEstimationMethod(covariance ? estimator.COVARIANCE_MATRIX : estimator.AVERAGE_3D_GRADIENT);
      estimator.setMaxDepthChangeFactor(static_cast<float>(depth_change_factor));
      estimator.setNormalSmoothingSize(static_cast<float>(std::min(smoothing, static_cast<double>(std::min(height, width) - 2))));
      estimator.setDepthDependentSmoothing(false);
      estimator.setInputCloud(cloud);
      estimator.compute(*normals);
      auto normal_end = Clock::now();
      normal_ms = std::chrono::duration<double, std::milli>(normal_end - begin).count();
      pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZ, pcl::Normal, pcl::Label> segmentation;
      segmentation.setMinInliers(static_cast<unsigned>(min_inliers));
      segmentation.setAngularThreshold(angular_degrees * 3.14159265358979323846 / 180.);
      segmentation.setDistanceThreshold(distance);
      segmentation.setMaximumCurvature(max_curvature);
      segmentation.setProjectPoints(false);
      segmentation.setInputCloud(cloud);
      segmentation.setInputNormals(normals);
      std::vector<pcl::PlanarRegion<pcl::PointXYZ>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZ>>> regions;
      std::vector<pcl::ModelCoefficients> models;
      std::vector<pcl::PointIndices> inliers, label_indices, boundary_indices;
      pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>);
      segmentation.segmentAndRefine(regions, models, inliers, labels, label_indices, boundary_indices);
      for (std::size_t i = 0; i < models.size(); ++i)
        if (models[i].values.size() >= 4 && !inliers[i].indices.empty())
          planes.push_back(describe(*cloud, models[i], inliers[i].indices));
      segmentation_ms = std::chrono::duration<double, std::milli>(Clock::now() - normal_end).count();
    }
  }
  py::array_t<std::int32_t> labels({height, width});
  auto* labels_data = labels.mutable_data();
  std::fill(labels_data, labels_data + height*width, -1);
  py::list descriptions;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    const auto& plane = planes[i];
    for (int index : plane.indices) labels_data[index] = static_cast<std::int32_t>(i);
    py::dict description;
    description["id"] = i;
    description["normal"] = py::make_tuple(plane.coefficients[0], plane.coefficients[1], plane.coefficients[2]);
    description["d"] = plane.coefficients[3];
    description["rmse_m"] = plane.rms;
    description["inlier_count"] = plane.indices.size();
    descriptions.append(description);
  }
  py::dict result;
  result["labels"] = labels;
  result["planes"] = descriptions;
  result["normal_ms"] = normal_ms;
  result["segmentation_ms"] = segmentation_ms;
  result["valid_points"] = valid_points;
  if (capture_normals) {
    py::array_t<float> image({height, width, py::ssize_t(3)});
    auto* data = image.mutable_data();
    std::fill(data, data + height*width*3, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i = 0; i < normals->size(); ++i)
      for (int axis = 0; axis < 3; ++axis) data[i*3+axis] = (*normals)[i].normal[axis];
    result["normals"] = image;
  }
  return result;
}

static py::dict fit(Input points, double threshold, int min_points, int iterations) {
  const auto view = points.request();
  if (view.ndim != 2 || view.shape[1] != 3 || threshold <= 0 || min_points < 3 || iterations < 1)
    throw std::invalid_argument("invalid PCL plane fit input/options");
  Cloud::Ptr cloud(new Cloud);
  const float* data = static_cast<const float*>(view.ptr);
  for (py::ssize_t i = 0; i < view.shape[0]; ++i)
    if (std::isfinite(data[3*i]) && std::isfinite(data[3*i+1]) && std::isfinite(data[3*i+2]) && data[3*i+2] > 0)
      cloud->push_back(pcl::PointXYZ(data[3*i], data[3*i+1], data[3*i+2]));
  if (cloud->size() < static_cast<std::size_t>(min_points))
    throw std::invalid_argument("insufficient valid depth samples");
  PlaneResult plane;
  {
    py::gil_scoped_release release;
    pcl::SACSegmentation<pcl::PointXYZ> segmentation(false);
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(threshold);
    segmentation.setMaxIterations(iterations);
    segmentation.setProbability(.995);
    segmentation.setInputCloud(cloud);
    pcl::PointIndices inliers;
    pcl::ModelCoefficients coefficients;
    segmentation.segment(inliers, coefficients);
    if (inliers.indices.size() < static_cast<std::size_t>(min_points) || coefficients.values.size() < 4)
      throw std::invalid_argument("PCL plane has insufficient inliers");
    plane = describe(*cloud, coefficients, inliers.indices);
  }
  py::dict result;
  result["normal"] = py::make_tuple(plane.coefficients[0], plane.coefficients[1], plane.coefficients[2]);
  result["d"] = plane.coefficients[3];
  result["rmse_m"] = plane.rms;
  result["support"] = static_cast<double>(plane.indices.size()) / cloud->size();
  return result;
}

PYBIND11_MODULE(_pcl_backend, module) {
  module.attr("pcl_version") = PCL_VERSION_PRETTY;
  module.def("segment", &segment, py::arg("xyz"), py::arg("min_inliers"), py::arg("distance"),
             py::arg("angular_degrees"), py::arg("smoothing"), py::arg("max_curvature"),
             py::arg("depth_change_factor"), py::arg("covariance") = true, py::arg("capture_normals") = true);
  module.def("fit_plane", &fit, py::arg("points"), py::arg("threshold") = .008,
             py::arg("min_points") = 24, py::arg("iterations") = 128);
}
