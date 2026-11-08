#include "box_pose_control/controller.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace box_pose_control {

void validate_parameters(const ControlParameters &parameters) {
  // 这些约束在构造时检查一次，正常控制周期只处理观测与控制计算。
  const double values[] = {
      parameters.target_distance_m, parameters.target_lateral_m, parameters.lateral_gain, parameters.forward_gain,
      parameters.lateral_tolerance_m, parameters.distance_tolerance_m,
      parameters.lateral_resume_m, parameters.distance_resume_m,
      parameters.max_forward_speed, parameters.max_lateral_speed};
  for (const double value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("control parameters must be finite");
    }
  }
  if (parameters.target_distance_m <= 0.0 || parameters.lateral_gain <= 0.0 ||
      parameters.forward_gain <= 0.0) {
    throw std::invalid_argument("target distance and control gains must be positive");
  }
  if (parameters.lateral_tolerance_m < 0.0 || parameters.distance_tolerance_m < 0.0 ||
      parameters.lateral_resume_m < parameters.lateral_tolerance_m ||
      parameters.distance_resume_m < parameters.distance_tolerance_m) {
    throw std::invalid_argument("resume thresholds must be at least the nonnegative tolerances");
  }
  if (parameters.max_forward_speed <= 0.0 || parameters.max_forward_speed > 0.3 ||
      parameters.max_lateral_speed <= 0.0 || parameters.max_lateral_speed > 0.3) {
    throw std::invalid_argument("axis speed limits must be in (0, 0.3] m/s");
  }
}

Controller::Controller(const ControlParameters &parameters) : parameters_(parameters) {
  validate_parameters(parameters_);
}

void Controller::reset() {
  // 重置后，即使误差落在容差与恢复门限之间，也要重新调整到停车容差内。
  lateral_aligned_ = false;
  distance_in_position_ = false;
}

void Controller::set_target_distance(double distance) {
  set_target(distance, parameters_.target_lateral_m);
}

void Controller::set_target(double distance, double lateral) {
  auto candidate = parameters_;
  candidate.target_distance_m = distance;
  candidate.target_lateral_m = lateral;
  validate_parameters(candidate);
  parameters_ = candidate;
  reset();
}

bool project_to_horizontal(const Eigen::Vector3d &center, const Eigen::Vector3d &up,
                           Eigen::Vector2d &position) {
  // UP 的长度可能不是 1；归一化后只使用方向，不让加速度模长影响控制速度。
  const double up_norm = up.stableNorm();
  if (!center.allFinite() || !up.allFinite() || !std::isfinite(up_norm) || up_norm < 1e-9) {
    return false;
  }
  const Eigen::Vector3d u = up / up_norm;
  // 光学坐标 x 向右、y 向下、z 向前；UP 给出实际竖直向上方向。
  // 将光轴投影到水平面，得到底盘前进 f；u × f 为底盘左移 l。
  Eigen::Vector3d forward = Eigen::Vector3d::UnitZ() - u * u.z();
  const double forward_norm = forward.norm();
  if (forward_norm < 0.25) return false;
  forward /= forward_norm;
  const Eigen::Vector3d left = u.cross(forward);
  position = {center.dot(forward), center.dot(left)};
  return position.allFinite();
}

VelocityCommand Controller::compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up) {
  Eigen::Vector2d position;
  if (!project_to_horizontal(center, up, position)) {
    reset();
    return {0.0, 0.0, ControlState::InvalidGeometry};
  }

  const double lateral_error = position.y() - parameters_.target_lateral_m;
  const double distance_error = position.x() - parameters_.target_distance_m;
  // 水平投影消除了高度；同一水平位置的不同高度方块产生相同速度。
  const double abs_lateral = std::abs(lateral_error);
  const double abs_distance = std::abs(distance_error);
  // 进入容差区后，直到越过较大的恢复阈值才重新动作，防止边界抖动。
  if (abs_lateral <= parameters_.lateral_tolerance_m) {
    lateral_aligned_ = true;
  } else if (abs_lateral > parameters_.lateral_resume_m) {
    lateral_aligned_ = false;
  }
  if (abs_distance <= parameters_.distance_tolerance_m) {
    distance_in_position_ = true;
  } else if (abs_distance > parameters_.distance_resume_m) {
    distance_in_position_ = false;
  }

  Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
  ControlState state = ControlState::InPosition;
  // 横向优先；两个分量直接对应底盘前进 vx、左移 vy。
  if (!lateral_aligned_) {
    state = ControlState::Aligning;
    velocity.y() = parameters_.lateral_gain * lateral_error;
  } else if (!distance_in_position_) {
    state = ControlState::AdjustingDistance;
    velocity.x() = parameters_.forward_gain * distance_error;
  } else {
    return {0.0, 0.0, state};
  }

  double scale = 1.0;
  if (std::abs(velocity.x()) > parameters_.max_forward_speed) {
    scale = std::min(scale, parameters_.max_forward_speed / std::abs(velocity.x()));
  }
  if (std::abs(velocity.y()) > parameters_.max_lateral_speed) {
    scale = std::min(scale, parameters_.max_lateral_speed / std::abs(velocity.y()));
  }
  // 保留水平速度方向，并满足两轴限速。
  return {velocity.x() * scale, velocity.y() * scale, state};
}

}  // namespace box_pose_control
