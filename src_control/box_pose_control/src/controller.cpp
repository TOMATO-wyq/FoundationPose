#include "box_pose_control/controller.hpp"

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace box_pose_control {

void validate_parameters(const ControlParameters &parameters) {
  // 这些约束在构造时检查一次，正常控制周期只处理观测与控制计算。
  const double values[] = {
      parameters.target_distance_m, parameters.lateral_gain, parameters.forward_gain,
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
  // ROS 参数回调先统一验证，再更新控制器并重新进入对中/距离判断。
  parameters_.target_distance_m = distance;
  reset();
}

VelocityCommand Controller::compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up) {
  // UP 的长度可能不是 1；归一化后只使用方向，不让加速度模长影响控制速度。
  const double up_norm = up.stableNorm();
  if (!center.allFinite() || !up.allFinite() || !std::isfinite(up_norm) || up_norm < 1e-9) {
    reset();
    return {0.0, 0.0, ControlState::InvalidGeometry};
  }
  const Eigen::Vector3d u = up / up_norm;
  // 光学坐标 x 向右、y 向下、z 向前；UP 给出实际竖直向上方向。
  // 将光轴投影到水平面，得到底盘前进 f；u × f 为底盘左移 l。
  Eigen::Vector3d forward = Eigen::Vector3d::UnitZ() - u * u.z();
  const double forward_norm = forward.norm();
  if (forward_norm < 1e-9) {
    reset();
    return {0.0, 0.0, ControlState::InvalidGeometry};
  }
  forward /= forward_norm;
  const Eigen::Vector3d left = u.cross(forward);
  Eigen::Matrix2d jacobian;
  jacobian << forward.x(), left.x(), forward.z(), left.z();
  const Eigen::JacobiSVD<Eigen::Matrix2d> decomposition(
      jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // 最小奇异值过小意味着某个光学方向很难靠平移改变，求逆会放大速度。
  // 0.25 将这种放大限制在 4 倍以内；退化时停车，不切换成其他前进方向。
  if (decomposition.singularValues().minCoeff() < 0.25) {
    reset();
    return {0.0, 0.0, ControlState::InvalidGeometry};
  }

  const double lateral_error = center.x();
  const double distance_error = center.z() - parameters_.target_distance_m;
  // 只闭环 X/Z；Y 由相机与方块高度、俯仰角共同决定，底盘不能独立调节高度。
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

  Eigen::Vector2d desired_camera_motion = Eigen::Vector2d::Zero();
  ControlState state = ControlState::InPosition;
  // 横向优先：只有已经对中，才允许改变方块在相机中的深度。
  // 向量两个分量分别对应光学 X/Z，不是底盘 vx/vy，后面还需经过 J 转换。
  if (!lateral_aligned_) {
    state = ControlState::Aligning;
    desired_camera_motion.x() = parameters_.lateral_gain * lateral_error;
  } else if (!distance_in_position_) {
    state = ControlState::AdjustingDistance;
    desired_camera_motion.y() = parameters_.forward_gain * distance_error;
  } else {
    return {0.0, 0.0, state};
  }

  // 相机随底盘移动时，静止方块的观测变化为 -J*v。
  // 解 J*v=K*error 即令误差减小；水平相机下右侧方块对应 vy<0（右移）。
  // 横移阶段固定 Z；距离阶段固定 X，不能直接把相机 x/z 当作底盘轴。
  const Eigen::Vector2d velocity = decomposition.solve(desired_camera_motion);
  double scale = 1.0;
  if (std::abs(velocity.x()) > parameters_.max_forward_speed) {
    scale = std::min(scale, parameters_.max_forward_speed / std::abs(velocity.x()));
  }
  if (std::abs(velocity.y()) > parameters_.max_lateral_speed) {
    scale = std::min(scale, parameters_.max_lateral_speed / std::abs(velocity.y()));
  }
  // 两轴共同缩放，保留 J*v 的方向及俯仰/横滚补偿关系。
  return {velocity.x() * scale, velocity.y() * scale, state};
}

}  // namespace box_pose_control
