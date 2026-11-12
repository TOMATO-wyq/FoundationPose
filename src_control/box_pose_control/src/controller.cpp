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
      parameters.max_forward_speed, parameters.max_lateral_speed, parameters.camera_yaw_deg,
      parameters.yaw_tolerance_deg, parameters.yaw_resume_deg, parameters.yaw_gain,
      parameters.max_yaw_speed, parameters.far_max_yaw_speed, parameters.yaw_fine_distance_m,
      parameters.yaw_unreliable_approach_speed, parameters.yaw_jump_deg};
  for (const double value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("control parameters must be finite");
    }
  }
  if (parameters.yaw_tolerance_deg < 0 || parameters.yaw_resume_deg <= parameters.yaw_tolerance_deg ||
      parameters.yaw_resume_deg >= 45 || parameters.yaw_gain <= 0 ||
      parameters.max_yaw_speed < 0.1 || parameters.max_yaw_speed > 1.0 ||
      parameters.far_max_yaw_speed < 0.1 || parameters.far_max_yaw_speed > parameters.max_yaw_speed ||
      parameters.yaw_fine_distance_m <= 0 || parameters.yaw_unreliable_approach_speed <= 0 ||
      parameters.yaw_unreliable_approach_speed > 0.5 || parameters.yaw_jump_deg <= 0 ||
      parameters.yaw_jump_deg >= 45 || parameters.yaw_confirm_frames < 2) {
    throw std::invalid_argument("invalid yaw control limits or hysteresis thresholds");
  }
  if (std::abs(parameters.camera_yaw_deg) > 180.0) {
    throw std::invalid_argument("camera_yaw_deg must be in [-180, 180] degrees");
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
  if (parameters.max_forward_speed <= 0.0 || parameters.max_forward_speed > 0.5 ||
      parameters.max_lateral_speed <= 0.0 || parameters.max_lateral_speed > 0.5) {
    throw std::invalid_argument("axis speed limits must be in (0, 0.5] m/s");
  }
}

Controller::Controller(const ControlParameters &parameters) : parameters_(parameters) {
  validate_parameters(parameters_);
}

void Controller::reset() {
  // 重置后，即使误差落在容差与恢复门限之间，也要重新调整到停车容差内。
  yaw_aligned_ = false;
  yaw_confirmed_ = 0;
  yaw_stamp_ = -1.0;
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
                           Eigen::Vector2d &position, double camera_yaw_deg) {
  // UP 的长度可能不是 1；归一化后只使用方向，不让加速度模长影响控制速度。
  const double up_norm = up.stableNorm();
  if (!center.allFinite() || !up.allFinite() || !std::isfinite(up_norm) || up_norm < 1e-9 ||
      !std::isfinite(camera_yaw_deg) || std::abs(camera_yaw_deg) > 180.0) {
    return false;
  }
  const Eigen::Vector3d u = up / up_norm;
  // 光学坐标 x 向右、y 向下、z 向前；UP 给出实际竖直向上方向。
  // 先构造相机的水平前方 f 和左侧 l，再用安装偏航换算到底盘轴向。
  Eigen::Vector3d forward = Eigen::Vector3d::UnitZ() - u * u.z();
  const double forward_norm = forward.norm();
  if (forward_norm < 0.25) return false;
  forward /= forward_norm;
  const Eigen::Vector3d left = u.cross(forward);
  const double camera_forward = center.dot(forward), camera_left = center.dot(left);
  const double yaw = camera_yaw_deg * std::acos(-1.0) / 180.0;
  const double c = std::cos(yaw), s = std::sin(yaw);
  // 左偏为正：相机正前方的点，在底盘系中同时位于前方和左侧。
  position = {c * camera_forward - s * camera_left, s * camera_forward + c * camera_left};
  return position.allFinite();
}

VelocityCommand Controller::compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up) {
  Eigen::Vector2d position;
  if (!project_to_horizontal(center, up, position, parameters_.camera_yaw_deg)) {
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
  // 两轴独立判断到位，同时按误差计算速度，避免横移/前后阶段来回切换。
  if (!lateral_aligned_) {
    velocity.y() = parameters_.lateral_gain * lateral_error;
  }
  if (!distance_in_position_) {
    velocity.x() = parameters_.forward_gain * distance_error;
  }
  if (!lateral_aligned_ && !distance_in_position_) {
    state = ControlState::AdjustingPosition;
  } else if (!lateral_aligned_) {
    state = ControlState::Aligning;
  } else if (!distance_in_position_) {
    state = ControlState::AdjustingDistance;
  } else {
    return {0.0, 0.0, state};
  }

  // 各轴独立限幅：前后误差较大时，不能把横向指令也按比例压低。
  // 默认前后增益 4、横向增益 2：ex=0.836、ey=0.045 时输出 (0.50, 0.09)。
  return {
    std::clamp(velocity.x(), -parameters_.max_forward_speed, parameters_.max_forward_speed),
    std::clamp(velocity.y(), -parameters_.max_lateral_speed, parameters_.max_lateral_speed), state};
}

namespace {
double wrap_cube(double angle) {
  const double period = std::acos(-1.0) / 2;
  return angle - period * std::floor((angle + period / 2) / period);
}
}

bool box_yaw_error(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &up,
                   double camera_yaw_deg, double &error) {
  if (!rotation.allFinite() || !up.allFinite() || up.stableNorm() < 1e-9 ||
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-5 ||
      std::abs(rotation.determinant() - 1.0) > 1e-5) return false;
  const auto u = up.normalized();
  int axis = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(rotation.col(i).dot(u)) < std::abs(rotation.col(axis).dot(u))) axis = i;
  }
  Eigen::Vector2d direction;
  if (!project_to_horizontal(rotation.col(axis), up, direction, camera_yaw_deg) ||
      direction.norm() < 0.5) return false;
  error = wrap_cube(std::atan2(direction.y(), direction.x()));
  return true;
}

VelocityCommand Controller::compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up,
                                    const Eigen::Matrix3d &rotation, double stamp_s) {
  auto command = compute(center, up);
  if (!parameters_.yaw_enabled || command.state == ControlState::InvalidGeometry) return command;
  Eigen::Vector2d position;
  project_to_horizontal(center, up, position, parameters_.camera_yaw_deg);
  const bool near = position.norm() <= parameters_.yaw_fine_distance_m;
  const double radians = std::acos(-1.0) / 180;
  double error;
  if (!box_yaw_error(rotation, up, parameters_.camera_yaw_deg, error) ||
      !std::isfinite(stamp_s) || stamp_s < 0 || stamp_s < yaw_stamp_) {
    yaw_confirmed_ = 0;
    yaw_stamp_ = -1;
  } else {
    command.yaw_error_rad = error;
    if (stamp_s > yaw_stamp_) {
      const double dt = stamp_s - yaw_stamp_;
      // 允许本车转动造成的角度变化；跳变、长时间断帧后重新确认。
      const bool consistent = yaw_stamp_ >= 0 && dt <= 0.5 &&
        std::abs(wrap_cube(error - previous_yaw_)) <=
          parameters_.yaw_jump_deg * radians + parameters_.max_yaw_speed * dt;
      yaw_confirmed_ = consistent ? std::min(yaw_confirmed_ + 1, parameters_.yaw_confirm_frames) : 1;
      previous_yaw_ = error;
      yaw_stamp_ = stamp_s;
    }
    command.yaw_ready = yaw_confirmed_ >= parameters_.yaw_confirm_frames;
  }
  if (!command.yaw_ready) {
    yaw_aligned_ = false;
    command.state = ControlState::WaitingYaw;
    if (near) command.vx = command.vy = 0;
    else {
      command.vx = std::clamp(command.vx, -parameters_.yaw_unreliable_approach_speed,
                              parameters_.yaw_unreliable_approach_speed);
      command.vy = std::clamp(command.vy, -parameters_.yaw_unreliable_approach_speed,
                              parameters_.yaw_unreliable_approach_speed);
    }
    return command;
  }
  const double absolute_error = std::abs(error);
  if (absolute_error <= parameters_.yaw_tolerance_deg * radians) yaw_aligned_ = true;
  else if (absolute_error > parameters_.yaw_resume_deg * radians) yaw_aligned_ = false;
  if (!yaw_aligned_) {
    const double limit = near ? parameters_.max_yaw_speed : parameters_.far_max_yaw_speed;
    // 串口分辨率为 0.1；容差外至少输出 0.1，避免小角差指令截断后不转。
    command.wz = std::copysign(std::clamp(parameters_.yaw_gain * absolute_error, 0.1, limit), error);
    if (absolute_error > 15 * radians) command.vx *= 0.25;
    if (command.state == ControlState::InPosition) command.state = ControlState::AligningYaw;
  }
  return command;
}

}  // namespace box_pose_control
