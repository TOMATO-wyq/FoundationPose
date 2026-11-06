#pragma once

#include <Eigen/Core>

namespace box_pose_control {

// 控制参数统一使用米、秒；目标距离指方块几何中心的光学 Z，不是前表面距离。
struct ControlParameters {
  double target_distance_m = 0.30;     // 相机前方的目标深度，m
  double lateral_gain = 1.0;          // 横向位置误差到期望平移速度的增益，s⁻¹
  double forward_gain = 1.0;          // 深度误差到期望平移速度的增益，s⁻¹
  double lateral_tolerance_m = 0.01;  // |X| 进入此范围后，横向视为已对中
  double distance_tolerance_m = 0.02; // |Z-目标距离| 进入此范围后，距离视为已到位
  double lateral_resume_m = 0.015;    // 已对中后，超过此误差才恢复横移
  double distance_resume_m = 0.03;    // 已到位后，超过此误差才恢复前后调整
  double max_forward_speed = 0.10;    // 底盘前后速度上限，m/s
  double max_lateral_speed = 0.10;    // 底盘左右速度上限，m/s
};

enum class ControlState {
  Aligning,          // 横移对中，暂不调整光学 Z
  AdjustingDistance, // 已对中，调整光学 Z 并保持光学 X
  InPosition,       // 横向和距离都在保持范围内，速度为零
  InvalidGeometry   // UP 或安装方向无法用于平面控制，速度为零
};

// 底盘坐标系中的速度：vx 正值向前，vy 正值向左；本模块不输出转向速度。
struct VelocityCommand {
  double vx = 0.0;
  double vy = 0.0;
  ControlState state = ControlState::InPosition;
};

// 初始化时集中检查参数取值及“恢复门限 >= 停车容差”的关系。
void validate_parameters(const ControlParameters &parameters);

// 纯控制计算，不持有 ROS 节点或串口；观测是否新鲜、目标 ID 是否有效由节点负责。
class Controller {
 public:
  explicit Controller(const ControlParameters &parameters);

  // center 是光学坐标中的方块中心（m），up 是同一坐标系中的竖直向上向量。
  // 每次计算同时更新迟滞状态，因此需要对同一锁定目标连续调用。
  VelocityCommand compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up);
  void reset();                              // 丢失目标或启停时清除上轮的到位记忆
  void set_target_distance(double distance); // 使用已由节点校验的新目标，并重置到位记忆

 private:
  ControlParameters parameters_;
  // 保留上次是否到位，才能区分“首次进入容差”和“到位后尚未越过恢复门限”。
  bool lateral_aligned_ = false;
  bool distance_in_position_ = false;
};

}  // namespace box_pose_control
