#pragma once

#include <Eigen/Core>
#include <limits>

namespace box_pose_control {

// 控制参数统一使用米、秒；目标是以相机为原点的底盘水平面坐标，不是表面距离。
struct ControlParameters {
  double target_distance_m = 0.30;     // 水平面前方目标距离，m
  double target_lateral_m = 0.0;      // 水平面左侧目标偏移，m（负值向右）
  double camera_yaw_deg = 0.0;        // 相机光轴相对车头的水平偏角，左偏为正，度
  double lateral_gain = 2.0;          // 横向位置误差到期望平移速度的增益，s⁻¹
  double forward_gain = 4.0;          // 水平前后误差到平移速度的增益，s⁻¹
  double lateral_tolerance_m = 0.02;  // 左右误差进入此范围后，横向视为已对中
  double distance_tolerance_m = 0.03; // 前后误差进入此范围后，距离视为已到位
  double lateral_resume_m = 0.03;     // 已对中后，超过此误差才恢复横移
  double distance_resume_m = 0.045;    // 已到位后，超过此误差才恢复前后调整
  double max_forward_speed = 0.50;    // 底盘前后速度上限，m/s
  bool yaw_enabled = true;
  double yaw_tolerance_deg = 5.0;
  double yaw_resume_deg = 8.0;
  double yaw_gain = 1.2;
  double max_yaw_speed = 0.4;         // rad/s，左转为正
  double far_max_yaw_speed = 0.2;
  double yaw_fine_distance_m = 0.7;   // 方块中心距相机的水平距离
  double yaw_unreliable_approach_speed = 0.15;
  double yaw_jump_deg = 8.0;          // 扣除允许运动后，相邻观测的角度变化余量
  int yaw_confirm_frames = 3;
  double max_lateral_speed = 0.50;    // 底盘左右速度上限，m/s
};

enum class ControlState {
  Aligning,          // 仅横向尚未到位，调整横移
  AdjustingDistance, // 仅前后尚未到位，调整距离
  AdjustingPosition, // 横向和前后均未到位，同时调整 XY
  InPosition,       // 所有启用的轴都在保持范围内，速度为零
  AligningYaw,      // XY 已到位，仍在调整朝向
  WaitingYaw,       // 朝向观测尚未稳定，禁止夹取
  InvalidGeometry   // UP 或安装方向无法用于平面控制，速度为零
};

// 底盘坐标系速度：vx 向前、vy 向左、wz 左转；线速度 m/s，角速度 rad/s。
struct VelocityCommand {
  double vx = 0.0;
  double vy = 0.0;
  ControlState state = ControlState::InPosition;
  double wz = 0.0;
  double yaw_error_rad = std::numeric_limits<double>::quiet_NaN();
  bool yaw_ready = false;
};

// 初始化时集中检查参数取值及“恢复门限 >= 停车容差”的关系。
void validate_parameters(const ControlParameters &parameters);

// 将完整光学坐标投影到水平面并补偿安装偏航，返回底盘 (前方, 左侧)。
// 原点仍是相机，不加入安装平移；忽略沿 UP 的高度。
// 光轴几乎竖直时无法可靠确定底盘前方，返回 false。
bool project_to_horizontal(const Eigen::Vector3d &center, const Eigen::Vector3d &up,
                           Eigen::Vector2d &position, double camera_yaw_deg = 0.0);

// 从完整旋转姿态提取相对底盘的最近方块面角差，范围 [-45°,45°)。
// 选择最接近水平的方块轴，以兼容等价的立方体轴置换。
bool box_yaw_error(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &up,
                   double camera_yaw_deg, double &error);

// 纯控制计算，不持有 ROS 节点或串口；观测是否新鲜、目标 ID 是否有效由节点负责。
class Controller {
 public:
  explicit Controller(const ControlParameters &parameters);

  // center 是光学坐标中的方块中心（m），up 是同一坐标系中的竖直向上向量。
  // 此双参数入口只计算 XY，供平移计算和回归测试使用；完整控制使用下方入口。
  VelocityCommand compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up);
  // 完整闭环入口；stamp_s 必须是观测源时间，同一帧重复计算不累加确认次数。
  VelocityCommand compute(const Eigen::Vector3d &center, const Eigen::Vector3d &up,
                          const Eigen::Matrix3d &rotation, double stamp_s);
  void reset();                              // 丢失目标或启停时清除上轮的到位记忆
  void set_target_distance(double distance); // 使用已由节点校验的新目标，并重置到位记忆
  void set_target(double distance, double lateral);

 private:
  ControlParameters parameters_;
  // 保留上次是否到位，才能区分“首次进入容差”和“到位后尚未越过恢复门限”。
  bool yaw_aligned_ = false;
  int yaw_confirmed_ = 0;
  double yaw_stamp_ = -1.0, previous_yaw_ = 0.0;
  bool lateral_aligned_ = false;
  bool distance_in_position_ = false;
};

}  // namespace box_pose_control
