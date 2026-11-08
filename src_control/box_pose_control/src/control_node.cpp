#include "box_pose_control/controller.hpp"
#include "box_pose_control/serial_port.hpp"

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>

namespace box_pose_control {
using Detections = vision_msgs::msg::Detection3DArray;
using Up = geometry_msgs::msg::Vector3Stamped;
using SteadyClock = std::chrono::steady_clock;

// 用图像采集时刻识别“同一帧”，不能用两条 ROS 消息各自到达节点的时间配对。
int64_t stamp_ns(const std_msgs::msg::Header &header) {
  return int64_t(header.stamp.sec) * 1000000000LL + header.stamp.nanosec;
}

// INFO 中保留固定状态名，方便终端检索；对应含义见 Controller 的状态枚举。
const char *state_name(ControlState state) {
  switch (state) {
    case ControlState::Aligning: return "aligning";
    case ControlState::AdjustingDistance: return "adjusting_distance";
    case ControlState::InPosition: return "in_position";
    case ControlState::InvalidGeometry: return "invalid_geometry";
  }
  return "invalid_geometry";
}

// 数据流：位姿/UP 回调 -> 同帧配对与目标锁定 -> 定时计算 -> Twist、串口及 INFO。
// 使用单线程 executor，回调与 tick 顺序执行，共享状态无需另外加锁。
class ControlNode : public rclcpp::Node {
 public:
  ControlNode() : Node("box_pose_control"), parameters_(read_control_parameters()),
      controller_(parameters_) {
    // dry_run 仍完成整套控制计算，只跳过串口打开和写入，便于不开底盘观察。
    dry_run_ = startup_parameter("dry_run", true);
    size_ = startup_parameter("size", std::string("big"));
    if (size_ != "big" && size_ != "small") {
      throw std::invalid_argument("size must be big or small");
    }
    target_size_m_ = size_ == "big" ? 0.35 : 0.20;
    const auto rate = startup_parameter("control_rate", 10.0);
    timeout_ = startup_parameter("observation_timeout_s", 0.5);
    serial_path_ = startup_parameter("serial_port", std::string("/dev/ttyUSB0"));
    baudrate_ = startup_parameter("baudrate", 961200);
    if (!std::isfinite(rate) || rate <= 0 || !std::isfinite(timeout_) || timeout_ <= 0 || baudrate_ <= 0) {
      throw std::invalid_argument("control_rate, observation_timeout_s and baudrate must be positive");
    }
    const auto detections_topic = startup_parameter("detections_topic", std::string("/box4d/detections"));
    const auto up_topic = startup_parameter("up_topic", std::string("/box4d/up_camera"));
    // “~” 展开为节点私有名称，默认输出 /box_pose_control/cmd_vel。
    // 该话题供观察；实车输出直接走本节点串口，不依赖旧 /cmd_vel 转发节点。
    velocity_pub_ = create_publisher<geometry_msgs::msg::Twist>("~/cmd_vel", 10);
    detections_sub_ = create_subscription<Detections>(detections_topic, 10,
        [this](Detections::ConstSharedPtr message) { receive_detections(message); });
    up_sub_ = create_subscription<Up>(up_topic, 10,
        [this](Up::ConstSharedPtr message) { receive_up(message); });
    enable_service_ = create_service<std_srvs::srv::SetBool>("~/set_enabled",
        [this](const std_srvs::srv::SetBool::Request::SharedPtr request,
               std_srvs::srv::SetBool::Response::SharedPtr response) {
          response->success = set_enabled(request->data);
          response->message = state_;
        });
    // 目标水平坐标允许运行时修改；更新后重置到位记忆，不更换锁定目标。
    parameter_callback_ = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> &updates) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          double distance = parameters_.target_distance_m;
          double lateral = parameters_.target_lateral_m;
          for (const auto &parameter : updates) {
            const bool is_distance = parameter.get_name() == "target_distance_m";
            if (!is_distance && parameter.get_name() != "target_lateral_m") continue;
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
                !std::isfinite(parameter.as_double()) || (is_distance && parameter.as_double() <= 0)) {
              result.successful = false;
              result.reason = "target coordinates must be finite doubles; target_distance_m must be positive";
              return result;
            }
            if (is_distance) distance = parameter.as_double();
            else lateral = parameter.as_double();
          }
          if (distance != parameters_.target_distance_m || lateral != parameters_.target_lateral_m) {
            parameters_.target_distance_m = distance;
            parameters_.target_lateral_m = lateral;
            controller_.set_target(distance, lateral);
          }
          return result;
        });
    // wall timer 与接收超时都不依赖 /clock；仿真时间暂停时仍能停车和打印。
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] { tick(); });
    if (startup_parameter("start_enabled", true)) set_enabled(true);
  }

  ~ControlNode() override { close_serial(); }

 private:
  // 其他参数只在启动时读取，避免运行中改变串口、频率或门限产生含糊的状态。
  template<typename T> T startup_parameter(const std::string &name, const T &value) {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    return declare_parameter<T>(name, value, descriptor);
  }

  ControlParameters read_control_parameters() {
    // YAML/命令行覆盖默认值；控制参数的集中校验交给 Controller 构造函数。
    ControlParameters p;
    p.target_distance_m = declare_parameter("target_distance_m", p.target_distance_m);
    p.target_lateral_m = declare_parameter("target_lateral_m", p.target_lateral_m);
    p.lateral_gain = startup_parameter("lateral_gain", p.lateral_gain);
    p.forward_gain = startup_parameter("forward_gain", p.forward_gain);
    p.lateral_tolerance_m = startup_parameter("lateral_tolerance_m", p.lateral_tolerance_m);
    p.distance_tolerance_m = startup_parameter("distance_tolerance_m", p.distance_tolerance_m);
    p.lateral_resume_m = startup_parameter("lateral_resume_m", p.lateral_resume_m);
    p.distance_resume_m = startup_parameter("distance_resume_m", p.distance_resume_m);
    p.max_forward_speed = startup_parameter("max_forward_speed", p.max_forward_speed);
    p.max_lateral_speed = startup_parameter("max_lateral_speed", p.max_lateral_speed);
    return p;
  }

  bool fresh(const std_msgs::msg::Header &header) {
    // 源时间戳检查推理结果是否已经过期；enabled_at_ 排除上轮排队中的旧帧。
    // 到达之后又多久未更新，则由 tick 中独立的单调时钟检查。
    const auto stamp = stamp_ns(header);
    const double age = (get_clock()->now().nanoseconds() - stamp) * 1e-9;
    return !header.frame_id.empty() && stamp > 0 && stamp >= enabled_at_ && age >= 0 && age <= timeout_;
  }

  const vision_msgs::msg::Detection3D *select_target(const Detections &message) const {
    // 尚未锁定时选最靠近光轴的有效方块；锁定后只查找同一 ID，不自动换目标。
    const vision_msgs::msg::Detection3D *selected = nullptr;
    double best = std::numeric_limits<double>::infinity();
    for (const auto &detection : message.detections) {
      // 即使上游同时发布两种规格，也只锁定所选规格；锁定后同样检查。
      const auto &size = detection.bbox.size;
      if (!std::isfinite(size.x) || !std::isfinite(size.y) || !std::isfinite(size.z) ||
          std::abs(size.x - target_size_m_) > 0.01 ||
          std::abs(size.y - target_size_m_) > 0.01 ||
          std::abs(size.z - target_size_m_) > 0.01) continue;
      const auto &p = detection.bbox.center.position;
      const auto &q = detection.bbox.center.orientation;
      const double quaternion_norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
      if (detection.id.empty() || !std::isfinite(p.x) || !std::isfinite(p.y) ||
          !std::isfinite(p.z) || p.z <= 0 || !std::isfinite(quaternion_norm) || quaternion_norm < 1e-12) continue;
      if (!target_id_.empty()) {
        if (detection.id == target_id_) return &detection;
        continue;
      }
      // X/Z、Y/Z 是归一化投影坐标，衡量相对光轴的偏离，避免只按物理横向距离选取。
      const double score = std::pow(p.x / p.z, 2) + std::pow(p.y / p.z, 2);
      if (score < best) { best = score; selected = &detection; }
    }
    return selected;
  }

  void invalidate(const std::string &reason, int64_t stamp) {
    // 清除可用观测后，下一个 tick（默认最多约 100 ms）输出零速。
    // 保留 target_id_，同一目标重新出现时可以恢复，其他方块不会接管控制。
    valid_pose_ = false;
    state_ = reason;
    controller_.reset();
    // 晚到的配对消息不能覆盖更新一帧的“目标已丢失”。
    invalid_before_ = std::max(invalid_before_, stamp);
  }

  void receive_detections(const Detections::ConstSharedPtr &message) {
    const auto stamp = stamp_ns(message->header);
    // 重复/倒序消息不刷新接收时间，防止旧帧持续到来时绕过超时停车。
    if (!enabled_ || stamp <= last_detection_stamp_) return;
    if (!fresh(message->header)) {
      invalidate("stale_pose", std::max(last_detection_stamp_, last_up_stamp_)); return;
    }
    last_detection_stamp_ = stamp;
    if (!frame_id_.empty() && frame_id_ != message->header.frame_id) {
      invalidate("frame_changed", stamp); return;
    }
    // 空检测/锁定目标消失直接停车，不等待这一帧的 UP 消息。
    if (!select_target(*message)) {
      invalidate(target_id_.empty() ? "no_target" : "target_lost", stamp); return;
    }
    auto &pending = pending_[stamp];
    pending.detections = message;
    // 记录检测真正到达的时刻；等待 UP 的延迟也计入接收超时。
    pending.received = SteadyClock::now();
    pair_frame(stamp);
  }

  void receive_up(const Up::ConstSharedPtr &message) {
    // 发布端把本次启动测得的 UP 附在每帧上；这里配对，不重新估计安装角度。
    const auto stamp = stamp_ns(message->header);
    if (!enabled_ || stamp <= last_up_stamp_) return;
    if (!fresh(message->header)) {
      invalidate("invalid_up", std::max(last_detection_stamp_, last_up_stamp_)); return;
    }
    last_up_stamp_ = stamp;
    const Eigen::Vector3d up(message->vector.x, message->vector.y, message->vector.z);
    if (!up.allFinite() || up.norm() < 1e-9 ||
        (!frame_id_.empty() && frame_id_ != message->header.frame_id)) {
      invalidate("invalid_up", stamp); return;
    }
    pending_[stamp].up = message;
    pair_frame(stamp);
  }

  void pair_frame(int64_t stamp) {
    // 两路 DDS 到达顺序不固定，只缓存最近 10 帧，按源时间戳精确配对。
    while (pending_.size() > 10) pending_.erase(pending_.begin());
    auto it = pending_.find(stamp);
    if (it == pending_.end() || !it->second.detections || !it->second.up) return;
    const auto pair = it->second;
    pending_.erase(it);
    // 配对晚于丢失通知、或早于已经使用过的帧时，不能让旧观测重新驱动车辆。
    if (stamp <= invalid_before_ || stamp <= last_pair_stamp_) return;
    const auto &header = pair.detections->header;
    if (!fresh(header) || header.frame_id != pair.up->header.frame_id ||
        (!frame_id_.empty() && header.frame_id != frame_id_)) {
      invalidate("invalid_pair", stamp); return;
    }
    const auto *target = select_target(*pair.detections);
    if (!target) { invalidate("target_lost", stamp); return; }
    // 首次成功配对才正式锁定 ID 和光学坐标系，此后只更新这个目标的中心位置。
    target_id_ = target->id;
    frame_id_ = header.frame_id;
    const auto &p = target->bbox.center.position;
    center_ = {p.x, p.y, p.z};
    const auto &u = pair.up->vector;
    up_ = {u.x, u.y, u.z};
    last_pair_stamp_ = stamp;
    received_ = pair.received;
    valid_pose_ = true;
  }

  bool close_serial() {
    // SerialPort::close() 先尝试发零速，再释放设备；未打开串口时直接返回。
    try { serial_.close(); return true; }
    catch (const std::exception &error) {
      RCLCPP_ERROR(get_logger(), "停车发送或关闭串口失败: %s", error.what());
      return false;
    }
  }

  bool set_enabled(bool enabled) {
    // 重复开启是幂等的，不打断正在跟踪的目标；故障状态允许通过服务显式重试。
    if (enabled_ == enabled && !fault_) return true;
    // 真正的启停切换先结束上一轮，清空观测与目标，让下一轮从新帧开始。
    enabled_ = false;
    valid_pose_ = false;
    pending_.clear();
    target_id_.clear();
    frame_id_.clear();
    controller_.reset();
    last_detection_stamp_ = last_up_stamp_ = last_pair_stamp_ = invalid_before_ = 0;
    const bool closed = close_serial();
    fault_ = !closed;
    state_ = closed ? "disabled" : "serial_fault";
    velocity_pub_->publish(geometry_msgs::msg::Twist{});
    if (!enabled || !closed) return closed;
    try {
      // 实车打开后先发零速，直到新观测满足条件才由 tick 发运动指令。
      if (!dry_run_) { serial_.open(serial_path_, baudrate_); serial_.send(0, 0); }
    } catch (const std::exception &error) {
      RCLCPP_ERROR(get_logger(), "启用串口失败: %s", error.what());
      close_serial();
      fault_ = true;
      state_ = "serial_fault";
      return false;
    }
    enabled_at_ = get_clock()->now().nanoseconds();
    enabled_ = true;
    fault_ = false;
    state_ = "waiting_pose";
    return true;
  }

  void tick() {
    // 每周期从零指令开始：禁用、无目标或超时都会自然保留零速。
    VelocityCommand command;
    if (enabled_ && valid_pose_) {
      // 单调时钟检查“多久没收到新观测”；ROS 时钟检查“这帧图像有多旧”。
      // 两者都检查，既覆盖推理/传输延迟，也覆盖 /clock 停止更新的情况。
      const double received_age = std::chrono::duration<double>(SteadyClock::now() - received_).count();
      const double source_age = (get_clock()->now().nanoseconds() - last_pair_stamp_) * 1e-9;
      if (received_age > timeout_ || source_age < 0 || source_age > timeout_) {
        invalidate("pose_timeout", last_pair_stamp_);
      } else {
        command = controller_.compute(center_, up_);
        state_ = state_name(command.state);
      }
    }
    const double vx = command.vx, vy = command.vy;
    const char *send_status = dry_run_ ? "not_opened" : "closed";
    if (!dry_run_ && enabled_) {
      try { serial_.send(vx, vy); send_status = "sent"; }
      catch (const std::exception &error) {
        RCLCPP_ERROR(get_logger(), "串口发送失败: %s", error.what());
        // 写失败结束本轮并释放串口，不在定时器中自动重连或恢复运动。
        enabled_ = valid_pose_ = false;
        fault_ = true;
        state_ = "serial_fault";
        close_serial();
        send_status = "failed";
      }
    }
    geometry_msgs::msg::Twist output;
    // Twist 其余分量保持默认零值，尤其不控制底盘旋转和竖直方向。
    output.linear.x = fault_ ? 0.0 : vx;
    output.linear.y = fault_ ? 0.0 : vy;
    velocity_pub_->publish(output);
    const double missing = std::numeric_limits<double>::quiet_NaN();
    // 丢失时位置显示 nan，避免把上一帧位置误看成当前观测；速度仍明确为零。
    Eigen::Vector2d horizontal(missing, missing);
    if (valid_pose_) project_to_horizontal(center_, up_, horizontal);
    const double x = horizontal.x(), y = horizontal.y();
    // cmd_* 是本周期计算指令；wire_* 是截断到 mm/s 后的编码值，并非实测车速。
    // serial=sent/failed 表示该次写入结果；dry_run 始终显示 not_opened。
    RCLCPP_INFO(get_logger(),
        "mode=%s size=%s state=%s id=%s plane_x=%.3f plane_y=%.3f ex=%.3f ey=%.3f "
        "cmd_vx=%+.4f cmd_vy=%+.4f vw=0.000 wire_vx=%+.3f wire_vy=%+.3f serial=%s",
        dry_run_ ? "dry_run" : "serial", size_.c_str(), state_.c_str(), target_id_.empty() ? "-" : target_id_.c_str(),
        x, y, x - parameters_.target_distance_m, y - parameters_.target_lateral_m, vx, vy,
        std::trunc(vx * 1000) / 1000, std::trunc(vy * 1000) / 1000, send_status);
  }

  struct PendingFrame {
    // 一条消息先到时保留半帧，另一条消息到达相同时间戳后再组合使用。
    Detections::ConstSharedPtr detections;
    Up::ConstSharedPtr up;
    SteadyClock::time_point received;
  };
  ControlParameters parameters_;
  Controller controller_;
  SerialPort serial_;
  std::map<int64_t, PendingFrame> pending_;
  Eigen::Vector3d center_ = Eigen::Vector3d::Zero(), up_ = Eigen::Vector3d::Zero();
  SteadyClock::time_point received_;
  // 以下都是 ROS 源时间戳（纳秒）；received_ 单独使用 steady_clock 计量超时。
  int64_t enabled_at_ = 0, last_detection_stamp_ = 0, last_up_stamp_ = 0;
  int64_t last_pair_stamp_ = 0, invalid_before_ = 0;
  // enabled_ 表示本轮启用；valid_pose_ 表示当前可计算；fault_ 表示串口故障锁止。
  bool dry_run_ = true, enabled_ = false, valid_pose_ = false, fault_ = false;
  double timeout_ = 0.5;
  double target_size_m_ = 0.35;
  std::string size_;
  int baudrate_ = 961200;
  std::string target_id_, frame_id_, serial_path_, state_ = "disabled";
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
  rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
  rclcpp::Subscription<Up>::SharedPtr up_sub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
};
}  // namespace box_pose_control

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<box_pose_control::ControlNode>();
    rclcpp::spin(node);
    node.reset();  // SIGINT 后仍用串口发送最后的停车帧。
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("box_pose_control"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
