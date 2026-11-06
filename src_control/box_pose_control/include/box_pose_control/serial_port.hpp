#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace box_pose_control {

using VelocityFrame = std::array<std::uint8_t, 12>;

// 将底盘前进、左移速度（m/s）编码为旧控制板协议；角速度固定为零。
VelocityFrame encode_velocity(double vx, double vy);

// 仅负责串口生命周期和速度帧发送，不参与目标选择或位置控制。
// 实例独占设备；dry_run 模式由上层跳过打开和发送，不占用串口。
class SerialPort {
public:
  SerialPort() = default;
  ~SerialPort();

  SerialPort(const SerialPort &) = delete;
  SerialPort & operator=(const SerialPort &) = delete;

  // 重开时先关闭原设备；初始化失败不会留下未释放的文件描述符。
  void open(const std::string & path, int baudrate);
  // 发送一帧速度，最多等待 20 ms；成功只表示写入完成，不表示收到控制板回包。
  void send(double vx, double vy);
  // 关闭前尽力发送零速；即使发送失败，也释放设备并向调用者报告异常。
  void close();
  bool is_open() const;

private:
  // -1 表示未打开；只有设备配置全部成功后，才接管新文件描述符。
  int fd_ = -1;
};

}  // namespace box_pose_control
