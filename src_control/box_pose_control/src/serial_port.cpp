#include "box_pose_control/serial_port.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <asm/termbits.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace box_pose_control {
namespace {

std::runtime_error io_error(const std::string & operation)
{
  return std::runtime_error(operation + ": " + std::strerror(errno));
}

unsigned int baud_flag(int baudrate)
{
  // 标准速率与 pySerial 一样使用 B* 常量，仅自定义速率使用 BOTHER。
  switch (baudrate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    default: return BOTHER;
  }
}

void write_speed(VelocityFrame & frame, std::size_t offset, double speed)
{
  // 线速度用有符号 int16 表示 mm/s，小端序；转换时向零截断。
  const auto value = static_cast<std::int16_t>(std::clamp(speed, -0.50, 0.50) * 1000.0);
  const auto bits = static_cast<std::uint16_t>(value);
  frame[offset] = static_cast<std::uint8_t>(bits & 0xff);
  frame[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
}

}  // namespace

VelocityFrame encode_velocity(double vx, double vy, std::int8_t mode, double wire_wz)
{
  if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(wire_wz)) {
    throw std::invalid_argument("velocity values must be finite");
  }
  VelocityFrame frame{};
  frame[0] = 0xa5;
  write_speed(frame, 1, vx);
  write_speed(frame, 3, vy);
  // 与旧键盘接口相同：角速度乘 10、有符号小端；方向转换由节点完成。
  const auto angular = static_cast<std::uint16_t>(static_cast<std::int16_t>(
    std::clamp(wire_wz, -3276.7, 3276.7) * 10.0));
  frame[5] = static_cast<std::uint8_t>(angular & 0xff);
  frame[6] = static_cast<std::uint8_t>(angular >> 8);
  frame[7] = static_cast<std::uint8_t>(mode);
  // 以下字节位置从 0 开始：CRC8 初值为 0、多项式为 0x07，
  // 仅覆盖下标 1～7，不包含帧头 A5、CRC 字节和帧尾 FF。
  std::uint8_t crc = 0;
  for (std::size_t i = 1; i <= 7; ++i) {
    crc ^= frame[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = static_cast<std::uint8_t>((crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1);
    }
  }
  frame[8] = crc;
  frame[9] = 0xff;
  return frame;
}

SerialPort::~SerialPort()
{
  try {
    close();
  } catch (const std::runtime_error &) {
    // 析构必须释放设备，错误由显式 close() 或 send() 的调用者记录。
  }
}

void SerialPort::open(const std::string & path, int baudrate)
{
  if (baudrate <= 0) {
    throw std::invalid_argument("baudrate must be positive");
  }
  close();
  // 非阻塞方式打开，发送时由 poll 控制等待时间，避免卡住 ROS 定时回调。
  const int fd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    throw io_error("open " + path);
  }

  try {
    // 与旧 Python 串口的 exclusive=True 使用相同的 flock 锁。
    if (::flock(fd, LOCK_EX | LOCK_NB) < 0) {
      throw io_error("lock " + path);
    }
    termios2 settings{};
    if (::ioctl(fd, TCGETS2, &settings) < 0) {
      throw io_error("read serial settings");
    }
    // raw 模式关闭终端字符处理；8N1 为 8 位数据、无校验、1 位停止位，
    // 同时禁用软硬件流控，确保二进制速度帧逐字节原样传输。
    settings.c_iflag = 0;
    settings.c_oflag = 0;
    settings.c_lflag = 0;
    settings.c_cflag &= ~(CBAUD | CIBAUD | CSIZE | CSTOPB | PARENB | CMSPAR | CRTSCTS);
    settings.c_cflag |= baud_flag(baudrate) | CS8 | CLOCAL | CREAD;
    // 默认使用标准 B115200；termios2 + BOTHER 保留自定义波特率支持。
    settings.c_ispeed = static_cast<unsigned int>(baudrate);
    settings.c_ospeed = static_cast<unsigned int>(baudrate);
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0;
    if (::ioctl(fd, TCSETS2, &settings) < 0) {
      throw io_error("set serial settings");
    }
    // 对齐键盘端 pySerial 的默认 DTR/RTS 电平；禁用流控不等于设置线路电平。
    int modem_lines = TIOCM_DTR | TIOCM_RTS;
    if (::ioctl(fd, TIOCMBIS, &modem_lines) < 0 && errno != EINVAL && errno != ENOTTY) {
      throw io_error("set serial DTR/RTS");
    }
    if (::ioctl(fd, TCFLSH, TCIFLUSH) < 0) {
      throw io_error("flush serial input");
    }
  } catch (...) {
    // 配置尚未成功，fd 仍由本函数负责清理，不交给对象保存。
    ::close(fd);
    throw;
  }
  fd_ = fd;
}

void SerialPort::send(double vx, double vy, std::int8_t mode, double wire_wz)
{
  if (fd_ < 0) {
    throw std::runtime_error("serial port is not open");
  }
  const auto frame = encode_velocity(vx, vy, mode, wire_wz);
  // 单调时钟不受系统校时影响；整帧所有 write/poll 共用 20 ms 预算。
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
  std::size_t offset = 0;
  while (offset < frame.size()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("serial write timed out after 20 ms");
    }
    const auto written = ::write(fd_, frame.data() + offset, frame.size() - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
      throw io_error("write serial frame");
    }

    // 部分写入仍沿用本帧的同一截止时间，不为每次重试重新计时。
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) {
      throw std::runtime_error("serial write timed out after 20 ms");
    }
    const int wait_ms = static_cast<int>(
      std::chrono::ceil<std::chrono::milliseconds>(remaining).count());
    pollfd descriptor{fd_, POLLOUT, 0};
    const int ready = ::poll(&descriptor, 1, wait_ms);
    if (ready < 0 && errno == EINTR) {
      continue;
    }
    if (ready < 0) {
      throw io_error("poll serial output");
    }
    if (ready == 0) {
      throw std::runtime_error("serial write timed out after 20 ms");
    }
    if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      throw std::runtime_error("serial device disconnected while writing");
    }
  }
}

void SerialPort::close()
{
  if (fd_ < 0) {
    return;
  }
  const int fd = fd_;
  try {
    // 主动关闭或析构时都先尝试停车，控制板断链时允许这次发送报错。
    send(0.0, 0.0);
  } catch (...) {
    // 停车发送失败也必须释放描述符及独占锁，再把原异常交回上层。
    fd_ = -1;
    ::close(fd);
    throw;
  }
  fd_ = -1;
  if (::close(fd) < 0) {
    throw io_error("close serial port");
  }
}

bool SerialPort::is_open() const
{
  return fd_ >= 0;
}

}  // namespace box_pose_control
