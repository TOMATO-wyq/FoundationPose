#include "box_pose_control/serial_port.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <string>

#include <poll.h>
#include <pty.h>
#include <unistd.h>

namespace box_pose_control {
namespace {

TEST(VelocityFrameTest, MatchesLegacyWireFormatAndCrc)
{
  const VelocityFrame expected{
    0xa5, 0x7b, 0x00, 0xd3, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0xff};
  EXPECT_EQ(encode_velocity(0.1239, -0.0459), expected);
}

TEST(VelocityFrameTest, ClampsBothSpeedsAtHardwareLimit)
{
  const auto frame = encode_velocity(2.0, -3.0);
  EXPECT_EQ(frame[1], 0x2c);
  EXPECT_EQ(frame[2], 0x01);
  EXPECT_EQ(frame[3], 0xd4);
  EXPECT_EQ(frame[4], 0xfe);
  EXPECT_EQ(frame, encode_velocity(0.30, -0.30));
}

TEST(VelocityFrameTest, EncodesZeroSpeedWithoutAngularVelocityOrFlags)
{
  const VelocityFrame expected{
    0xa5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff};
  EXPECT_EQ(encode_velocity(0.0, 0.0), expected);
  EXPECT_EQ(encode_velocity(0.0009, -0.0009), expected);
}

class SerialPortTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    int slave_fd = -1;
    char name[128]{};
    ASSERT_EQ(::openpty(&master_fd_, &slave_fd, name, nullptr, nullptr), 0);
    path_ = name;
    ::close(slave_fd);
  }

  void TearDown() override
  {
    if (master_fd_ >= 0) {
      ::close(master_fd_);
    }
  }

  VelocityFrame read_frame()
  {
    VelocityFrame frame{};
    std::size_t offset = 0;
    while (offset < frame.size()) {
      pollfd descriptor{master_fd_, POLLIN, 0};
      const int ready = ::poll(&descriptor, 1, 200);
      EXPECT_EQ(ready, 1);
      if (ready != 1) {
        break;
      }
      const auto count = ::read(master_fd_, frame.data() + offset, frame.size() - offset);
      EXPECT_GT(count, 0);
      if (count <= 0) {
        break;
      }
      offset += static_cast<std::size_t>(count);
    }
    EXPECT_EQ(offset, frame.size());
    return frame;
  }

  int master_fd_ = -1;
  std::string path_;
};

TEST_F(SerialPortTest, SendsFrameAtNonstandardBaudAndStopsOnClose)
{
  SerialPort serial;
  EXPECT_FALSE(serial.is_open());
  serial.open(path_, 961200);
  EXPECT_TRUE(serial.is_open());
  serial.send(0.1239, -0.0459);
  EXPECT_EQ(read_frame(), encode_velocity(0.1239, -0.0459));
  serial.close();
  EXPECT_FALSE(serial.is_open());
  EXPECT_EQ(read_frame(), encode_velocity(0.0, 0.0));
  EXPECT_NO_THROW(serial.close());
}

TEST_F(SerialPortTest, ExclusiveLockIsReleasedForReopening)
{
  SerialPort first;
  SerialPort second;
  first.open(path_, 961200);
  EXPECT_THROW(second.open(path_, 961200), std::runtime_error);
  EXPECT_FALSE(second.is_open());
  first.close();
  EXPECT_EQ(read_frame(), encode_velocity(0.0, 0.0));

  EXPECT_NO_THROW(second.open(path_, 961200));
  second.send(0.1, 0.0);
  EXPECT_EQ(read_frame(), encode_velocity(0.1, 0.0));
  second.close();
  EXPECT_EQ(read_frame(), encode_velocity(0.0, 0.0));

  EXPECT_NO_THROW(first.open(path_, 961200));
  first.close();
  EXPECT_EQ(read_frame(), encode_velocity(0.0, 0.0));
}

TEST_F(SerialPortTest, DestructorSendsStopFrame)
{
  {
    SerialPort serial;
    serial.open(path_, 961200);
    serial.send(0.1, 0.0);
    EXPECT_EQ(read_frame(), encode_velocity(0.1, 0.0));
  }
  EXPECT_EQ(read_frame(), encode_velocity(0.0, 0.0));
}

TEST_F(SerialPortTest, DisconnectReportsErrorAndCloseReleasesDescriptor)
{
  SerialPort serial;
  serial.open(path_, 961200);
  ::close(master_fd_);
  master_fd_ = -1;
  EXPECT_THROW(serial.send(0.1, 0.0), std::runtime_error);
  EXPECT_THROW(serial.close(), std::runtime_error);
  EXPECT_FALSE(serial.is_open());
  EXPECT_NO_THROW(serial.close());
}

TEST_F(SerialPortTest, DestructorDoesNotThrowAfterDisconnect)
{
  SerialPort serial;
  serial.open(path_, 961200);
  ::close(master_fd_);
  master_fd_ = -1;
}

TEST_F(SerialPortTest, BackpressureTimesOutWithinOneFrameBudget)
{
  SerialPort serial;
  serial.open(path_, 961200);
  bool timed_out = false;
  // 不读取主端，填满真实 PTY 输出队列，触发部分写入或 EAGAIN。
  for (int frame = 0; frame < 10000; ++frame) {
    const auto start = std::chrono::steady_clock::now();
    try {
      serial.send(0.1, 0.02);
    } catch (const std::runtime_error & error) {
      const auto elapsed = std::chrono::steady_clock::now() - start;
      EXPECT_NE(std::string(error.what()).find("timed out"), std::string::npos);
      EXPECT_LT(elapsed, std::chrono::milliseconds(200));
      timed_out = true;
      break;
    }
  }
  EXPECT_TRUE(timed_out);
  EXPECT_THROW(serial.close(), std::runtime_error);
  EXPECT_FALSE(serial.is_open());
}

TEST(SerialPortStateTest, OpeningMissingDeviceAndSendingWhileClosedReportErrors)
{
  SerialPort serial;
  EXPECT_THROW(serial.open("/no-such-box-pose-control-device", 961200), std::runtime_error);
  EXPECT_FALSE(serial.is_open());
  EXPECT_THROW(serial.send(0.1, 0.0), std::runtime_error);
}

}  // namespace
}  // namespace box_pose_control
