#include "box_pose_control/controller.hpp"
#include "box_pose_control/serial_port.hpp"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace box_pose_control {
namespace {

const Eigen::Vector3d kLevelUp(0.0, -1.0, 0.0);

Eigen::Matrix<double, 3, 2> horizontal_axes(const Eigen::Vector3d &up) {
  const Eigen::Vector3d u = up.normalized();
  const Eigen::Vector3d forward = (Eigen::Vector3d::UnitZ() - u * u.z()).normalized();
  Eigen::Matrix<double, 3, 2> axes;
  axes.col(0) = forward;
  axes.col(1) = u.cross(forward);
  return axes;
}

TEST(Controller, ValidatesControlParameters) {
  EXPECT_NO_THROW(validate_parameters(ControlParameters{}));
  ControlParameters parameters;
  parameters.target_distance_m = 0.0;
  EXPECT_THROW(Controller{parameters}, std::invalid_argument);
  parameters = ControlParameters{};
  parameters.forward_gain = -1.0;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.lateral_tolerance_m = -0.01;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.lateral_resume_m = 0.005;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.distance_resume_m = 0.01;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.max_forward_speed = 0.501;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.max_lateral_speed = 0.0;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.lateral_gain = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.distance_tolerance_m = std::numeric_limits<double>::infinity();
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters = ControlParameters{};
  parameters.camera_yaw_deg = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
  parameters.camera_yaw_deg = 181.0;
  EXPECT_THROW(validate_parameters(parameters), std::invalid_argument);
}

TEST(Controller, UsesCorrectLeftRightAndForwardBackwardSigns) {
  Controller controller(ControlParameters{});
  auto command = controller.compute({0.05, 0.8, 0.30}, kLevelUp);
  EXPECT_NEAR(command.vx, 0.0, 1e-12);
  EXPECT_NEAR(command.vy, -0.10, 1e-12);
  EXPECT_EQ(command.state, ControlState::Aligning);
  command = controller.compute({-0.05, -0.8, 0.30}, kLevelUp);
  EXPECT_NEAR(command.vy, 0.10, 1e-12);
  command = controller.compute({0.0, 10.0, 0.36}, kLevelUp);
  EXPECT_NEAR(command.vx, 0.24, 1e-12);
  EXPECT_NEAR(command.vy, 0.0, 1e-12);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  command = controller.compute({0.0, -10.0, 0.24}, kLevelUp);
  EXPECT_NEAR(command.vx, -0.24, 1e-12);
}

TEST(Controller, ControlsBothAxesTogetherAndStopsEachAxisIndependently) {
  Controller controller(ControlParameters{});
  const auto align = controller.compute({0.05, 0.0, 0.60}, kLevelUp);
  EXPECT_EQ(align.state, ControlState::AdjustingPosition);
  EXPECT_GT(align.vx, 0.0);
  EXPECT_LT(align.vy, 0.0);
  EXPECT_NEAR(align.vx, 0.50, 1e-12);
  EXPECT_NEAR(align.vy, -0.10, 1e-12);
  const auto distance = controller.compute({0.009, 0.0, 0.60}, kLevelUp);
  EXPECT_EQ(distance.state, ControlState::AdjustingDistance);
  EXPECT_GT(distance.vx, 0.0);
  EXPECT_NEAR(distance.vy, 0.0, 1e-12);
  const auto lateral = controller.compute({0.05, 0.0, 0.31}, kLevelUp);
  EXPECT_EQ(lateral.state, ControlState::Aligning);
  EXPECT_DOUBLE_EQ(lateral.vx, 0.0);
  EXPECT_LT(lateral.vy, 0.0);
}

TEST(Controller, LateralHysteresisAvoidsBoundaryChatterAndResetClearsIt) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.025, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingPosition);
  EXPECT_EQ(controller.compute({0.019, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingDistance);
  EXPECT_EQ(controller.compute({0.029, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingDistance);
  EXPECT_EQ(controller.compute({0.031, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingPosition);
  EXPECT_EQ(controller.compute({-0.025, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingPosition);
  controller.compute({0.0, 0.0, 0.50}, kLevelUp);
  controller.reset();
  EXPECT_EQ(controller.compute({0.025, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingPosition);
}

TEST(Controller, SaturatingOneAxisDoesNotSuppressTheOtherAxis) {
  for (double forward_sign : {-1.0, 1.0}) {
    for (double left_sign : {-1.0, 1.0}) {
      ControlParameters parameters;
      parameters.target_distance_m = 1.0;
      Controller controller(parameters);
      // 复现实车日志中前后误差约 0.836 m、横向误差约 0.045 m 的情况。
      auto command = controller.compute(
        {-left_sign * 0.045, 0.0, 1.0 + forward_sign * 0.836}, kLevelUp);
      EXPECT_EQ(command.state, ControlState::AdjustingPosition);
      EXPECT_NEAR(command.vx, forward_sign * 0.50, 1e-12);
      EXPECT_NEAR(command.vy, left_sign * 0.09, 1e-12);
      controller.reset();
      command = controller.compute(
        {-left_sign * 0.836, 0.0, 1.0 + forward_sign * 0.045}, kLevelUp);
      EXPECT_NEAR(command.vx, forward_sign * 0.18, 1e-12);
      EXPECT_NEAR(command.vy, left_sign * 0.50, 1e-12);
    }
  }
}

TEST(Controller, DistanceHysteresisHoldsUntilResumeThreshold) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.34}, kLevelUp).state,
            ControlState::AdjustingDistance);
  const auto in_position = controller.compute({0.0, 0.0, 0.31}, kLevelUp);
  EXPECT_EQ(in_position.state, ControlState::InPosition);
  EXPECT_DOUBLE_EQ(in_position.vx, 0.0);
  EXPECT_DOUBLE_EQ(in_position.vy, 0.0);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.34}, kLevelUp).state,
            ControlState::InPosition);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.346}, kLevelUp).state,
            ControlState::AdjustingDistance);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.26}, kLevelUp).state,
            ControlState::AdjustingDistance);
  controller.compute({0.0, 0.0, 0.30}, kLevelUp);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.26}, kLevelUp).state,
            ControlState::InPosition);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.254}, kLevelUp).state,
            ControlState::AdjustingDistance);
}

TEST(Controller, ChangingTargetResetsHysteresisAndUsesNewDistance) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.30}, kLevelUp).state,
            ControlState::InPosition);
  controller.set_target_distance(0.26);
  const auto command = controller.compute({0.0, 0.0, 0.30}, kLevelUp);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  EXPECT_NEAR(command.vx, 0.16, 1e-12);
  controller.set_target_distance(0.30);
  EXPECT_EQ(controller.compute({0.025, 0.0, 0.30}, kLevelUp).state,
            ControlState::Aligning);
}

TEST(Controller, HoldsBothAxesThroughObservationJitterUntilResumeThresholds) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.015, 0.0, 0.325}, kLevelUp).state, ControlState::InPosition);
  // 两轴分别越过停车容差但仍在恢复门限内，模拟视角/深度噪声。
  for (int frame = 0; frame < 20; ++frame) {
    const auto command = controller.compute(
      {frame % 2 ? 0.029 : -0.029, 0.0, frame % 2 ? 0.344 : 0.256}, kLevelUp);
    EXPECT_EQ(command.state, ControlState::InPosition);
    EXPECT_DOUBLE_EQ(command.vx, 0.0);
    EXPECT_DOUBLE_EQ(command.vy, 0.0);
  }
  const auto resumed = controller.compute({0.031, 0.0, 0.346}, kLevelUp);
  EXPECT_EQ(resumed.state, ControlState::AdjustingPosition);
  EXPECT_GT(resumed.vx, 0.0);
  EXPECT_LT(resumed.vy, 0.0);
}

TEST(Controller, ProjectsTiltedPositionsAndIgnoresHeight) {
  const Eigen::Vector3d orientations[] = {
      kLevelUp, -kLevelUp, Eigen::Vector3d::UnitX(),
      Eigen::Vector3d(0.0, -std::sqrt(0.75), -0.5),
      Eigen::Vector3d(0.3, -std::sqrt(0.66), -0.5)};
  for (const auto &up : orientations) {
    const auto axes = horizontal_axes(up);
    for (double height : {-0.4, 0.0, 0.6}) {
      Controller controller(ControlParameters{});
      const Eigen::Vector3d center = axes * Eigen::Vector2d(0.35, 0.05) + height * up;
      auto command = controller.compute(center, up * 4.0);
      EXPECT_EQ(command.state, ControlState::AdjustingPosition);
      EXPECT_NEAR(command.vx, 0.20, 1e-12);
      EXPECT_NEAR(command.vy, 0.10, 1e-12);
      command = controller.compute(axes * Eigen::Vector2d(0.35, 0.0) + height * up, up);
      EXPECT_EQ(command.state, ControlState::AdjustingDistance);
      EXPECT_NEAR(command.vx, 0.20, 1e-12);
      EXPECT_NEAR(command.vy, 0.0, 1e-12);
      command = controller.compute(axes * Eigen::Vector2d(0.30, 0.0) + height * up, up);
      EXPECT_EQ(command.state, ControlState::InPosition);
    }
  }
}

TEST(Controller, CompensatesMountYawWithPitchRollAndHeight) {
  const double mount_yaw = std::atan2(15.0, 20.0) * 180.0 / std::acos(-1.0);
  const Eigen::Vector3d ups[] = {
    kLevelUp, -kLevelUp, Eigen::Vector3d(0.3, -std::sqrt(0.66), -0.5)};
  for (const auto &up : ups) {
    const auto axes = horizontal_axes(up);
    for (double sign : {-1.0, 1.0}) {
      ControlParameters parameters;
      parameters.target_distance_m = 0.5;
      parameters.camera_yaw_deg = sign * mount_yaw;
      Controller controller(parameters);
      for (double height : {-0.4, 0.0, 0.6}) {
        Eigen::Vector2d position;
        const Eigen::Vector3d on_optical_axis = axes * Eigen::Vector2d(0.5, 0.0) + height * up;
        ASSERT_TRUE(project_to_horizontal(on_optical_axis, up, position, parameters.camera_yaw_deg));
        EXPECT_NEAR(position.x(), 0.4, 1e-12);
        EXPECT_NEAR(position.y(), sign * 0.3, 1e-12);
        // 底盘正前方 0.5 m 对应相机前方 0.4 m、横向 -/+0.3 m；不添加安装平移。
        const Eigen::Vector3d desired = axes * Eigen::Vector2d(0.4, -sign * 0.3) + height * up;
        EXPECT_EQ(controller.compute(desired, up).state, ControlState::InPosition);
        const Eigen::Vector3d displaced = axes * Eigen::Vector2d(0.8 * 0.65 + sign * 0.6 * 0.08,
                                      -sign * 0.6 * 0.65 + 0.8 * 0.08) + height * up;
        const auto command = controller.compute(displaced, up);
        EXPECT_EQ(command.state, ControlState::AdjustingPosition);
        EXPECT_NEAR(command.vx, 0.50, 1e-12);
        EXPECT_NEAR(command.vy, 0.16, 1e-12);
      }
    }
  }
  Eigen::Vector2d position;
  EXPECT_FALSE(project_to_horizontal({0.0, 0.0, 0.5}, kLevelUp, position,
               std::numeric_limits<double>::infinity()));
}

TEST(Controller, UsesConfigurableHorizontalTargetAndResetsOnChange) {
  ControlParameters parameters;
  parameters.target_distance_m = 0.5;
  parameters.target_lateral_m = -0.08;
  Controller controller(parameters);
  EXPECT_EQ(controller.compute({0.08, 2.0, 0.5}, kLevelUp).state, ControlState::InPosition);
  controller.set_target(0.5, 0.02);
  auto command = controller.compute({0.08, 2.0, 0.5}, kLevelUp);
  EXPECT_EQ(command.state, ControlState::Aligning);
  EXPECT_NEAR(command.vy, -0.20, 1e-12);
  EXPECT_THROW(controller.set_target(0.5, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW(controller.set_target(-0.1, 0.0), std::invalid_argument);
}

TEST(Controller, LimitsHorizontalSpeedsWithTilt) {
  ControlParameters parameters;
  parameters.max_forward_speed = 0.10;
  parameters.max_lateral_speed = 0.02;
  Controller controller(parameters);
  const Eigen::Vector3d up(0.5, -std::sqrt(0.39), -0.6);
  const auto axes = horizontal_axes(up);
  auto command = controller.compute(axes * Eigen::Vector2d(1.0, -0.5), up);
  EXPECT_EQ(command.state, ControlState::AdjustingPosition);
  EXPECT_NEAR(command.vx, 0.10, 1e-12);
  EXPECT_NEAR(command.vy, -0.02, 1e-12);
  command = controller.compute(axes * Eigen::Vector2d(1.0, 0.0), up);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  EXPECT_NEAR(command.vx, 0.10, 1e-12);
  EXPECT_NEAR(command.vy, 0.0, 1e-12);
}

TEST(Controller, RejectsDegenerateGeometryAndNonfiniteInputs) {
  Controller controller(ControlParameters{});
  const Eigen::Vector3d invalid_ups[] = {
      Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d(0.0, -0.2, -std::sqrt(0.96)),
      Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), -1.0, 0.0)};
  for (const auto &up : invalid_ups) {
    const auto command = controller.compute({0.1, 0.0, 0.50}, up);
    EXPECT_EQ(command.state, ControlState::InvalidGeometry);
    EXPECT_DOUBLE_EQ(command.vx, 0.0);
    EXPECT_DOUBLE_EQ(command.vy, 0.0);
  }
  const auto command = controller.compute(
      {std::numeric_limits<double>::infinity(), 0.0, 0.50}, kLevelUp);
  EXPECT_EQ(command.state, ControlState::InvalidGeometry);
}

TEST(Controller, ClosedLoopConvergesWithTiltAndSpeedLimitsAtTenHertz) {
  ControlParameters parameters;
  parameters.max_forward_speed = 0.08;
  parameters.max_lateral_speed = 0.06;
  Controller controller(parameters);
  const Eigen::Vector3d up(0.3, -std::sqrt(0.66), -0.5);
  const auto axes = horizontal_axes(up);
  Eigen::Vector3d center(0.24, 0.12, 0.85);
  VelocityCommand command;
  bool moved_both_axes = false;
  for (int tick = 0; tick < 400; ++tick) {
    command = controller.compute(center, up);
    ASSERT_NE(command.state, ControlState::InvalidGeometry);
    EXPECT_LE(std::abs(command.vx), parameters.max_forward_speed + 1e-12);
    EXPECT_LE(std::abs(command.vy), parameters.max_lateral_speed + 1e-12);
    const Eigen::Vector3d previous = center;
    center -= 0.1 * axes * Eigen::Vector2d(command.vx, command.vy);
    if (command.state == ControlState::AdjustingPosition) {
      EXPECT_NE(command.vx, 0.0);
      EXPECT_NE(command.vy, 0.0);
      moved_both_axes = true;
    }
    if (command.state == ControlState::Aligning) {
      EXPECT_NEAR(center.dot(axes.col(0)), previous.dot(axes.col(0)), 1e-12);
    }
    if (command.state == ControlState::AdjustingDistance) {
      EXPECT_NEAR(center.dot(axes.col(1)), previous.dot(axes.col(1)), 1e-12);
    }
  }
  EXPECT_TRUE(moved_both_axes);
  EXPECT_EQ(command.state, ControlState::InPosition);
  EXPECT_LE(std::abs(center.dot(axes.col(1)) - parameters.target_lateral_m), parameters.lateral_tolerance_m);
  EXPECT_LE(std::abs(center.dot(axes.col(0)) - parameters.target_distance_m), parameters.distance_tolerance_m);
  EXPECT_DOUBLE_EQ(command.vx, 0.0);
  EXPECT_DOUBLE_EQ(command.vy, 0.0);
}

}  // namespace
}  // namespace box_pose_control

namespace box_pose_control {
namespace {
Eigen::Matrix3d cube_rotation(double degrees) {
  const double angle = degrees * std::acos(-1.0) / 180;
  const auto axes = horizontal_axes(kLevelUp);
  Eigen::Matrix3d result;
  result.col(0) = std::cos(angle) * axes.col(0) + std::sin(angle) * axes.col(1);
  result.col(1) = -std::sin(angle) * axes.col(0) + std::cos(angle) * axes.col(1);
  result.col(2) = kLevelUp;
  return result;
}

VelocityCommand confirmed(Controller &controller, double degrees, double start = 1.0,
                          Eigen::Vector3d center = {0, 0, 0.3}) {
  VelocityCommand command;
  for (int i = 0; i < 3; ++i)
    command = controller.compute(center, kLevelUp, cube_rotation(degrees), start + i * 0.1);
  return command;
}

TEST(YawController, CubeSymmetryMountAndTilt) {
  const double radians = std::acos(-1.0) / 180;
  for (double angle : {-179., -91., -89., -20., 0., 20., 89., 91., 179.}) {
    double measured;
    ASSERT_TRUE(box_yaw_error(cube_rotation(angle), kLevelUp, 0, measured));
    EXPECT_NEAR(std::remainder((measured / radians) - angle, 90), 0, 1e-10);
  }
  double measured;
  ASSERT_TRUE(box_yaw_error(cube_rotation(10), kLevelUp, -10, measured));
  EXPECT_NEAR(measured, 0, 1e-12);
  const Eigen::Matrix3d tilt = Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitX()).toRotationMatrix();
  ASSERT_TRUE(box_yaw_error(tilt * cube_rotation(20), tilt * kLevelUp, 0, measured));
  EXPECT_NEAR(measured / radians, 20, 1e-10);
  // 等价的有向轴置换不改变最近面角差。
  Eigen::Matrix3d permutation;
  permutation << 0, 0, 1, 1, 0, 0, 0, 1, 0;
  ASSERT_TRUE(box_yaw_error(cube_rotation(20) * permutation, kLevelUp, 0, measured));
  EXPECT_NEAR(measured / radians, 20, 1e-10);
}

TEST(YawController, UsesHysteresisAndRequiresDistinctFrames) {
  Controller controller(ControlParameters{});
  for (int i = 0; i < 10; ++i) {
    const auto pending = controller.compute({0, 0, .3}, kLevelUp, cube_rotation(12), 1.0);
    EXPECT_FALSE(pending.yaw_ready);
    EXPECT_EQ(pending.wz, 0);
  }
  auto command = confirmed(controller, 12, 1.1);
  EXPECT_GT(command.wz, 0);
  EXPECT_EQ(command.state, ControlState::AligningYaw);
  command = confirmed(controller, 4, 1.4);
  EXPECT_EQ(command.wz, 0);
  EXPECT_EQ(command.state, ControlState::InPosition);
  command = confirmed(controller, 7, 1.7);
  EXPECT_EQ(command.wz, 0);
  command = confirmed(controller, 9, 2.0);
  EXPECT_GE(command.wz, 0.1);
  controller.reset();
  command = confirmed(controller, -12, 3.0);
  EXPECT_LT(command.wz, 0);
  EXPECT_FALSE(controller.compute({0, 0, .3}, kLevelUp, cube_rotation(-12), 3.1).yaw_ready);
}

TEST(YawController, GatesJumpsAndKeepsFarApproachSlow) {
  Controller controller(ControlParameters{});
  auto pending = controller.compute({.1, 0, 1.5}, kLevelUp, cube_rotation(20), 1);
  EXPECT_FALSE(pending.yaw_ready);
  EXPECT_LE(pending.vx, .15);
  EXPECT_EQ(pending.wz, 0);
  auto far = confirmed(controller, 20, 1.1, {.1, 0, 1.5});
  EXPECT_NEAR(far.wz, .2, 1e-12);
  EXPECT_GT(far.vx, 0);
  EXPECT_LT(far.vy, 0);
  auto near = controller.compute({.1, 0, .6}, kLevelUp, cube_rotation(20), 1.4);
  EXPECT_GT(near.wz, far.wz);
  auto jump = controller.compute({.1, 0, .6}, kLevelUp, cube_rotation(-20), 1.5);
  EXPECT_FALSE(jump.yaw_ready);
  EXPECT_EQ(jump.vx, 0);
  EXPECT_EQ(jump.vy, 0);
  EXPECT_EQ(jump.wz, 0);
  const auto invalid = controller.compute({0, 0, .3}, kLevelUp, Eigen::Matrix3d::Zero(), 1.6);
  EXPECT_FALSE(invalid.yaw_ready);
  EXPECT_EQ(invalid.wz, 0);
}

TEST(YawController, SimulationConvergesWithSimultaneousTranslationAndRotation) {
  Controller controller(ControlParameters{});
  Eigen::Vector2d camera(0, 0), target(1.5, .2);
  double heading = 0;
  VelocityCommand command;
  for (int i = 0; i < 1000; ++i) {
    const double c = std::cos(heading), sn = std::sin(heading);
    const Eigen::Vector2d delta = target - camera;
    const double forward = c * delta.x() + sn * delta.y();
    const double left = -sn * delta.x() + c * delta.y();
    command = controller.compute({-left, 0, forward}, kLevelUp,
      cube_rotation(25 - heading * 180 / std::acos(-1.0)), 1 + i * .05);
    camera += .05 * Eigen::Vector2d(c * command.vx - sn * command.vy,
                                   sn * command.vx + c * command.vy);
    heading += .05 * command.wz;
  }
  EXPECT_EQ(command.state, ControlState::InPosition);
  EXPECT_LE(std::abs(command.yaw_error_rad) * 180 / std::acos(-1.0), 5);
  EXPECT_EQ(command.vx, 0);
  EXPECT_EQ(command.vy, 0);
  EXPECT_EQ(command.wz, 0);
}

TEST(YawController, WireFeedbackTakesShortTurnWithoutGrowingError) {
  const double radians = std::acos(-1.0) / 180;
  for (double initial : {-44.9, -40., -12., -6., 6., 12., 40., 44.9}) {
    Controller controller(ControlParameters{});
    double heading = 0, previous_error = std::abs(initial);
    VelocityCommand command;
    for (int i = 0; i < 150; ++i) {
      command = controller.compute({0, 0, .3}, kLevelUp,
        cube_rotation(initial - heading / radians), 1 + i * .1);
      EXPECT_LE(std::abs(command.yaw_error_rad) / radians, 45);
      EXPECT_LE(std::abs(command.yaw_error_rad) / radians, previous_error + 1e-8);
      previous_error = std::abs(command.yaw_error_rad) / radians;
      // 当前实车反馈约定：串口正值使车头左转，使用真实线缆编码后的量化值积分。
      const auto frame = encode_velocity(0, 0, 0, command.wz);
      const unsigned bits = unsigned(frame[5]) | (unsigned(frame[6]) << 8);
      const int signed_value = bits < 32768 ? int(bits) : int(bits) - 65536;
      heading += .1 * signed_value / 10.0;
      EXPECT_LE(std::abs(heading) / radians, std::abs(initial) + 1e-8);
    }
    EXPECT_EQ(command.state, ControlState::InPosition);
    EXPECT_LE(std::abs(initial - heading / radians), 5);
    EXPECT_EQ(command.wz, 0);
  }
}

TEST(YawController, ValidatesYawParameters) {
  ControlParameters p;
  p.yaw_resume_deg = 4;
  EXPECT_THROW(Controller{p}, std::invalid_argument);
  p = ControlParameters{}; p.max_yaw_speed = .05;
  EXPECT_THROW(Controller{p}, std::invalid_argument);
  p = ControlParameters{}; p.yaw_confirm_frames = 1;
  EXPECT_THROW(Controller{p}, std::invalid_argument);
  p = ControlParameters{}; p.yaw_gain = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Controller{p}, std::invalid_argument);
  p = ControlParameters{}; p.yaw_enabled = false;
  Controller legacy(p);
  const auto command = legacy.compute({0, 0, .3}, kLevelUp, Eigen::Matrix3d::Zero(), 1);
  EXPECT_EQ(command.state, ControlState::InPosition);
  EXPECT_EQ(command.wz, 0);
}
}  // namespace
}  // namespace box_pose_control
