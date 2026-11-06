#include "box_pose_control/controller.hpp"

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
  parameters.max_forward_speed = 0.301;
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
}

TEST(Controller, UsesCorrectLeftRightAndForwardBackwardSigns) {
  Controller controller(ControlParameters{});
  auto command = controller.compute({0.05, 0.8, 0.30}, kLevelUp);
  EXPECT_NEAR(command.vx, 0.0, 1e-12);
  EXPECT_NEAR(command.vy, -0.05, 1e-12);
  EXPECT_EQ(command.state, ControlState::Aligning);
  command = controller.compute({-0.05, -0.8, 0.30}, kLevelUp);
  EXPECT_NEAR(command.vy, 0.05, 1e-12);
  command = controller.compute({0.0, 10.0, 0.36}, kLevelUp);
  EXPECT_NEAR(command.vx, 0.06, 1e-12);
  EXPECT_NEAR(command.vy, 0.0, 1e-12);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  command = controller.compute({0.0, -10.0, 0.24}, kLevelUp);
  EXPECT_NEAR(command.vx, -0.06, 1e-12);
}

TEST(Controller, AlignsBeforeAdjustingDistance) {
  Controller controller(ControlParameters{});
  const auto align = controller.compute({0.05, 0.0, 0.60}, kLevelUp);
  EXPECT_EQ(align.state, ControlState::Aligning);
  EXPECT_NEAR(align.vx, 0.0, 1e-12);
  EXPECT_LT(align.vy, 0.0);
  const auto distance = controller.compute({0.009, 0.0, 0.60}, kLevelUp);
  EXPECT_EQ(distance.state, ControlState::AdjustingDistance);
  EXPECT_GT(distance.vx, 0.0);
  EXPECT_NEAR(distance.vy, 0.0, 1e-12);
}

TEST(Controller, LateralHysteresisAvoidsBoundaryChatterAndResetClearsIt) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.012, 0.0, 0.50}, kLevelUp).state,
            ControlState::Aligning);
  EXPECT_EQ(controller.compute({0.010, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingDistance);
  EXPECT_EQ(controller.compute({0.015, 0.0, 0.50}, kLevelUp).state,
            ControlState::AdjustingDistance);
  EXPECT_EQ(controller.compute({0.016, 0.0, 0.50}, kLevelUp).state,
            ControlState::Aligning);
  EXPECT_EQ(controller.compute({-0.012, 0.0, 0.50}, kLevelUp).state,
            ControlState::Aligning);
  controller.compute({0.0, 0.0, 0.50}, kLevelUp);
  controller.reset();
  EXPECT_EQ(controller.compute({0.012, 0.0, 0.50}, kLevelUp).state,
            ControlState::Aligning);
}

TEST(Controller, DistanceHysteresisHoldsUntilResumeThreshold) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.325}, kLevelUp).state,
            ControlState::AdjustingDistance);
  const auto in_position = controller.compute({0.0, 0.0, 0.31}, kLevelUp);
  EXPECT_EQ(in_position.state, ControlState::InPosition);
  EXPECT_DOUBLE_EQ(in_position.vx, 0.0);
  EXPECT_DOUBLE_EQ(in_position.vy, 0.0);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.325}, kLevelUp).state,
            ControlState::InPosition);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.331}, kLevelUp).state,
            ControlState::AdjustingDistance);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.275}, kLevelUp).state,
            ControlState::AdjustingDistance);
  controller.compute({0.0, 0.0, 0.30}, kLevelUp);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.275}, kLevelUp).state,
            ControlState::InPosition);
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.269}, kLevelUp).state,
            ControlState::AdjustingDistance);
}

TEST(Controller, ChangingTargetResetsHysteresisAndUsesNewDistance) {
  Controller controller(ControlParameters{});
  EXPECT_EQ(controller.compute({0.0, 0.0, 0.30}, kLevelUp).state,
            ControlState::InPosition);
  controller.set_target_distance(0.275);
  const auto command = controller.compute({0.0, 0.0, 0.30}, kLevelUp);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  EXPECT_NEAR(command.vx, 0.025, 1e-12);
  controller.set_target_distance(0.30);
  EXPECT_EQ(controller.compute({0.012, 0.0, 0.30}, kLevelUp).state,
            ControlState::Aligning);
}

TEST(Controller, CompensatesPurePitchAndNormalizesUp) {
  Controller controller(ControlParameters{});
  const Eigen::Vector3d up(0.0, -std::sqrt(0.75), -0.5);
  const auto command = controller.compute({0.0, 0.0, 0.35}, up * 4.0);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  EXPECT_NEAR(command.vx, 0.05 / std::sqrt(0.75), 1e-12);
  EXPECT_NEAR(command.vy, 0.0, 1e-12);
  const Eigen::Vector3d camera_motion = horizontal_axes(up) * Eigen::Vector2d(command.vx, command.vy);
  EXPECT_NEAR(camera_motion.x(), 0.0, 1e-12);
  EXPECT_NEAR(camera_motion.z(), 0.05, 1e-12);
}

TEST(Controller, CompensatesPureRollAndCombinedPitchRoll) {
  Controller controller(ControlParameters{});
  const Eigen::Vector3d roll_up(0.5, -std::sqrt(0.75), 0.0);
  auto command = controller.compute({0.05, 0.0, 0.60}, roll_up);
  EXPECT_EQ(command.state, ControlState::Aligning);
  EXPECT_NEAR(command.vx, 0.0, 1e-12);
  EXPECT_NEAR(command.vy, -0.05 / std::sqrt(0.75), 1e-12);
  const Eigen::Vector3d combined_up(0.3, -std::sqrt(0.66), -0.5);
  command = controller.compute({0.0, 0.0, 0.35}, combined_up);
  const Eigen::Vector3d camera_motion = horizontal_axes(combined_up) * Eigen::Vector2d(command.vx, command.vy);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  EXPECT_GT(std::abs(command.vy), 0.0);
  EXPECT_NEAR(camera_motion.x(), 0.0, 1e-12);
  EXPECT_NEAR(camera_motion.z(), 0.05, 1e-12);
}

TEST(Controller, ScalesBothAxesTogetherToPreserveCompensation) {
  ControlParameters parameters;
  parameters.max_forward_speed = 0.10;
  parameters.max_lateral_speed = 0.02;
  Controller controller(parameters);
  const Eigen::Vector3d up(0.5, -std::sqrt(0.39), -0.6);
  const auto command = controller.compute({0.0, 0.0, 1.0}, up);
  const Eigen::Vector3d camera_motion = horizontal_axes(up) * Eigen::Vector2d(command.vx, command.vy);
  EXPECT_EQ(command.state, ControlState::AdjustingDistance);
  EXPECT_NEAR(std::abs(command.vy), parameters.max_lateral_speed, 1e-12);
  EXPECT_LT(std::abs(command.vx), parameters.max_forward_speed);
  EXPECT_NEAR(camera_motion.x(), 0.0, 1e-12);
  EXPECT_GT(camera_motion.z(), 0.0);
  EXPECT_LT(camera_motion.z(), 0.70);
}

TEST(Controller, RejectsDegenerateGeometryAndNonfiniteInputs) {
  Controller controller(ControlParameters{});
  const Eigen::Vector3d invalid_ups[] = {
      Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d::UnitX(),
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
  for (int tick = 0; tick < 400; ++tick) {
    command = controller.compute(center, up);
    ASSERT_NE(command.state, ControlState::InvalidGeometry);
    EXPECT_LE(std::abs(command.vx), parameters.max_forward_speed + 1e-12);
    EXPECT_LE(std::abs(command.vy), parameters.max_lateral_speed + 1e-12);
    const Eigen::Vector3d previous = center;
    center -= 0.1 * axes * Eigen::Vector2d(command.vx, command.vy);
    if (command.state == ControlState::Aligning) {
      EXPECT_NEAR(center.z(), previous.z(), 1e-12);
    }
    if (command.state == ControlState::AdjustingDistance) {
      EXPECT_NEAR(center.x(), previous.x(), 1e-12);
    }
  }
  EXPECT_EQ(command.state, ControlState::InPosition);
  EXPECT_LE(std::abs(center.x()), parameters.lateral_tolerance_m);
  EXPECT_LE(std::abs(center.z() - parameters.target_distance_m), parameters.distance_tolerance_m);
  EXPECT_DOUBLE_EQ(command.vx, 0.0);
  EXPECT_DOUBLE_EQ(command.vy, 0.0);
}

}  // namespace
}  // namespace box_pose_control
