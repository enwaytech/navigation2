// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <thread>
#include <random>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_mppi_controller/motion_models.hpp"
#include "nav2_mppi_controller/critics/angular_velocity_speed_limit_critic.hpp"
#include "nav2_mppi_controller/critics/constraint_critic.hpp"
#include "nav2_mppi_controller/critics/goal_angle_critic.hpp"
#include "nav2_mppi_controller/critics/goal_critic.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "nav2_mppi_controller/critics/cost_critic.hpp"
#include "nav2_mppi_controller/critics/obstacle_bypass_critic.hpp"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"
#include "nav2_mppi_controller/critics/path_angle_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/critics/prefer_forward_critic.hpp"
#include "nav2_mppi_controller/critics/twirling_critic.hpp"
#include "nav2_mppi_controller/critics/velocity_deadband_critic.hpp"
#include "utils_test.cpp"  // NOLINT

// Tests the various critic plugin functions

// ROS lock used from utils_test.cpp

using namespace mppi;  // NOLINT
using namespace mppi::critics;  // NOLINT
using namespace mppi::utils;  // NOLINT

class PathAngleCriticWrapper : public PathAngleCritic
{
public:
  PathAngleCriticWrapper()
  : PathAngleCritic()
  {
  }

  void setMode(int mode)
  {
    mode_ = static_cast<PathAngleMode>(mode);
  }
};

TEST(CriticTests, ConstraintsCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  // provide velocities in constraints, should not have any costs
  state.vx = 0.40 * Eigen::ArrayXXf::Ones(1000, 30);
  state.vy = Eigen::ArrayXXf::Zero(1000, 30);
  state.wz = Eigen::ArrayXXf::Ones(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly and that defaults are reasonable
  ConstraintCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");
  EXPECT_TRUE(critic.getMaxVelConstraint() > 0.0);
  EXPECT_TRUE(critic.getMinVelConstraint() < 0.0);

  // Scoring testing
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0, 1e-6);

  // provide out of maximum velocity constraint
  state.vx.row(999).setConstant(0.60f);
  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  // 4.0 weight * 0.1 model_dt * 0.1 error introduced * 30 timesteps = 1.2
  EXPECT_NEAR(costs(999), 1.2, 0.01);
  costs.setZero();

  // provide out of minimum velocity constraint
  state.vx.row(1).setConstant(-0.45f);
  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  // 4.0 weight * 0.1 model_dt * 0.1 error introduced * 30 timesteps = 1.2
  EXPECT_NEAR(costs(1), 1.2, 0.01);
  costs.setZero();

  // Now with ackermann, all in constraint so no costs to score
  state.vx.setConstant(0.40f);
  state.wz.setConstant(1.5f);
  data.motion_model = std::make_shared<AckermannMotionModel>(&param_handler, node->get_name());
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0, 1e-6);

  // Now violating the ackermann constraints
  state.wz.setConstant(2.5f);
  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  // 4.0 weight * 0.1 model_dt * (0.2 - 0.4/2.5) * 30 timesteps = 0.48
  EXPECT_NEAR(costs(1), 0.48, 0.01);
}

TEST(CriticTests, ObstacleCriticMisalignedParams) {
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", true);

  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  ObstaclesCritic critic;
  // Expect throw when settings mismatched
  EXPECT_THROW(
    critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler),
    nav2_core::ControllerException
  );
}

TEST(CriticTests, ObstacleCriticAlignedParams) {
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", false);

  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  ObstaclesCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");
}


TEST(CriticTests, CostCriticMisAlignedParams) {
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", true);
  costmap_ros->on_configure(lstate);

  CostCritic critic;
  // Expect throw when settings mismatched
  EXPECT_THROW(
    critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler),
    nav2_core::ControllerException
  );
}

TEST(CriticTests, CostCriticAlignedParams) {
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", false);
  costmap_ros->on_configure(lstate);

  CostCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");
}

TEST(CriticTests, GoalAngleCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly
  GoalAngleCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path too far from `threshold_to_consider` to consider
  state.pose.pose.position.x = 1.0;
  path.x(9) = 10.0;
  path.y(9) = 0.0;
  path.yaws(9) = 3.14;
  goal.position.x = 10.0;
  goal.position.y = 0.0;
  goal.orientation.x = 0.0;
  goal.orientation.y = 0.0;
  goal.orientation.z = 1.0;
  goal.orientation.w = 0.0;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0, 1e-6);

  // Let's move it even closer, just to be sure it still doesn't trigger
  state.pose.pose.position.x = 9.2;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0, 1e-6);

  // provide state pose and path below `threshold_to_consider` to consider
  state.pose.pose.position.x = 9.7;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  EXPECT_NEAR(costs(0), 9.42, 0.02);  // (3.14 - 0.0) * 3.0 weight
}

TEST(CriticTests, GoalAngleCriticSymmetric)
{
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Make sure initializes correctly
  auto getParam = param_handler.getParamGetter("critic");
  bool symmetric_yaw_tolerance = true;
  getParam(symmetric_yaw_tolerance, "symmetric_yaw_tolerance", true);
  GoalAngleCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // provide state poses and path too far from `threshold_to_consider` to consider
  state.pose.pose.position.x = 9.7;
  path.x(9) = 10.0;
  path.y(9) = 0.0;
  path.yaws(9) = 3.14;
  goal.position.x = 10.0;
  goal.position.y = 0.0;
  goal.orientation.x = 0.0;
  goal.orientation.y = 0.0;
  goal.orientation.z = 1.0;
  goal.orientation.w = 0.0;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);

  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  EXPECT_NEAR(costs(0), 0, 0.02);  // Should be zero cost due to symmetry

  path.yaws(9) = 0.0;
  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  EXPECT_NEAR(costs(0), 0, 0.02);  // (0.0 - 0.0) * 3.0 weight

  path.yaws(9) = 1.57;
  critic.score(data);
  EXPECT_GT(costs.sum(), 0);
  EXPECT_NEAR(costs(0), 4.71, 0.02);  // (1.57 - 0.0) * 3.0 weight
}

TEST(CriticTests, GoalCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly
  GoalCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing with all trajectories set to 0

  // provide state poses and path far, should not trigger
  state.pose.pose.position.x = 1.0;
  path.x(9) = 10.0;
  path.y(9) = 0.0;
  goal.position.x = 10.0;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs(2), 0.0, 1e-6);  // (0 * 5.0 weight
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);  // Should all be 0 * 1000
  costs.setZero();

  // provide state pose and path close
  path.x(9) = 0.5;
  path.y(9) = 0.0;
  goal.position.x = 0.5;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs(2), 2.5, 1e-6);  // (sqrt(10.0 * 10.0) * 5.0 weight
  EXPECT_NEAR(costs.sum(), 2500.0, 1e-3);  // should be 2.5 * 1000
}

TEST(CriticTests, PathAngleCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally

  // Initialization testing

  // Make sure initializes correctly
  PathAngleCriticWrapper critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path close, within pose tolerance so won't do anything
  state.pose.pose.position.x = 0.0;
  state.pose.pose.position.y = 0.0;
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with less than PI/2 angular diff.
  path.x(9) = 0.95;
  goal.position.x = 0.95;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  data.furthest_reached_path_point = 2;  // So it grabs the 2 + offset_from_furthest_ = 6th point
  path.x(6) = 1.0;  // angle between path point and pose = 0 < max_angle_to_furthest_
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path point and pose > max_angle_to_furthest_
  path.y(6) = 4.0;
  critic.score(data);
  EXPECT_GT(costs.sum(), 0.0);
  EXPECT_NEAR(costs(0), 3.9947, 1e-2);  // atan2(4,-1) [1.81] * 2.2 weight

  // Set mode to no directional preferences + reset costs
  critic.setMode(1);
  costs.setZero();

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = 1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path pt and pose < max_angle_to_furthest_ IF non-directional
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 4.0;
  critic.score(data);
  EXPECT_GT(costs.sum(), 0.0);
  // should use reverse orientation as the closer angle in no dir preference mode
  EXPECT_NEAR(costs(0), 2.9167, 1e-2);

  // Set mode to consider path directionality + reset costs
  critic.setMode(2);
  costs.setZero();

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = 1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path pt and pose < max_angle_to_furthest_ IF non-directional
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 4.0;
  critic.score(data);
  EXPECT_GT(costs.sum(), 0.0);
  // should use reverse orientation as the closer angle in no dir preference mode
  EXPECT_NEAR(costs(0), 2.9167, 1e-2);

  PathAngleMode mode;
  mode = PathAngleMode::FORWARD_PREFERENCE;
  EXPECT_EQ(modeToStr(mode), std::string("Forward Preference"));
  mode = PathAngleMode::CONSIDER_FEASIBLE_PATH_ORIENTATIONS;
  EXPECT_EQ(modeToStr(mode), std::string("Consider Feasible Path Orientations"));
  mode = PathAngleMode::NO_DIRECTIONAL_PREFERENCE;
  EXPECT_EQ(modeToStr(mode), std::string("No Directional Preference"));
  mode = static_cast<PathAngleMode>(4);
  EXPECT_EQ(modeToStr(mode), std::string("Invalid mode!"));
}

TEST(CriticTests, PreferForwardCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally

  // Initialization testing

  // Make sure initializes correctly
  PreferForwardCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path far away, not within positional tolerances
  state.pose.pose.position.x = 1.0;
  path.x(9) = 10.0;
  goal.position.x = 10.0;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0f, 1e-6f);

  // provide state pose and path close to trigger behavior but with all forward motion
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  state.vx.setOnes();
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0f, 1e-6f);

  // provide state pose and path close to trigger behavior but with all reverse motion
  state.vx.setConstant(-1.0f);
  critic.score(data);
  EXPECT_GT(costs.sum(), 0.0f);
  EXPECT_NEAR(costs(0), 15.0f, 1e-3f);  // 1.0 * 0.1 model_dt * 5.0 weight * 30 length
}

TEST(CriticTests, TwirlingCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally
  data.goal_checker = &goal_checker;

  // Initialization testing

  // Make sure initializes correctly
  TwirlingCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path far away, not within positional tolerances
  state.pose.pose.position.x = 1.0;
  path.x(9) = 10.0;
  goal.position.x = 10.0;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path close to trigger behavior but with no angular variation
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  state.wz.setZero();
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // Provide nearby with some motion
  state.wz.row(0).setConstant(10.0f);
  critic.score(data);
  EXPECT_NEAR(costs(0), 100.0, 1e-6);  // (mean(10.0) * 10.0 weight
  costs.setZero();

  // Now try again with some wiggling noise
  std::mt19937 engine;
  std::normal_distribution<float> normal_dist = std::normal_distribution(0.0f, 0.5f);
  state.wz.row(0) = Eigen::ArrayXf::NullaryExpr(30, [&]() {return normal_dist(engine);});
  critic.score(data);
  EXPECT_NEAR(costs(0), 2.581, 4e-1);  // (mean of noise with mu=0, sigma=0.5 * 10.0 weight
}

TEST(CriticTests, PathFollowCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(6);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally
  data.goal_checker = &goal_checker;

  // Initialization testing

  // Make sure initializes correctly
  PathFollowCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and goal close within positional tolerances
  state.pose.pose.position.x = 2.0;
  path.x(5) = 1.8;
  goal.position.x = 1.8;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path far enough to enable
  // pose differential is (0, 0) and (0.15, 0)
  path.x(5) = 0.15;
  goal.position.x = 0.15;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 750.0, 1e-2);  // 0.15 * 5 weight * 1000
}

TEST(CriticTests, PathAlignCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally
  data.goal_checker = &goal_checker;

  // Initialization testing

  // Make sure initializes correctly
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path close within positional tolerances
  state.pose.pose.position.x = 1.0;
  path.x(9) = 0.85;
  goal.position.x = 0.85;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path far enough to enable
  // but data furthest point reached is 0 and offset default is 20, so returns
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path far enough to enable, with data to pass condition
  // but with empty trajectories and paths, should still be zero
  *data.furthest_reached_path_point = 21;
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // provide state pose and path far enough to enable, with data to pass condition
  // and with a valid path to pass invalid path condition
  state.pose.pose.position.x = 0.0;
  data.path_pts_valid.reset();  // Recompute on new path
  path.reset(22);
  path.x(0) = 0;
  path.x(1) = 0.1;
  path.x(2) = 0.2;
  path.x(3) = 0.3;
  path.x(4) = 0.4;
  path.x(5) = 0.5;
  path.x(6) = 0.6;
  path.x(7) = 0.7;
  path.x(8) = 0.8;
  path.x(9) = 0.9;
  path.x(10) = 0.9;
  path.x(11) = 0.9;
  path.x(12) = 0.9;
  path.x(13) = 0.9;
  path.x(14) = 0.9;
  path.x(15) = 0.9;
  path.x(16) = 0.9;
  path.x(17) = 0.9;
  path.x(18) = 0.9;
  path.x(19) = 0.9;
  path.x(20) = 0.9;
  path.x(21) = 0.9;
  goal.position.x = 0.9;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  generated_trajectories.x.setConstant(0.66f);
  critic.score(data);
  // 0.66 * 1000 * 10 weight * 6 num pts eval / 6 normalization term
  EXPECT_NEAR(costs.sum(), 6600.0, 1e-2);

  // provide state pose and path far enough to enable, with data to pass condition
  // but path is blocked in collision
  auto * costmap = costmap_ros->getCostmap();
  // island in the middle of lethal cost to cross. Costmap defaults to size 5x5 @ 10cm resolution
  for (unsigned int i = 11; i <= 30; ++i) {  // 1.1m-3m
    for (unsigned int j = 11; j <= 30; ++j) {  // 1.1m-3m
      costmap->setCost(i, j, 254);
    }
  }

  data.path_pts_valid.reset();  // Recompute on new path
  costs.setZero();
  path.x.setConstant(1.5f);
  path.y.setConstant(1.5f);
  goal.position.x = 1.5;
  state.local_path_length = std::abs(state.pose.pose.position.x - goal.position.x);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);
}

TEST(CriticTests, VelocityDeadbandCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  auto getParam = param_handler.getParamGetter("critic");
  std::vector<double> deadband_velocities_;
  getParam(deadband_velocities_, "deadband_velocities", std::vector<double>{0.08, 0.08, 0.08});
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<OmniMotionModel>();

  // Initialization testing

  // Make sure initializes correctly and that defaults are reasonable
  VelocityDeadbandCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide velocities out of deadband bounds, should not have any costs
  state.vx.setConstant(0.80f);
  state.vy.setConstant(0.60f);
  state.wz.setConstant(0.80f);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0, 1e-6);

  // Test cost value
  state.vx.setConstant(0.01f);
  state.vy.setConstant(0.02f);
  state.wz.setConstant(0.021f);
  critic.score(data);
  // 35.0 weight * 0.1 model_dt * (0.07 + 0.06 + 0.059) * 30 timesteps = 56.7
  EXPECT_NEAR(costs(1), 19.845, 0.01);
}

TEST(CriticTests, ObstacleBypassCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  // 70-point path so target_idx (45) < path_segments_count (69) avoids zero tangent
  path.reset(70);
  for (int i = 0; i < 70; ++i) {
    path.x(i) = 0.5f + i * 0.1f;
    path.y(i) = 2.5f;
  }
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1000);
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Obstacle at cells [25,30]x[22,27] = world [2.5,3.0]x[2.2,2.7]
  // Path at y=2.5 (cell_y=25) passes through obstacle. Indices 20-25 are blocked.
  // Perpendicular scan from obstacle: left(y+) free at s=3, right(y-) free at s=4
  auto * costmap = costmap_ros->getCostmap();
  for (unsigned int i = 25; i <= 30; ++i) {
    for (unsigned int j = 22; j <= 27; ++j) {
      costmap->setCost(i, j, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
  }

  // Initialization
  ObstacleBypassCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // -- Scenario 1: Early exit - path too short --
  state.local_path_length = 0.3f;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // -- Scenario 2: Early exit - not advanced enough on path --
  state.local_path_length = 6.9f;
  data.furthest_reached_path_point = 10;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // -- Scenario 3: No obstacles - clear path (!path_blocked && !bypass_active_) --
  data.furthest_reached_path_point = 25;
  data.path_pts_valid = std::vector<bool>(69, true);
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // -- Scenario 4: Path blocked - left side bypass with exact costs --
  // findPathCosts marks indices 20-25 invalid from costmap. 5 counted (0..24).
  // occupancy_ratio=5/25=0.2, path_blocked=true
  // blocked_idx=20, resume_idx=26, obstacle_idx=23, path_yaw=0
  // computeBypassTarget: left preferred (fewer lethal+inflated cells above path)
  // signed_offset = +(4*0.1+1.0) = +1.4, target_idx=45, target=(5.0, 3.9)
  data.path_pts_valid = std::nullopt;
  data.furthest_reached_path_point = 25;
  generated_trajectories.x.col(29).setConstant(5.0f);
  generated_trajectories.y.col(29).setConstant(3.9f);
  generated_trajectories.x(1, 29) = 5.0f;  generated_trajectories.y(1, 29) = 2.9f;
  generated_trajectories.x(2, 29) = 4.0f;  generated_trajectories.y(2, 29) = 3.9f;
  generated_trajectories.x(3, 29) = 5.0f;  generated_trajectories.y(3, 29) = 1.9f;

  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0, 1e-2);
  EXPECT_NEAR(costs(1), 4.667, 0.02);   // weight * 1.0m
  EXPECT_NEAR(costs(2), 4.667, 0.02);   // weight * 1.0m
  EXPECT_NEAR(costs(3), 9.333, 0.02);   // weight * 2.0m
  EXPECT_LT(costs(1), costs(3));
  costs.setZero();

  // -- Scenario 5: Hysteresis keeps bypass active --
  // bypass_active_=true from Scenario 4. Pre-set 2 invalid pts: ratio=2/25=0.08
  // path_blocked=(0.08>0.07 && 2>2.0)=false. 0.08 >= 0.035 → bypass stays active.
  // Same obstacle in costmap → same bypass side/offset → same target (5.0, 3.9)
  data.path_pts_valid = std::vector<bool>(69, true);
  (*data.path_pts_valid)[22] = false;
  (*data.path_pts_valid)[23] = false;
  data.furthest_reached_path_point = 25;

  critic.score(data);
  EXPECT_GT(costs.sum(), 0.0);
  EXPECT_NEAR(costs(1), 4.667, 0.02);   // traj 1 is 1.0m from target
  costs.setZero();

  // -- Scenario 6: Hysteresis deactivation --
  // bypass_active_=true. All valid: ratio=0.0 < 0.035 → deactivates, returns
  data.path_pts_valid = std::vector<bool>(69, true);
  data.furthest_reached_path_point = 25;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // -- Scenario 7: Power parameter, power=2 --
  node->set_parameter(rclcpp::Parameter("critic.cost_power", 2));
  critic = ObstacleBypassCritic();
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  data.path_pts_valid = std::nullopt;
  data.furthest_reached_path_point = 25;
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0, 1e-2);
  EXPECT_NEAR(costs(1), 21.78, 0.1);    // (weight * 1.0)^2
  EXPECT_NEAR(costs(3), 87.11, 0.1);    // (weight * 2.0)^2
  costs.setZero();

  // -- Scenario 8: Right side preferred bypass --
  // Extend obstacle to [25,30]x[22,29]: more cells above path than below
  // Right side has fewer lethal cells → right preferred
  // Verify by checking trajectory below path costs less than trajectory above
  for (unsigned int i = 25; i <= 30; ++i) {
    for (unsigned int j = 28; j <= 29; ++j) {
      costmap->setCost(i, j, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
  }
  node->set_parameter(rclcpp::Parameter("critic.cost_power", 1));
  critic = ObstacleBypassCritic();
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  data.path_pts_valid = std::nullopt;
  data.furthest_reached_path_point = 25;
  generated_trajectories.x.col(29).setConstant(5.0f);
  generated_trajectories.y(0, 29) = 0.5f;   // far below path (toward right bypass)
  generated_trajectories.y(1, 29) = 4.5f;   // far above path (away from right bypass)

  critic.score(data);
  EXPECT_GT(costs.sum(), 0.0);
  EXPECT_LT(costs(0), costs(1));   // below path (right side) is closer to target
  costs.setZero();

  // Restore obstacle to original [25,30]x[22,27]
  for (unsigned int i = 25; i <= 30; ++i) {
    for (unsigned int j = 28; j <= 29; ++j) {
      costmap->setCost(i, j, nav2_costmap_2d::FREE_SPACE);
    }
  }

  // -- Scenario 9: Blocked to end of path --
  // resume_idx scans to end without finding valid → returns early
  critic = ObstacleBypassCritic();
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  data.path_pts_valid = std::vector<bool>(69, true);
  for (int i = 20; i < 69; ++i) {
    (*data.path_pts_valid)[i] = false;
  }
  data.furthest_reached_path_point = 25;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);

  // -- Scenario 10: Degenerate tangent (tangent_len < 1e-6) --
  critic = ObstacleBypassCritic();
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  float saved_x22 = path.x(22);
  float saved_x23 = path.x(23);
  path.x(22) = 2.75f;
  path.x(23) = 2.75f;
  data.path_pts_valid = std::vector<bool>(69, true);
  (*data.path_pts_valid)[21] = false;
  (*data.path_pts_valid)[22] = false;
  (*data.path_pts_valid)[23] = false;
  data.furthest_reached_path_point = 25;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);
  path.x(22) = saved_x22;
  path.x(23) = saved_x23;

  // -- Scenario 11: Both sides blocked (computeBypassTarget returns nullopt) --
  for (unsigned int i = 0; i < costmap->getSizeInCellsX(); ++i) {
    for (unsigned int j = 0; j < costmap->getSizeInCellsY(); ++j) {
      costmap->setCost(i, j, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
  }
  critic = ObstacleBypassCritic();
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  data.path_pts_valid = std::vector<bool>(69, true);
  for (int i = 20; i <= 25; ++i) {
    (*data.path_pts_valid)[i] = false;
  }
  data.furthest_reached_path_point = 25;
  critic.score(data);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-6);
}

TEST(CriticTests, AngularVelocitySpeedLimitCritic)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  node->declare_parameter("critic.min_angular_velocity", rclcpp::ParameterValue(0.125));
  node->declare_parameter("critic.max_angular_velocity", rclcpp::ParameterValue(0.375));
  node->declare_parameter("critic.min_speed", rclcpp::ParameterValue(0.25));
  node->declare_parameter("critic.max_speed", rclcpp::ParameterValue(1.375));
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1003, 57);
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1003);
  float model_dt = 0.05f;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<AckermannMotionModel>(&param_handler, node->get_name());

  AngularVelocitySpeedLimitCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Below min_angular_velocity there is no cost, whatever the speed
  state.vx.setConstant(1.375f);
  state.wz.setConstant(0.1f);
  critic.score(data);
  EXPECT_EQ(costs.sum(), 0.0f);

  // Above max_angular_velocity speeds over min_speed are penalized:
  // 10 weight * (1.0 - 0.25) * 0.05 dt * 57 steps = 21.375
  state.vx.setConstant(1.0f);
  state.wz.setConstant(-0.5f);
  critic.score(data);
  EXPECT_NEAR(costs(0), 21.375, 1e-3);
  EXPECT_NEAR(costs(1002), 21.375, 1e-3);

  // Random velocities: same result as the original row-wise expression. Only up to rounding
  // here, because the compiler may contract the reference expression differently in this file.
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> vx_dist(-0.5f, 1.5f), vy_dist(-0.3f, 0.3f),
  wz_dist(-0.8f, 0.8f);
  for (Eigen::Index i = 0; i < state.vx.size(); i++) {
    state.vx(i) = vx_dist(gen);
    state.vy(i) = vy_dist(gen);
    state.wz(i) = wz_dist(gen);
  }
  state.wz(0, 3) = 0.125f;  // at min_angular_velocity
  state.wz(1, 3) = -0.375f;  // at max_angular_velocity

  const float min_wz = 0.125f, max_wz = 0.375f, min_speed = 0.25f, max_speed = 1.375f;
  const auto wz = state.wz.abs();
  const auto speed = (state.vx.square() + state.vy.square()).sqrt();
  const auto wz_ratio = (wz / max_wz).min(1.0f);
  const auto allowed_speed = (1.0f - wz_ratio) * max_speed + wz_ratio * min_speed;
  const auto speed_violation = (speed - allowed_speed).max(0.0f);
  const auto mask = (wz >= min_wz).cast<float>();
  const Eigen::ArrayXf per_traj = (mask * speed_violation * model_dt).rowwise().sum().eval();

  costs.setZero();
  critic.score(data);
  const Eigen::ArrayXf expected = (per_traj * 10.0f).eval();
  EXPECT_GT(expected.sum(), 0.0f);
  for (Eigen::Index i = 0; i < costs.size(); i++) {
    EXPECT_NEAR(costs(i), expected(i), 1e-5f * expected(i) + 1e-6f) << i;
  }

  // Power
  node->set_parameter(rclcpp::Parameter("critic.cost_power", 2));
  critic = AngularVelocitySpeedLimitCritic();
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  costs.setZero();
  critic.score(data);
  const Eigen::ArrayXf expected_pow = (per_traj * 10.0f).pow(2u).eval();
  for (Eigen::Index i = 0; i < costs.size(); i++) {
    EXPECT_NEAR(costs(i), expected_pow(i), 1e-5f * expected_pow(i) + 1e-6f) << i;
  }
}

// The cost of PathAlignCritic, computed trajectory by trajectory through strided maps and
// angles::shortest_angular_distance
Eigen::ArrayXf referencePathAlignCost(
  const CriticData & data, const std::vector<bool> & path_pts_valid, int trajectory_point_step,
  bool use_path_orientations)
{
  const size_t path_segments_count = data.path.x.size() - 1;
  std::vector<float> path_integrated_distances(path_segments_count, 0.0f);
  std::vector<utils::Pose2D> path(path_segments_count);
  float dx = 0.0f, dy = 0.0f;
  for (unsigned int i = 1; i != path_segments_count; i++) {
    auto & pose = path[i - 1];
    pose.x = data.path.x(i - 1);
    pose.y = data.path.y(i - 1);
    pose.theta = data.path.yaws(i - 1);
    dx = data.path.x(i) - pose.x;
    dy = data.path.y(i) - pose.y;
    path_integrated_distances[i] = path_integrated_distances[i - 1] + sqrtf(dx * dx + dy * dy);
  }
  auto & final_pose = path[path_segments_count - 1];
  final_pose.x = data.path.x(path_segments_count - 1);
  final_pose.y = data.path.y(path_segments_count - 1);
  final_pose.theta = data.path.yaws(path_segments_count - 1);

  const int rows = data.trajectories.x.rows();
  const int cols = floor((data.trajectories.x.cols() - 1) / trajectory_point_step) + 1;
  const int outer_stride = rows * trajectory_point_step;
  using StridedMap = Eigen::Map<const Eigen::ArrayXXf, 0, Eigen::Stride<-1, -1>>;
  const StridedMap T_x(
    data.trajectories.x.data(), rows, cols, Eigen::Stride<-1, -1>(outer_stride, 1));
  const StridedMap T_y(
    data.trajectories.y.data(), rows, cols, Eigen::Stride<-1, -1>(outer_stride, 1));
  const StridedMap T_yaw(
    data.trajectories.yaws.data(), rows, cols, Eigen::Stride<-1, -1>(outer_stride, 1));

  Eigen::ArrayXf cost = Eigen::ArrayXf::Zero(rows);
  for (int t = 0; t < rows; ++t) {
    float summed_path_dist = 0.0f, traj_integrated_distance = 0.0f;
    unsigned int num_samples = 0u, path_pt = 0u;
    float Tx_m1 = T_x(t, 0);
    float Ty_m1 = T_y(t, 0);
    for (int p = 1; p < cols; p++) {
      const float Tx = T_x(t, p);
      const float Ty = T_y(t, p);
      dx = Tx - Tx_m1;
      dy = Ty - Ty_m1;
      Tx_m1 = Tx;
      Ty_m1 = Ty;
      traj_integrated_distance += sqrtf(dx * dx + dy * dy);
      path_pt = utils::findClosestPathPt(
        path_integrated_distances, traj_integrated_distance, path_pt);
      if (path_pts_valid[path_pt]) {
        const auto & pose = path[path_pt];
        dx = pose.x - Tx;
        dy = pose.y - Ty;
        num_samples++;
        if (use_path_orientations) {
          const float dyaw = angles::shortest_angular_distance(pose.theta, T_yaw(t, p));
          summed_path_dist += sqrtf(dx * dx + dy * dy + dyaw * dyaw);
        } else {
          summed_path_dist += sqrtf(dx * dx + dy * dy);
        }
      }
    }
    cost(t) = num_samples > 0u ? summed_path_dist / static_cast<float>(num_samples) : 0.0f;
  }
  return cost;
}

TEST(CriticTests, PathAlignCriticMatchesReference)
{
  // Standard preamble
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  node->declare_parameter("critic.use_path_orientations", rclcpp::ParameterValue(true));
  node->declare_parameter("critic.cost_weight", rclcpp::ParameterValue(14.0));
  node->declare_parameter("critic.trajectory_point_step", rclcpp::ParameterValue(4));
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::Trajectories trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  // Batch size not a multiple of the critic's block size
  const int batch_size = 300;
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(batch_size);
  float model_dt = 0.05f;
  CriticData data =
  {state, trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Curved path with 5 cm spacing, its last points invalid
  path.reset(200);
  float yaw = 0.3f;
  path.yaws(0) = yaw;
  for (Eigen::Index i = 1; i < path.x.size(); i++) {
    yaw += 0.02f;
    path.x(i) = path.x(i - 1) + 0.05f * std::cos(yaw);
    path.y(i) = path.y(i - 1) + 0.05f * std::sin(yaw);
    path.yaws(i) = yaw;
  }
  std::vector<bool> valid(path.x.size() - 1, true);
  for (size_t i = 150; i < 160; i++) {
    valid[i] = false;
  }

  // Random trajectories with headings beyond pi, so that angles get normalized
  std::mt19937 gen(3);
  std::uniform_real_distribution<float> speed(-0.3f, 1.4f), turn(-0.15f, 0.15f);
  trajectories.reset(batch_size, 101);
  for (int t = 0; t < batch_size; t++) {
    float traj_yaw = (t % 5) * 2.5f - 5.0f;
    const float v = speed(gen);
    trajectories.yaws(t, 0) = traj_yaw;
    for (Eigen::Index j = 1; j < trajectories.x.cols(); j++) {
      traj_yaw += turn(gen);
      trajectories.x(t, j) = trajectories.x(t, j - 1) + 0.05f * v * std::cos(traj_yaw);
      trajectories.y(t, j) = trajectories.y(t, j - 1) + 0.05f * v * std::sin(traj_yaw);
      trajectories.yaws(t, j) = traj_yaw;
    }
  }
  state.local_path_length = 10.0f;

  for (const int step : {4, 3, 1}) {
    for (const bool use_orientations : {true, false}) {
      node->set_parameter(rclcpp::Parameter("critic.trajectory_point_step", step));
      node->set_parameter(rclcpp::Parameter("critic.use_path_orientations", use_orientations));
      PathAlignCritic critic;
      critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
      data.furthest_reached_path_point = 30;
      data.path_pts_valid = valid;
      costs.setZero();
      critic.score(data);
      const Eigen::ArrayXf expected =
        (referencePathAlignCost(data, valid, step, use_orientations) * 14.0f).eval();
      EXPECT_GT(expected.sum(), 0.0f);
      for (Eigen::Index i = 0; i < costs.size(); i++) {
        EXPECT_NEAR(costs(i), expected(i), 1e-5f * expected(i) + 1e-6f) << step << " " << i;
      }
    }
  }
}
