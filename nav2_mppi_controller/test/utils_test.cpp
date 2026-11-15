// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <random>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_mppi_controller/models/path.hpp"

// Tests noise generator object

using namespace mppi::utils;  // NOLINT
using namespace mppi;  // NOLINT

class TestGoalChecker : public nav2_core::GoalChecker
{
public:
  TestGoalChecker() {}

  virtual void initialize(
    const nav2::LifecycleNode::WeakPtr & /*parent*/,
    const std::string & /*plugin_name*/,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS>/*costmap_ros*/) {}

  virtual void reset() {}

  virtual bool isGoalReached(
    const geometry_msgs::msg::Pose & /*query_pose*/,
    const geometry_msgs::msg::Pose & /*goal_pose*/,
    const geometry_msgs::msg::Twist & /*velocity*/,
    const nav_msgs::msg::Path & /*transformed_global_plan*/) {return false;}

  virtual bool getTolerances(
    geometry_msgs::msg::Pose & pose_tolerance,
    geometry_msgs::msg::Twist & /*vel_tolerance*/)
  {
    pose_tolerance.position.x = 0.25;
    pose_tolerance.position.y = 0.25;
    return true;
  }
};

TEST(UtilsTests, MarkerPopulationUtils)
{
  auto pose = createPose(1.0, 2.0, 3.0);
  EXPECT_EQ(pose.position.x, 1.0);
  EXPECT_EQ(pose.position.y, 2.0);
  EXPECT_EQ(pose.position.z, 3.0);
  EXPECT_EQ(pose.orientation.w, 1.0);

  auto scale = createScale(1.0, 2.0, 3.0);
  EXPECT_EQ(scale.x, 1.0);
  EXPECT_EQ(scale.y, 2.0);
  EXPECT_EQ(scale.z, 3.0);

  auto color = createColor(1.0, 2.0, 3.0, 0.0);
  EXPECT_EQ(color.r, 1.0);
  EXPECT_EQ(color.g, 2.0);
  EXPECT_EQ(color.b, 3.0);
  EXPECT_EQ(color.a, 0.0);

  auto marker = createMarker(999, pose, scale, color, "map", "ns");
  EXPECT_EQ(marker.header.frame_id, "map");
  EXPECT_EQ(marker.id, 999);
  EXPECT_EQ(marker.pose, pose);
  EXPECT_EQ(marker.scale, scale);
  EXPECT_EQ(marker.color, color);
  EXPECT_EQ(marker.ns, "ns");
}

TEST(UtilsTests, ConversionTests)
{
  geometry_msgs::msg::TwistStamped output;
  builtin_interfaces::msg::Time time;

  // Check population is correct
  output = toTwistStamped(0.5, 0.3, time, "map");
  EXPECT_NEAR(output.twist.linear.x, 0.5, 1e-6);
  EXPECT_NEAR(output.twist.linear.y, 0.0, 1e-6);
  EXPECT_NEAR(output.twist.angular.z, 0.3, 1e-6);
  EXPECT_EQ(output.header.frame_id, "map");
  EXPECT_EQ(output.header.stamp, time);

  output = toTwistStamped(0.5, 0.4, 0.3, time, "map");
  EXPECT_NEAR(output.twist.linear.x, 0.5, 1e-6);
  EXPECT_NEAR(output.twist.linear.y, 0.4, 1e-6);
  EXPECT_NEAR(output.twist.angular.z, 0.3, 1e-6);
  EXPECT_EQ(output.header.frame_id, "map");
  EXPECT_EQ(output.header.stamp, time);

  nav_msgs::msg::Path path;
  path.poses.resize(5);
  path.poses[2].pose.position.x = 5;
  path.poses[2].pose.position.y = 50;
  models::Path path_t = toTensor(path);

  // Check population is correct
  EXPECT_EQ(path_t.x.rows(), 5u);
  EXPECT_EQ(path_t.y.rows(), 5u);
  EXPECT_EQ(path_t.yaws.rows(), 5u);
  EXPECT_EQ(path_t.x(2), 5);
  EXPECT_EQ(path_t.y(2), 50);
  EXPECT_NEAR(path_t.yaws(2), 0.0, 1e-6);
}

TEST(UtilsTests, AnglesTests)
{
  // Test angle normalization by creating insane angles
  Eigen::ArrayXf angles(100);
  angles.setConstant(1.0f);

  for (unsigned int i = 0; i != angles.size(); i++) {
    angles(i) = i * i;
    if (i % 2 == 0) {
      angles(i) *= -1;
    }
  }

  auto norm_ang = normalize_angles(angles).eval();
  for (unsigned int i = 0; i != norm_ang.size(); i++) {
    EXPECT_TRUE((norm_ang(i) >= -M_PIF) && (norm_ang(i) <= M_PIF));
  }

  // Test shortest angular distance
  Eigen::ArrayXf zero_angles(100);
  zero_angles.setZero();
  auto ang_dist = shortest_angular_distance(angles, zero_angles).eval();
  for (unsigned int i = 0; i != ang_dist.size(); i++) {
    EXPECT_TRUE((ang_dist(i) >= -M_PIF) && (ang_dist(i) <= M_PIF));
  }

  // Test point-pose angle
  geometry_msgs::msg::Pose pose;
  pose.position.x = 0.0;
  pose.position.y = 0.0;
  pose.orientation.w = 1.0;
  double point_x = 1.0, point_y = 0.0;
  bool forward_preference = true;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, forward_preference), 0.0, 1e-6);
  forward_preference = false;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, forward_preference), 0.0, 1e-6);
  point_x = -1.0;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, forward_preference), 0.0, 1e-6);
  forward_preference = true;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, forward_preference), M_PI, 1e-6);

  // Test point-pose angle with goal yaws
  point_x = 1.0;
  double point_yaw = 0.0;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, point_yaw), 0.0, 1e-6);
  point_yaw = M_PI;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, point_yaw), M_PI, 1e-6);
  point_yaw = 0.1;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, point_yaw), 0.0, 1e-3);
  point_yaw = 3.04159;
  EXPECT_NEAR(posePointAngle(pose, point_x, point_y, point_yaw), M_PI, 1e-3);
}

// True if both values have the same bits, or both are NaN
template<typename T>
bool sameValue(const T a, const T b)
{
  if (std::isnan(a) || std::isnan(b)) {
    return std::isnan(a) && std::isnan(b);
  }
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// Angles around every multiple of pi up to 6 pi, a sweep up to ~150 rad and special values
template<typename T>
std::vector<T> angleTestValues()
{
  std::vector<T> values;
  for (int k = -6; k <= 6; k++) {
    const T base = static_cast<T>(k * M_PI);
    T below = base, above = base;
    for (int n = 0; n < 4; n++) {
      values.push_back(below);
      values.push_back(above);
      below = std::nextafter(below, -std::numeric_limits<T>::infinity());
      above = std::nextafter(above, std::numeric_limits<T>::infinity());
    }
  }
  for (int k = -400; k <= 400; k++) {
    values.push_back(static_cast<T>(k * 0.37));
  }
  values.push_back(static_cast<T>(0.0));
  values.push_back(static_cast<T>(-0.0));
  values.push_back(std::numeric_limits<T>::infinity());
  values.push_back(-std::numeric_limits<T>::infinity());
  values.push_back(std::numeric_limits<T>::quiet_NaN());
  return values;
}

TEST(UtilsTests, AnglesMatchFmodReference)
{
  // fmodTwoPi gives the same bits as std::fmod, also at the multiples of its divisor
  auto values_d = angleTestValues<double>();
  auto values_f = angleTestValues<float>();
  for (int k = -3; k <= 3; k++) {
    values_d.push_back(k * (2.0 * M_PI));
    values_f.push_back(static_cast<float>(k) * (2.0f * M_PIF));
  }
  for (const double x : values_d) {
    EXPECT_TRUE(sameValue(fmodTwoPi(x, 2.0 * M_PI), std::fmod(x, 2.0 * M_PI))) << x;
  }
  for (const float x : values_f) {
    EXPECT_TRUE(sameValue(fmodTwoPi(x, 2.0f * M_PIF), std::fmod(x, 2.0f * M_PIF))) << x;
  }

  // normalizeAngle and shortestAngularDistance give the same bits as the angles package
  for (const double x : values_d) {
    EXPECT_TRUE(sameValue(normalizeAngle(x), angles::normalize_angle(x))) << x;
  }
  for (size_t i = 0; i < values_f.size(); i++) {
    const float from = values_f[i];
    const float to = values_f[(i * 7 + 3) % values_f.size()];
    EXPECT_TRUE(
      sameValue(
        shortestAngularDistance(from, to),
        angles::shortest_angular_distance(from, to))) << from << " " << to;
  }

  // normalize_angles gives the same bits as std::fmod
  Eigen::ArrayXf angles = Eigen::Map<const Eigen::ArrayXf>(values_f.data(), values_f.size());
  const Eigen::ArrayXf normalized = normalize_angles(angles).eval();
  for (Eigen::Index i = 0; i < angles.size(); i++) {
    const float remainder = std::fmod(angles(i) + M_PIF, 2.0f * M_PIF);
    const float expected = remainder < 0.0f ? remainder + M_PIF : remainder - M_PIF;
    EXPECT_TRUE(sameValue(normalized(i), expected)) << angles(i);
  }
}

TEST(UtilsTests, FurthestAndClosestReachedPoint)
{
  models::State state;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(100, 2);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(10);
  Eigen::ArrayXf costs;
  float model_dt = 0.1;

  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references

  // Attempt to set furthest point if notionally set, should not change
  data.furthest_reached_path_point = 99999;
  setPathFurthestPointIfNotSet(data);
  EXPECT_EQ(data.furthest_reached_path_point, 99999);

  // Attempt to set if not set already with no other information, should fail
  CriticData data2 =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references
  setPathFurthestPointIfNotSet(data2);
  EXPECT_EQ(data2.furthest_reached_path_point, 0);

  // Test the actual computation of the path point reached
  generated_trajectories.x = Eigen::ArrayXXf::Zero(100, 2);
  generated_trajectories.x.col(1).setOnes();
  generated_trajectories.y = Eigen::ArrayXXf::Zero(100, 2);
  generated_trajectories.yaws = Eigen::ArrayXXf::Zero(100, 2);

  nav_msgs::msg::Path plan;
  plan.poses.resize(10);
  for (unsigned int i = 0; i != plan.poses.size(); i++) {
    plan.poses[i].pose.position.x = 0.2 * i;
    plan.poses[i].pose.position.y = 0.0;
  }
  path = toTensor(plan);

  CriticData data3 =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references
  EXPECT_EQ(findPathFurthestReachedPoint(data3), 5);
}

// Verify that the returned index of findPathFurthestReachedPoint() doesn't
// exceed what is physically reachable (longest trajectory's arc-length)
TEST(UtilsTests, FurthestReachedPointUturn)
{
  models::State state;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  Eigen::ArrayXf costs;
  float model_dt = 0.1;

  // U-turn path: (0, 0) to (8, 0), turn (8, 1), and go back to (0, 1)
  const float path_x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  const float path_y[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  nav_msgs::msg::Path plan;
  plan.poses.resize(18);
  for (unsigned int i = 0; i != 18; i++) {
    plan.poses[i].pose.position.x = path_x[i];
    plan.poses[i].pose.position.y = path_y[i];
  }
  path = toTensor(plan);

  // 10 identical trajectories from (0,0) to (5,1), arc-length ~5.1m
  generated_trajectories.x = Eigen::ArrayXXf::Zero(10, 4);
  generated_trajectories.x.col(1).setConstant(5.0f / 3.0f);
  generated_trajectories.x.col(2).setConstant(10.0f / 3.0f);
  generated_trajectories.x.col(3).setConstant(5.0f);
  generated_trajectories.y = Eigen::ArrayXXf::Zero(10, 4);
  generated_trajectories.y.col(1).setConstant(1.0f / 3.0f);
  generated_trajectories.y.col(2).setConstant(2.0f / 3.0f);
  generated_trajectories.y.col(3).setConstant(1.0f);
  generated_trajectories.yaws = Eigen::ArrayXXf::Zero(10, 4);

  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references

  size_t result = findPathFurthestReachedPoint(data);
  EXPECT_LE(result, 6u);
}

// findPathFurthestReachedPoint without skipping trajectories that cannot raise the result
size_t referenceFurthestReachedPoint(const CriticData & data)
{
  const int traj_cols = data.trajectories.x.cols();
  const int n_rows = static_cast<int>(data.trajectories.x.rows());
  const int n_cols = static_cast<int>(data.path.x.size());
  std::vector<float> path_integrated_dists(n_cols, 0.0f);
  for (int i = 1; i < n_cols; ++i) {
    const float dx = data.path.x(i) - data.path.x(i - 1);
    const float dy = data.path.y(i) - data.path.y(i - 1);
    path_integrated_dists[i] = path_integrated_dists[i - 1] + sqrtf(dx * dx + dy * dy);
  }
  const Eigen::ArrayXf traj_integrated_dists =
    ((data.trajectories.x.rightCols(traj_cols - 1) -
    data.trajectories.x.leftCols(traj_cols - 1)).square() +
    (data.trajectories.y.rightCols(traj_cols - 1) -
    data.trajectories.y.leftCols(traj_cols - 1)).square())
    .sqrt().rowwise().sum();
  const auto & traj_x_end = data.trajectories.x.col(traj_cols - 1);
  const auto & traj_y_end = data.trajectories.y.col(traj_cols - 1);
  int max_idx = 0;
  for (int i = 0; i < n_rows; ++i) {
    int max_reachable_idx = static_cast<int>(
      std::lower_bound(
        path_integrated_dists.begin(), path_integrated_dists.end(),
        traj_integrated_dists(i)) - path_integrated_dists.begin());
    max_reachable_idx = std::min(max_reachable_idx, n_cols - 1);
    Eigen::Index eucl_idx;
    ((data.path.x.head(max_reachable_idx + 1) - traj_x_end(i)).square() +
    (data.path.y.head(max_reachable_idx + 1) - traj_y_end(i)).square()).minCoeff(&eucl_idx);
    max_idx = std::max(max_idx, static_cast<int>(eucl_idx));
    if (max_idx == n_cols - 1) {
      break;
    }
  }
  return static_cast<size_t>(max_idx);
}

TEST(UtilsTests, FurthestReachedPointMatchesReference)
{
  models::State state;
  models::Trajectories trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  Eigen::ArrayXf costs;
  float model_dt = 0.05f;
  CriticData data =
  {state, trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr, std::nullopt, std::nullopt};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> speed(-0.3f, 1.4f);
  std::uniform_real_distribution<float> turn(-0.05f, 0.05f);
  for (int run = 0; run < 20; run++) {
    // Curved path with 5 cm spacing
    path.reset(150 + run * 10);
    float yaw = 0.0f;
    for (Eigen::Index i = 1; i < path.x.size(); i++) {
      yaw += 0.02f * std::sin(0.3f * run + 0.05f * i);
      path.x(i) = path.x(i - 1) + 0.05f * std::cos(yaw);
      path.y(i) = path.y(i - 1) + 0.05f * std::sin(yaw);
      path.yaws(i) = yaw;
    }
    // Random trajectories, rows not a multiple of the packet size
    trajectories.reset(203, 60);
    for (Eigen::Index t = 0; t < trajectories.x.rows(); t++) {
      float traj_yaw = 0.0f;
      const float v = speed(gen);
      for (Eigen::Index j = 1; j < trajectories.x.cols(); j++) {
        traj_yaw += turn(gen);
        trajectories.x(t, j) = trajectories.x(t, j - 1) + 0.1f * v * std::cos(traj_yaw);
        trajectories.y(t, j) = trajectories.y(t, j - 1) + 0.1f * v * std::sin(traj_yaw);
      }
    }
    EXPECT_EQ(findPathFurthestReachedPoint(data), referenceFurthestReachedPoint(data));
  }
}

TEST(UtilsTests, findPathCosts)
{
  models::State state;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  path.reset(50);
  Eigen::ArrayXf costs;
  float model_dt = 0.1;

  CriticData data =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references

  // Test not set if already set, should not change
  data.path_pts_valid = std::vector<bool>(10, false);
  for (unsigned int i = 0; i != 10; i++) {
    (*data.path_pts_valid)[i] = false;
  }
  EXPECT_TRUE(data.path_pts_valid);
  setPathCostsIfNotSet(data, nullptr);
  EXPECT_EQ(data.path_pts_valid->size(), 10u);

  CriticData data3 =
  {state, generated_trajectories, path, goal, costs, std::nullopt, model_dt, false, nullptr,
    nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references

  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", true);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  auto * costmap = costmap_ros->getCostmap();
  // island in the middle of lethal cost to cross. Costmap defaults to size 5x5 @ 10cm resolution
  for (unsigned int i = 10; i <= 30; ++i) {  // 1m-3m
    for (unsigned int j = 10; j <= 30; ++j) {  // 1m-3m
      costmap->setCost(i, j, 254);
    }
  }
  for (unsigned int i = 40; i <= 45; ++i) {  // 4m-4.5m
    for (unsigned int j = 45; j <= 45; ++j) {  // 4m-4.5m
      costmap->setCost(i, j, 253);
    }
  }

  path.x(1) = 999999999;  // OFF COSTMAP
  path.y(1) = 999999999;
  path.x(10) = 1.5;  // IN LETHAL
  path.y(10) = 1.5;
  path.x(20) = 4.2;  // IN INFLATED
  path.y(20) = 4.2;

  // This should be evaluated and have real outputs now
  setPathCostsIfNotSet(data3, costmap_ros);
  EXPECT_TRUE(data3.path_pts_valid.has_value());
  for (unsigned int i = 0; i != path.x.size() - 1; i++) {
    if (i == 1 || i == 10) {
      EXPECT_FALSE((*data3.path_pts_valid)[i]);
    } else {
      EXPECT_TRUE((*data3.path_pts_valid)[i]);
    }
  }
}

TEST(UtilsTests, SmootherMatchesWindowDefinition)
{
  // Every element is the 9 point Savitzky-Golay filter over the unfiltered sequence, which is
  // extended by the control history before its start and by its last element after its end
  Eigen::Array<float, 9, 1> filter;
  filter << -21.0f, 14.0f, 39.0f, 54.0f, 59.0f, 54.0f, 39.0f, 14.0f, -21.0f;
  filter /= 231.0f;
  std::mt19937 engine(3);
  std::uniform_real_distribution<float> value(-1.0f, 1.0f);

  for (int size : {21, 22, 30, 57, 100}) {
    for (bool shift : {false, true}) {
      models::ControlSequence sequence;
      sequence.vx = Eigen::ArrayXf::NullaryExpr(size, [&]() {return value(engine);});
      sequence.vy = Eigen::ArrayXf::NullaryExpr(size, [&]() {return value(engine);});
      sequence.wz = Eigen::ArrayXf::NullaryExpr(size, [&]() {return value(engine);});
      std::array<mppi::models::Control, 4> history;
      for (auto & control : history) {
        control = {value(engine), value(engine), value(engine)};
      }
      const models::ControlSequence unfiltered = sequence;
      const auto unfiltered_history = history;
      models::OptimizerSettings settings;
      settings.shift_control_sequence = shift;

      savitskyGolayFilter(sequence, history, settings);

      auto expect_filtered = [&](
        const Eigen::ArrayXf & filtered, const Eigen::ArrayXf & initial,
        const std::array<float, 4> & initial_history) {
          const int last = size - 1;
          for (int i = 0; i < last; i++) {
            Eigen::Array<float, 9, 1> window;
            for (int k = 0; k < 9; k++) {
              const int index = i + k - 4;
              window(k) = index < 0 ? initial_history[index + 4] : initial(std::min(index, last));
            }
            // up to the rounding of the sum of the 9 products
            EXPECT_NEAR(filtered(i), (window * filter).sum(), 1e-6) <<
              "size " << size << " at " << i;
          }
          EXPECT_EQ(filtered(last), initial(last));
        };
      expect_filtered(
        sequence.vx, unfiltered.vx,
        {unfiltered_history[0].vx, unfiltered_history[1].vx, unfiltered_history[2].vx,
          unfiltered_history[3].vx});
      expect_filtered(
        sequence.vy, unfiltered.vy,
        {unfiltered_history[0].vy, unfiltered_history[1].vy, unfiltered_history[2].vy,
          unfiltered_history[3].vy});
      expect_filtered(
        sequence.wz, unfiltered.wz,
        {unfiltered_history[0].wz, unfiltered_history[1].wz, unfiltered_history[2].wz,
          unfiltered_history[3].wz});
    }
  }
}

TEST(UtilsTests, SmootherTest)
{
  models::ControlSequence noisey_sequence, sequence_init;
  noisey_sequence.vx = 0.2 * Eigen::ArrayXf::Ones(30);
  noisey_sequence.vy = 0.0 * Eigen::ArrayXf::Ones(30);
  noisey_sequence.wz = 0.3 * Eigen::ArrayXf::Ones(30);

  // Make the sequence noisy
  std::mt19937 engine;
  std::normal_distribution<float> normal_dist = std::normal_distribution(0.0f, 0.2f);
  auto noises = Eigen::ArrayXf::NullaryExpr(
    30, [&]() {return normal_dist(engine);});
  noisey_sequence.vx += noises;
  noisey_sequence.vy += noises;
  noisey_sequence.wz += noises;
  sequence_init = noisey_sequence;

  std::array<mppi::models::Control, 4> history, history_init;
  history[3].vx = 0.1;
  history[3].vy = 0.0;
  history[3].wz = 0.3;
  history[2].vx = 0.1;
  history[2].vy = 0.0;
  history[2].wz = 0.3;
  history[1].vx = 0.1;
  history[1].vy = 0.0;
  history[1].wz = 0.3;
  history[0].vx = 0.0;
  history[0].vy = 0.0;
  history[0].wz = 0.0;
  history_init = history;

  models::OptimizerSettings settings;
  settings.shift_control_sequence = false;  // so result stores 0th value in history

  savitskyGolayFilter(noisey_sequence, history, settings);

  // Check history is propagated backward
  EXPECT_NEAR(history_init[3].vx, history[2].vx, 0.02);
  EXPECT_NEAR(history_init[3].vy, history[2].vy, 0.02);
  EXPECT_NEAR(history_init[3].wz, history[2].wz, 0.02);

  // Check history element is updated for first command
  EXPECT_NEAR(history[3].vx, 0.2, 0.05);
  EXPECT_NEAR(history[3].vy, 0.0, 0.035);
  EXPECT_NEAR(history[3].wz, 0.23, 0.02);

  // Check that path is smoother
  float smoothed_val{0}, original_val{0};
  for (unsigned int i = 1; i != noisey_sequence.vx.size() - 1; i++) {
    smoothed_val += fabs(noisey_sequence.vx(i) - 0.2);
    smoothed_val += fabs(noisey_sequence.vy(i) - 0.0);
    smoothed_val += fabs(noisey_sequence.wz(i) - 0.3);

    original_val += fabs(sequence_init.vx(i) - 0.2);
    original_val += fabs(sequence_init.vy(i) - 0.0);
    original_val += fabs(sequence_init.wz(i) - 0.3);
  }

  EXPECT_LT(smoothed_val, original_val);
}

TEST(UtilsTests, ShiftColumnsByOnePlaceTest)
{
  // Try with scalar value
  Eigen::ArrayXf scalar_val(1);
  scalar_val(0) = 5;
  utils::shiftColumnsByOnePlace(scalar_val, 1);
  EXPECT_EQ(scalar_val.size(), 1);
  EXPECT_EQ(scalar_val(0), 5);

  // Try with one dimensional array, shift right
  Eigen::ArrayXf array_1d(4);
  array_1d << 1, 2, 3, 4;
  utils::shiftColumnsByOnePlace(array_1d, 1);
  EXPECT_EQ(array_1d.size(), 4);
  EXPECT_EQ(array_1d(0), 1);
  EXPECT_EQ(array_1d(1), 1);
  EXPECT_EQ(array_1d(2), 2);
  EXPECT_EQ(array_1d(3), 3);

  // Try with one dimensional array, shift left
  array_1d(1) = 5;
  utils::shiftColumnsByOnePlace(array_1d, -1);
  EXPECT_EQ(array_1d.size(), 4);
  EXPECT_EQ(array_1d(0), 5);
  EXPECT_EQ(array_1d(1), 2);
  EXPECT_EQ(array_1d(2), 3);
  EXPECT_EQ(array_1d(2), 3);

  // Try with two dimensional array, shift right
  // 1 2 3 4        1 1 2 3
  // 5 6 7 8    ->  5 5 6 7
  // 9 10 11 12     9 9 10 11
  Eigen::ArrayXXf array_2d(3, 4);
  array_2d << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  utils::shiftColumnsByOnePlace(array_2d, 1);
  EXPECT_EQ(array_2d.rows(), 3);
  EXPECT_EQ(array_2d.cols(), 4);
  EXPECT_EQ(array_2d(0, 0), 1);
  EXPECT_EQ(array_2d(1, 0), 5);
  EXPECT_EQ(array_2d(2, 0), 9);
  EXPECT_EQ(array_2d(0, 1), 1);
  EXPECT_EQ(array_2d(1, 1), 5);
  EXPECT_EQ(array_2d(2, 1), 9);
  EXPECT_EQ(array_2d(0, 2), 2);
  EXPECT_EQ(array_2d(1, 2), 6);
  EXPECT_EQ(array_2d(2, 2), 10);
  EXPECT_EQ(array_2d(0, 3), 3);
  EXPECT_EQ(array_2d(1, 3), 7);
  EXPECT_EQ(array_2d(2, 3), 11);

  array_2d.col(0).setZero();

  // Try with two dimensional array, shift left
  // 0 1 2 3      1 2 3 3
  // 0 5 6 7   -> 5 6 7 7
  // 0 9 10 11    9 10 11 11
  utils::shiftColumnsByOnePlace(array_2d, -1);
  EXPECT_EQ(array_2d.rows(), 3);
  EXPECT_EQ(array_2d.cols(), 4);
  EXPECT_EQ(array_2d(0, 0), 1);
  EXPECT_EQ(array_2d(1, 0), 5);
  EXPECT_EQ(array_2d(2, 0), 9);
  EXPECT_EQ(array_2d(0, 1), 2);
  EXPECT_EQ(array_2d(1, 1), 6);
  EXPECT_EQ(array_2d(2, 1), 10);
  EXPECT_EQ(array_2d(0, 2), 3);
  EXPECT_EQ(array_2d(1, 2), 7);
  EXPECT_EQ(array_2d(2, 2), 11);
  EXPECT_EQ(array_2d(0, 3), 3);
  EXPECT_EQ(array_2d(1, 3), 7);
  EXPECT_EQ(array_2d(2, 3), 11);

  // Try with invalid direction value.
  EXPECT_THROW(utils::shiftColumnsByOnePlace(array_2d, -2), std::logic_error);
}

TEST(UtilsTests, NormalizeYawsBetweenPointsTest)
{
  Eigen::ArrayXf last_yaws(10);
  last_yaws.setZero();

  Eigen::ArrayXf yaw_between_points(10);
  yaw_between_points.setZero(10);

  // Try with both angles 0
  Eigen::ArrayXf yaws_between_points_corrected = utils::normalize_yaws_between_points(
    last_yaws,
    yaw_between_points);
  EXPECT_TRUE(yaws_between_points_corrected.isApprox(yaw_between_points));

  // Try with yaw between points as pi/4
  yaw_between_points.setConstant(M_PIF_2 / 2);
  yaws_between_points_corrected = utils::normalize_yaws_between_points(
    last_yaws,
    yaw_between_points);
  EXPECT_TRUE(yaws_between_points_corrected.isApprox(yaw_between_points));

  // Try with yaw between points as pi/2
  yaw_between_points.setConstant(M_PIF_2);
  yaws_between_points_corrected = utils::normalize_yaws_between_points(
    last_yaws,
    yaw_between_points);
  EXPECT_TRUE(yaws_between_points_corrected.isApprox(yaw_between_points));

  // Try with a few yaw between points  more than pi/2
  yaw_between_points[1] = 1.2 * M_PIF_2;
  yaws_between_points_corrected = utils::normalize_yaws_between_points(
    last_yaws,
    yaw_between_points);
  EXPECT_NEAR(yaws_between_points_corrected[1], -0.8 * M_PIF_2, 1e-3);
  EXPECT_NEAR(yaws_between_points_corrected[0], yaw_between_points[0], 1e-3);
  EXPECT_NEAR(yaws_between_points_corrected[9], yaw_between_points[9], 1e-3);

  // Try with goal angle 0
  float goal_angle = 0;
  yaws_between_points_corrected = utils::normalize_yaws_between_points(
    goal_angle,
    yaw_between_points);
  EXPECT_NEAR(yaws_between_points_corrected[1], -0.8 * M_PIF_2, 1e-3);
}

TEST(UtilsTests, toTrajectoryMsgTest)
{
  Eigen::ArrayXXf trajectory(5, 3);
  trajectory <<
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0;

  models::ControlSequence control_sequence;
  control_sequence.vx = Eigen::ArrayXf::Ones(5);
  control_sequence.wz = Eigen::ArrayXf::Ones(5);
  control_sequence.vy = Eigen::ArrayXf::Zero(5);

  std_msgs::msg::Header header;
  header.frame_id = "map";
  header.stamp = rclcpp::Time(100, 0, RCL_ROS_TIME);

  auto trajectory_msg = utils::toTrajectoryMsg(
    trajectory, control_sequence, 1.0, header);

  EXPECT_EQ(trajectory_msg->header.frame_id, "map");
  EXPECT_EQ(trajectory_msg->header.stamp, header.stamp);
  EXPECT_EQ(trajectory_msg->points.size(), 5u);
  EXPECT_EQ(trajectory_msg->points[0].pose.position.x, 0.0);
  EXPECT_EQ(trajectory_msg->points[0].pose.position.y, 0.0);
  EXPECT_EQ(trajectory_msg->points[1].pose.position.x, 1.0);
  EXPECT_EQ(trajectory_msg->points[1].pose.position.y, 1.0);
  EXPECT_EQ(trajectory_msg->points[2].pose.position.x, 2.0);
  EXPECT_EQ(trajectory_msg->points[2].pose.position.y, 2.0);
  EXPECT_EQ(trajectory_msg->points[3].pose.position.x, 3.0);
  EXPECT_EQ(trajectory_msg->points[3].pose.position.y, 3.0);
  EXPECT_EQ(trajectory_msg->points[4].pose.position.x, 4.0);
  EXPECT_EQ(trajectory_msg->points[4].pose.position.y, 4.0);

  EXPECT_EQ(trajectory_msg->points[0].velocity.linear.x, 1.0);
  EXPECT_EQ(trajectory_msg->points[0].velocity.linear.y, 0.0);
  EXPECT_EQ(trajectory_msg->points[0].velocity.angular.z, 1.0);
  EXPECT_EQ(trajectory_msg->points[1].velocity.linear.x, 1.0);
  EXPECT_EQ(trajectory_msg->points[1].velocity.linear.y, 0.0);
  EXPECT_EQ(trajectory_msg->points[1].velocity.angular.z, 1.0);
  EXPECT_EQ(trajectory_msg->points[2].velocity.linear.x, 1.0);
  EXPECT_EQ(trajectory_msg->points[2].velocity.linear.y, 0.0);
  EXPECT_EQ(trajectory_msg->points[2].velocity.angular.z, 1.0);
  EXPECT_EQ(trajectory_msg->points[3].velocity.linear.x, 1.0);
  EXPECT_EQ(trajectory_msg->points[3].velocity.linear.y, 0.0);
  EXPECT_EQ(trajectory_msg->points[3].velocity.angular.z, 1.0);
  EXPECT_EQ(trajectory_msg->points[4].velocity.linear.x, 1.0);
  EXPECT_EQ(trajectory_msg->points[4].velocity.linear.y, 0.0);
  EXPECT_EQ(trajectory_msg->points[4].velocity.angular.z, 1.0);

  EXPECT_EQ(trajectory_msg->points[0].time_from_start, rclcpp::Duration(0, 0));
  EXPECT_EQ(trajectory_msg->points[1].time_from_start, rclcpp::Duration(1, 0));
  EXPECT_EQ(trajectory_msg->points[2].time_from_start, rclcpp::Duration(2, 0));
  EXPECT_EQ(trajectory_msg->points[3].time_from_start, rclcpp::Duration(3, 0));
  EXPECT_EQ(trajectory_msg->points[4].time_from_start, rclcpp::Duration(4, 0));
}

TEST(UtilsTests, getLastPathPoseTest)
{
  nav_msgs::msg::Path path;
  path.poses.resize(10);
  path.poses[9].pose.position.x = 5.0;
  path.poses[9].pose.position.y = 50.0;
  path.poses[9].pose.orientation.x = 0.0;
  path.poses[9].pose.orientation.y = 0.0;
  path.poses[9].pose.orientation.z = 1.0;
  path.poses[9].pose.orientation.w = 0.0;

  models::Path path_t = toTensor(path);
  geometry_msgs::msg::Pose last_path_pose = utils::getLastPathPose(path_t);

  EXPECT_EQ(last_path_pose.position.x, 5);
  EXPECT_EQ(last_path_pose.position.y, 50);
  EXPECT_NEAR(last_path_pose.orientation.x, 0.0, 1e-3);
  EXPECT_NEAR(last_path_pose.orientation.y, 0.0, 1e-3);
  EXPECT_NEAR(last_path_pose.orientation.z, 1.0, 1e-3);
  EXPECT_NEAR(last_path_pose.orientation.w, 0.0, 1e-3);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  rclcpp::init(0, nullptr);

  int result = RUN_ALL_TESTS();

  rclcpp::shutdown();

  return result;
}
