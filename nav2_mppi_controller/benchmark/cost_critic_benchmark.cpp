// Copyright (c) 2026 enway GmbH
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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nav2_costmap_2d/cost_values.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_costmap_2d/footprint_collision_checker.hpp>
#include <nav2_costmap_2d/inflation_layer_interface.hpp>

#include "nav2_mppi_controller/critic_data.hpp"
#include "nav2_mppi_controller/critics/cost_critic.hpp"
#include "nav2_mppi_controller/models/path.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/trajectories.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"

// Standalone CostCritic benchmark using the robot's nav2_params.yaml values:
// 30x30 m local costmap at 0.1 m, padded footprint, inflation radius 5.0 / scaling 0.7,
// batch_size 2000, time_steps 100, model_dt 0.05, trajectory_point_step 2.
// Scenes:
//   0 corridor: walls 3 m to each side and a box narrowing the left side, so trajectories
//     mix inflated cost, footprint checks and collisions.
//   1 wall: the robot drives close along a single wall, so nearly every pose needs a
//     footprint check (matches field measurements of ~90% footprint checks per lookup).

namespace
{

constexpr int kBatchSize = 2000;
constexpr int kTimeSteps = 100;
constexpr float kModelDt = 0.05f;
constexpr double kResolution = 0.1;
constexpr double kRobotX = 12.0;
constexpr double kRobotY = 15.0;
constexpr double kWallDistance = 1.7;
constexpr float kWallWzBiasStd = 0.03f;

std::shared_ptr<nav2_costmap_2d::Costmap2DROS> makeCostmapRos()
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    {"width", 30},
    {"height", 30},
    {"resolution", kResolution},
    {"footprint", "[ [2.230, 0.65], [-1.4312, 0.65], [-1.4312, -0.65], [2.230, -0.65] ]"},
    {"footprint_padding", 0.1},
    {"plugins", std::vector<std::string>{"inflation_layer"}},
    {"inflation_layer.plugin", "nav2_costmap_2d::InflationLayer"},
    {"inflation_layer.inflation_radius", 5.0},
    {"inflation_layer.cost_scaling_factor", 0.7},
  });
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(options);
  costmap_ros->on_configure(rclcpp_lifecycle::State{});

  // Plugins are added after the initial resize, so resize again to let the
  // inflation layer pick up the resolution.
  auto * costmap = costmap_ros->getCostmap();
  costmap_ros->getLayeredCostmap()->resizeMap(
    costmap->getSizeInCellsX(), costmap->getSizeInCellsY(), kResolution, 0.0, 0.0);
  return costmap_ros;
}

void fillScene(std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, int scene)
{
  auto * costmap = costmap_ros->getCostmap();
  auto inflation = nav2_costmap_2d::InflationLayerInterface::getInflationLayer(
    costmap_ros, "inflation_layer");

  std::vector<std::pair<int, int>> obstacles;
  auto addRect = [&](double x0, double y0, double x1, double y1) {
      for (double x = x0; x <= x1; x += kResolution) {
        for (double y = y0; y <= y1; y += kResolution) {
          unsigned int mx, my;
          if (costmap->worldToMap(x, y, mx, my)) {
            obstacles.emplace_back(mx, my);
          }
        }
      }
    };
  if (scene == 0) {
    // Corridor walls 3 m left and right of the robot centre line
    addRect(8.0, kRobotY + 3.0, 28.0, kRobotY + 3.0);
    addRect(8.0, kRobotY - 3.0, 28.0, kRobotY - 3.0);
    // Box narrowing the left side of the corridor; straight trajectories just pass it
    addRect(19.0, kRobotY + 0.9, 19.6, kRobotY + 3.0);
  } else {
    // Single wall left of the robot centre line
    addRect(8.0, kRobotY + kWallDistance, 28.0, kRobotY + kWallDistance);
  }

  const int radius_cells = static_cast<int>(inflation->getInflationRadius() / kResolution);
  const int size_x = static_cast<int>(costmap->getSizeInCellsX());
  const int size_y = static_cast<int>(costmap->getSizeInCellsY());
  for (const auto & [ox, oy] : obstacles) {
    for (int x = std::max(0, ox - radius_cells); x < std::min(size_x, ox + radius_cells + 1);
      ++x)
    {
      for (int y = std::max(0, oy - radius_cells); y < std::min(size_y, oy + radius_cells + 1);
        ++y)
      {
        const double dist = std::hypot(x - ox, y - oy);
        if (dist > radius_cells) {
          continue;
        }
        const unsigned char cost = inflation->computeCost(dist);
        if (cost > costmap->getCost(x, y)) {
          costmap->setCost(x, y, cost);
        }
      }
    }
  }
}

// MPPI-like rollouts: nominal forward motion plus per-step Gaussian noise, with a
// per-trajectory curvature bias so the batch fans out across the corridor.
void fillTrajectories(mppi::models::Trajectories & trajectories, float wz_bias_std)
{
  trajectories.reset(kBatchSize, kTimeSteps);
  std::mt19937 rng(42);
  std::normal_distribution<float> vx_noise(0.0f, 0.2f);
  std::normal_distribution<float> wz_noise(0.0f, 0.6f);
  std::normal_distribution<float> wz_bias(0.0f, wz_bias_std);
  constexpr float kVxMax = 1.67f;
  constexpr float kWzMax = 0.8534f;

  for (int i = 0; i < kBatchSize; ++i) {
    float x = static_cast<float>(kRobotX);
    float y = static_cast<float>(kRobotY);
    float yaw = 0.0f;
    const float bias = wz_bias(rng);
    for (int j = 0; j < kTimeSteps; ++j) {
      const float vx = std::clamp(1.2f + vx_noise(rng), 0.0f, kVxMax);
      const float wz = std::clamp(bias + wz_noise(rng), -kWzMax, kWzMax);
      yaw += wz * kModelDt;
      x += vx * std::cos(yaw) * kModelDt;
      y += vx * std::sin(yaw) * kModelDt;
      trajectories.x(i, j) = x;
      trajectories.y(i, j) = y;
      trajectories.yaws(i, j) = yaw;
    }
  }
}

struct WorkCounts
{
  double centre_lookups{0};
  double non_free_poses{0};
  double footprint_checks{0};
};

// Replays CostCritic::score() single-threaded and counts the work per call:
// centre-cost lookups, poses with non-zero cost, and full footprint checks.
WorkCounts countWork(
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  const mppi::models::Trajectories & trajectories, int point_step)
{
  auto * costmap = costmap_ros->getCostmap();
  auto inflation = nav2_costmap_2d::InflationLayerInterface::getInflationLayer(
    costmap_ros, "inflation_layer");
  const float possible_collision_cost = static_cast<float>(inflation->computeCost(
      costmap_ros->getLayeredCostmap()->getCircumscribedRadius() / kResolution));
  const auto footprint = costmap_ros->getRobotFootprint();
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap);

  WorkCounts counts;
  for (int i = 0; i < kBatchSize; ++i) {
    for (int j = (kTimeSteps - 1) / point_step * point_step; j >= 0; j -= point_step) {
      counts.centre_lookups++;
      unsigned int mx, my;
      float cost = 255.0f;
      if (costmap->worldToMap(trajectories.x(i, j), trajectories.y(i, j), mx, my)) {
        cost = costmap->getCost(mx, my);
        if (cost < 1.0f) {
          continue;
        }
      }
      counts.non_free_poses++;
      if (cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
        cost == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
      {
        break;
      }
      float score_cost = cost;
      if (cost >= possible_collision_cost || possible_collision_cost < 1.0f) {
        counts.footprint_checks++;
        score_cost = static_cast<float>(checker.footprintCostAtPose(
            trajectories.x(i, j), trajectories.y(i, j), trajectories.yaws(i, j), footprint));
      }
      if (score_cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
        score_cost == nav2_costmap_2d::NO_INFORMATION)
      {
        break;
      }
    }
  }
  return counts;
}

double percentile(std::vector<double> samples, double p)
{
  const size_t k = static_cast<size_t>(p * static_cast<double>(samples.size() - 1));
  std::nth_element(samples.begin(), samples.begin() + k, samples.end());
  return samples[k];
}

}  // namespace

static void BM_CostCritic(benchmark::State & state)
{
  const int num_threads = static_cast<int>(state.range(0));
  const int scene = static_cast<int>(state.range(1));

  auto costmap_ros = makeCostmapRos();
  fillScene(costmap_ros, scene);

  rclcpp::NodeOptions options;
  options.parameter_overrides({
    {"CostCritic.consider_footprint", true},
    {"CostCritic.cost_power", 1},
    {"CostCritic.cost_weight", 3.81},
    {"CostCritic.collision_cost", 1000000.0},
    {"CostCritic.critical_cost", 300.0},
    {"CostCritic.near_collision_cost", 253},
    {"CostCritic.near_goal_distance", 1.0},
    {"CostCritic.trajectory_point_step", 2},
    {"CostCritic.inflation_layer_name", "inflation_layer"},
    {"CostCritic.num_threads", num_threads},
  });
  auto node = std::make_shared<nav2::LifecycleNode>("cost_critic_benchmark", options);
  std::string name = "mppi";
  mppi::ParametersHandler param_handler(node, name);

  mppi::critics::CostCritic critic;
  critic.on_configure(node, "mppi", "CostCritic", costmap_ros, &param_handler);

  mppi::models::State robot_state;
  robot_state.local_path_length = 10.0f;
  mppi::models::Trajectories trajectories;
  // Following a wall, the sampled trajectories fan out less than in open space
  fillTrajectories(trajectories, scene == 0 ? 0.08f : kWallWzBiasStd);
  mppi::models::Path path;
  geometry_msgs::msg::Pose goal;
  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(kBatchSize);
  float model_dt = kModelDt;
  mppi::CriticData data =
  {robot_state, trajectories, path, goal, costs, std::nullopt, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};

  std::vector<double> samples_us;
  for (auto _ : state) {
    costs.setZero();
    const auto start = std::chrono::steady_clock::now();
    critic.score(data);
    const auto end = std::chrono::steady_clock::now();
    samples_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    benchmark::DoNotOptimize(costs.data());
  }

  state.counters["p50_us"] = percentile(samples_us, 0.50);
  state.counters["p95_us"] = percentile(samples_us, 0.95);
  state.counters["max_us"] = *std::max_element(samples_us.begin(), samples_us.end());
  // A collision adds collision_cost * weight / strided_cols = 300, a free trajectory at most 4.5
  state.counters["colliding"] = static_cast<double>((costs > 100.0f).count());
  const auto work = countWork(costmap_ros, trajectories, 2);
  state.counters["lookups"] = work.centre_lookups;
  state.counters["non_free"] = work.non_free_poses;
  state.counters["fp_checks"] = work.footprint_checks;
  state.counters["fp_ratio"] = work.footprint_checks / work.centre_lookups;
}

BENCHMARK(BM_CostCritic)->ArgNames({"threads", "scene"})
->ArgsProduct({{1, 2, 4, 8}, {0, 1}})
->Unit(benchmark::kMicrosecond)->UseRealTime()->MinTime(2.0);

int main(int argc, char ** argv)
{
  // rclcpp is shut down explicitly here: from a static destructor it aborts inside rmw_zenoh
  rclcpp::init(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  rclcpp::shutdown();
  return 0;
}
