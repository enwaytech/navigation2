// Copyright (c) 2026 Enway GmbH
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
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "nav2_util/line_iterator.hpp"
#include "nav2_mppi_controller/critics/cost_critic.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"

// Compares the optimized CostCritic collision checking with the reference implementation:
// nav2_costmap_2d::FootprintCollisionChecker and CostCritic::inCollision()

using namespace mppi;  // NOLINT
using namespace mppi::critics;  // NOLINT

class CostCriticWrapper : public CostCritic
{
public:
  using CostCritic::updateCostmapInfo;
  using CostCritic::updateBlockedCellDistance;
  using CostCritic::footprintInCollision;
  using CostCritic::transformFootprintVertex;
  using CostCritic::blocked_cell_distance_;
  using CostCritic::possible_collision_cost_;
  using CostCritic::is_tracking_unknown_;
  using CostCritic::updateFootprintPrefilter;
  using CostCritic::footprintCollisionFree;
  using CostCritic::prefilter_active_;
  using CostCritic::prefilter_min_distance_;

  // Footprint branch of inCollision(): a center cost that is neither lethal nor inscribed and
  // not below possible_collision_cost_ makes it run FootprintCollisionChecker
  bool referenceFootprintInCollision(
    float x, float y, float theta, const nav2_costmap_2d::Footprint & footprint)
  {
    return inCollision(
      static_cast<float>(nav2_costmap_2d::NO_INFORMATION), x, y, theta, footprint);
  }

  // The scoring loop of CostCritic::score() before its optimization, single-threaded
  void referenceScore(CriticData & data)
  {
    updateCostmapInfo();
    auto * costmap = collision_checker_.getCostmap();
    nav2_costmap_2d::Footprint footprint;
    if (consider_footprint_) {
      footprint = costmap_ros_->getRobotFootprint();
    }
    const bool near_goal = data.state.local_path_length < near_goal_distance_;

    Eigen::ArrayXf repulsive_cost(data.costs.rows());
    repulsive_cost.setZero();
    bool all_trajectories_collide = true;

    int strided_traj_cols = floor((data.trajectories.x.cols() - 1) / trajectory_point_step_) + 1;
    int strided_traj_rows = data.trajectories.x.rows();
    int outer_stride = strided_traj_rows * trajectory_point_step_;
    const auto traj_x = Eigen::Map<const Eigen::ArrayXXf, 0, Eigen::Stride<-1, -1>>(
      data.trajectories.x.data(), strided_traj_rows, strided_traj_cols,
      Eigen::Stride<-1, -1>(outer_stride, 1));
    const auto traj_y = Eigen::Map<const Eigen::ArrayXXf, 0, Eigen::Stride<-1, -1>>(
      data.trajectories.y.data(), strided_traj_rows, strided_traj_cols,
      Eigen::Stride<-1, -1>(outer_stride, 1));
    const auto traj_yaw = Eigen::Map<const Eigen::ArrayXXf, 0, Eigen::Stride<-1, -1>>(
      data.trajectories.yaws.data(), strided_traj_rows, strided_traj_cols,
      Eigen::Stride<-1, -1>(outer_stride, 1));

    for (int i = 0; i < strided_traj_rows; ++i) {
      bool trajectory_collide = false;
      float pose_cost = 0.0f;
      float & traj_cost = repulsive_cost(i);
      for (int j = strided_traj_cols - 1; j >= 0; j--) {
        float Tx = traj_x(i, j);
        float Ty = traj_y(i, j);
        unsigned int x_i = 0u, y_i = 0u;
        if (!worldToMapFloat(Tx, Ty, x_i, y_i)) {
          pose_cost = 255.0f;
        } else {
          pose_cost = static_cast<float>(costmap->getCost(getIndex(x_i, y_i)));
          if (pose_cost < 1.0f) {
            continue;
          }
        }
        if (inCollision(pose_cost, Tx, Ty, traj_yaw(i, j), footprint)) {
          traj_cost = collision_cost_;
          trajectory_collide = true;
          break;
        }
        if (pose_cost >= static_cast<float>(near_collision_cost_)) {
          traj_cost += critical_cost_;
        } else if (!near_goal) {
          traj_cost += pose_cost;
        }
      }
      all_trajectories_collide &= trajectory_collide;
    }

    if (power_ > 1u) {
      data.costs += (repulsive_cost *
        (weight_ / static_cast<float>(strided_traj_cols))).pow(power_);
    } else {
      data.costs += repulsive_cost * (weight_ / static_cast<float>(strided_traj_cols));
    }
    data.fail_flag = all_trajectories_collide;
  }
};

namespace
{

geometry_msgs::msg::Point point(double x, double y)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  return p;
}

// Rectangular robot footprint of the Enway sweeper with its padding, a triangle, a concave
// polygon, an off-center irregular polygon and a two point footprint
std::vector<nav2_costmap_2d::Footprint> testFootprints()
{
  return {
    {point(2.33, 0.75), point(-1.5312, 0.75), point(-1.5312, -0.75), point(2.33, -0.75)},
    {point(0.9, 0.0), point(-0.4, 0.55), point(-0.4, -0.55)},
    {point(0.6, 0.5), point(-0.6, 0.5), point(-0.6, -0.5), point(0.6, -0.5), point(0.0, 0.05)},
    {point(1.13, 0.31), point(0.42, 0.97), point(-0.77, 0.44), point(-0.21, -0.36),
      point(0.66, -0.58)},
    {point(0.73, 0.12), point(-0.51, -0.2)},
  };
}

struct TestCostmap
{
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros;
  bool track_unknown;
};

// Costmap with an inflation layer, parameters overridden or added by name
TestCostmap makeCostmap(
  const std::string & name, bool track_unknown, double resolution, double inflation_radius = 1.6,
  const std::map<std::string, std::string> & overrides = {})
{
  std::map<std::string, std::string> parameters = {
    {"plugins", "[inflation_layer]"},
    {"inflation_layer.plugin", "nav2_costmap_2d::InflationLayer"},
    {"inflation_layer.inflation_radius", std::to_string(inflation_radius)},
    {"inflation_layer.cost_scaling_factor", "2.5"},
    {"width", "12"},
    {"height", "10"},
    {"resolution", std::to_string(resolution)},
    {"origin_x", "-1.3"},
    {"origin_y", "2.1"},
    {"rolling_window", "false"},
    {"track_unknown_space", track_unknown ? "true" : "false"},
    {"footprint", "'[[0.5, 0.3], [-0.5, 0.3], [-0.5, -0.3], [0.5, -0.3]]'"},
  };
  for (const auto & [parameter, value] : overrides) {
    parameters[parameter] = value;
  }
  // Costmap2DROS only keeps the arguments of the node options
  std::vector<std::string> arguments = {"--ros-args"};
  for (const auto & [parameter, value] : parameters) {
    arguments.push_back("-p");
    arguments.push_back(parameter + ":=" + value);
  }
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    name, "/", false, rclcpp::NodeOptions().arguments(arguments));
  costmap_ros->on_configure(rclcpp_lifecycle::State{});
  return {costmap_ros, track_unknown};
}

// Random obstacles (single cells and boxes) inflated by the costmap plugins, optionally further
// lethal boxes written after them like a keepout filter, then the costmap filters run, and
// optionally unknown cells
void fillRandomCostmap(
  nav2_costmap_2d::Costmap2DROS & costmap_ros, std::mt19937 & rng, bool with_unknown,
  bool with_keepout = false)
{
  auto & costmap = *costmap_ros.getCostmap();
  const unsigned int size_x = costmap.getSizeInCellsX();
  const unsigned int size_y = costmap.getSizeInCellsY();
  std::uniform_int_distribution<unsigned int> cell_x(0, size_x - 1);
  std::uniform_int_distribution<unsigned int> cell_y(0, size_y - 1);
  std::uniform_int_distribution<unsigned int> box_size(0, 4);
  std::uniform_int_distribution<int> count(0, 40);

  std::memset(costmap.getCharMap(), nav2_costmap_2d::FREE_SPACE, size_x * size_y);
  const int num_cells = count(rng);
  for (int n = 0; n < num_cells; ++n) {
    costmap.setCost(cell_x(rng), cell_y(rng), nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  auto add_boxes = [&](int num_boxes) {
      for (int n = 0; n < num_boxes; ++n) {
        const unsigned int x0 = cell_x(rng), y0 = cell_y(rng);
        const unsigned int w = box_size(rng), h = box_size(rng);
        for (unsigned int y = y0; y < std::min(size_y, y0 + h + 1); ++y) {
          for (unsigned int x = x0; x < std::min(size_x, x0 + w + 1); ++x) {
            costmap.setCost(x, y, nav2_costmap_2d::LETHAL_OBSTACLE);
          }
        }
      }
    };
  add_boxes(count(rng) / 5);
  auto * layered_costmap = costmap_ros.getLayeredCostmap();
  for (auto & layer : *layered_costmap->getPlugins()) {
    layer->updateCosts(costmap, 0, 0, static_cast<int>(size_x), static_cast<int>(size_y));
  }
  if (with_keepout) {
    add_boxes(1 + count(rng) / 10);
  }
  for (auto & layer : *layered_costmap->getFilters()) {
    layer->updateCosts(costmap, 0, 0, static_cast<int>(size_x), static_cast<int>(size_y));
  }
  if (with_unknown) {
    const int num_unknown = 3 * count(rng);
    for (int n = 0; n < num_unknown; ++n) {
      costmap.setCost(cell_x(rng), cell_y(rng), nav2_costmap_2d::NO_INFORMATION);
    }
  }
}

struct CriticSetup
{
  nav2::LifecycleNode::SharedPtr node;
  std::string name{"test"};
  std::unique_ptr<ParametersHandler> param_handler;
  CostCriticWrapper critic;
};

void configureCritic(
  CriticSetup & setup, const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros,
  bool use_prefilter = true)
{
  setup.node = std::make_shared<nav2::LifecycleNode>("cost_critic_test_node");
  setup.param_handler = std::make_unique<ParametersHandler>(setup.node, setup.name);
  auto getParam = setup.param_handler->getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", true);
  // not a multiple of 1/2^k to make the order of the sums matter
  double critical_cost;
  getParam(critical_cost, "critical_cost", 300.3);
  bool use_footprint_prefilter;
  getParam(use_footprint_prefilter, "use_footprint_prefilter", use_prefilter);
  setup.critic.on_configure(setup.node, "mppi", "critic", costmap_ros, setup.param_handler.get());
}

}  // namespace

TEST(CostCriticTest, BlockedCellDistanceIsChessboardDistance)
{
  std::mt19937 rng(42);
  auto test_costmap = makeCostmap("distance_costmap", true, 0.1);
  CriticSetup setup;
  configureCritic(setup, test_costmap.costmap_ros);
  auto & costmap = *test_costmap.costmap_ros->getCostmap();
  const int size_x = static_cast<int>(costmap.getSizeInCellsX());
  const int size_y = static_cast<int>(costmap.getSizeInCellsY());
  std::uniform_int_distribution<int> region_x(0, size_x - 1);
  std::uniform_int_distribution<int> region_y(0, size_y - 1);

  for (int map = 0; map < 20; ++map) {
    fillRandomCostmap(*test_costmap.costmap_ros, rng, map % 2 == 0);
    setup.critic.updateCostmapInfo();
    // whole map, then random regions, some of them touching the map border
    int x_min = 0, y_min = 0, x_max = size_x - 1, y_max = size_y - 1;
    if (map >= 4) {
      x_min = region_x(rng), x_max = region_x(rng), y_min = region_y(rng), y_max = region_y(rng);
      if (x_min > x_max) {
        std::swap(x_min, x_max);
      }
      if (y_min > y_max) {
        std::swap(y_min, y_max);
      }
      x_min = map % 3 == 0 ? 0 : x_min;
      y_max = map % 5 == 0 ? size_y - 1 : y_max;
    }
    setup.critic.updateBlockedCellDistance(x_min, y_min, x_max, y_max);

    // blocked cells of the region and virtual blocked cells around its sides inside the map
    std::vector<std::pair<int, int>> blocked;
    for (int y = y_min - 1; y <= y_max + 1; ++y) {
      for (int x = x_min - 1; x <= x_max + 1; ++x) {
        const bool in_map = x >= 0 && y >= 0 && x < size_x && y < size_y;
        const bool in_region = x >= x_min && y >= y_min && x <= x_max && y <= y_max;
        if (in_map && (!in_region || costmap.getCost(x, y) >= nav2_costmap_2d::LETHAL_OBSTACLE)) {
          blocked.emplace_back(x, y);
        }
      }
    }
    for (int y = y_min; y <= y_max; ++y) {
      for (int x = x_min; x <= x_max; ++x) {
        int expected = 255;
        for (const auto & [bx, by] : blocked) {
          expected = std::min(expected, std::max(std::abs(bx - x), std::abs(by - y)));
        }
        ASSERT_EQ(setup.critic.blocked_cell_distance_[y * size_x + x], expected) <<
          "map " << map << " cell " << x << ", " << y;
      }
    }
  }
}

TEST(CostCriticTest, FootprintVertexTransformMatchesCostmap)
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> position(-40.0, 40.0);
  std::uniform_real_distribution<float> yaw(-4.0f, 4.0f);
  for (const auto & footprint : testFootprints()) {
    for (int n = 0; n < 20000; ++n) {
      const double x = position(rng), y = position(rng);
      const double theta = static_cast<double>(yaw(rng));
      std::vector<geometry_msgs::msg::Point> expected;
      nav2_costmap_2d::transformFootprint(x, y, theta, footprint, expected);
      const double cos_th = cos(theta);
      const double sin_th = sin(theta);
      for (size_t i = 0; i < footprint.size(); ++i) {
        double wx, wy;
        CostCriticWrapper::transformFootprintVertex(x, y, cos_th, sin_th, footprint[i], wx, wy);
        // Equal up to the rounding of the products, which depends on how each library
        // is compiled
        ASSERT_NEAR(wx, expected[i].x, 1e-12);
        ASSERT_NEAR(wy, expected[i].y, 1e-12);
      }
    }
  }
}

TEST(CostCriticTest, FootprintCollisionMatchesReference)
{
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> yaw(-3.5f, 3.5f);
  std::uniform_int_distribution<int> right_angle(-4, 4);
  std::bernoulli_distribution on_cell_border(0.2);
  std::bernoulli_distribution axis_aligned(0.2);
  size_t num_poses = 0, num_collisions = 0;

  for (bool track_unknown : {false, true}) {
    for (double resolution : {0.1, 0.05}) {
      auto test_costmap = makeCostmap(
        std::string("footprint_costmap_") + (track_unknown ? "t" : "f") +
        std::to_string(static_cast<int>(resolution * 100)), track_unknown, resolution);
      CriticSetup setup;
      configureCritic(setup, test_costmap.costmap_ros);
      auto & costmap = *test_costmap.costmap_ros->getCostmap();
      const double origin_x = costmap.getOriginX(), origin_y = costmap.getOriginY();
      const double size_x = costmap.getSizeInMetersX(), size_y = costmap.getSizeInMetersY();
      std::uniform_real_distribution<float> pos_x(
        static_cast<float>(origin_x - 1.0), static_cast<float>(origin_x + size_x + 1.0));
      std::uniform_real_distribution<float> pos_y(
        static_cast<float>(origin_y - 1.0), static_cast<float>(origin_y + size_y + 1.0));
      std::uniform_int_distribution<int> cell(-10, static_cast<int>(size_x / resolution) + 10);

      for (const auto & footprint : testFootprints()) {
        for (int map = 0; map < 12; ++map) {
          fillRandomCostmap(*test_costmap.costmap_ros, rng, map % 3 != 0);
          setup.critic.updateCostmapInfo();
          setup.critic.updateBlockedCellDistance(
            0, 0, static_cast<int>(costmap.getSizeInCellsX()) - 1,
            static_cast<int>(costmap.getSizeInCellsY()) - 1);
          ASSERT_EQ(setup.critic.is_tracking_unknown_, track_unknown);
          ASSERT_LE(setup.critic.possible_collision_cost_, 255.0f);

          for (int n = 0; n < 1500; ++n) {
            float x = pos_x(rng), y = pos_y(rng), theta = yaw(rng);
            if (on_cell_border(rng)) {
              x = static_cast<float>(origin_x + resolution * cell(rng));
              y = static_cast<float>(origin_y + resolution * cell(rng));
            }
            if (axis_aligned(rng)) {
              theta = static_cast<float>(M_PI_2 * right_angle(rng));
            }
            const bool expected =
              setup.critic.referenceFootprintInCollision(x, y, theta, footprint);
            ASSERT_EQ(setup.critic.footprintInCollision(x, y, theta, footprint), expected) <<
              "pose " << x << ", " << y << ", " << theta << " map " << map;
            num_collisions += expected;
            ++num_poses;
          }
        }
      }
    }
  }
  // both outcomes are well represented
  EXPECT_GT(num_collisions, num_poses / 5);
  EXPECT_LT(num_collisions, num_poses * 4 / 5);
  std::cout << num_poses << " poses, " << num_collisions << " in collision" << std::endl;
}

TEST(CostCriticTest, ScoreMatchesReference)
{
  std::mt19937 rng(99);
  std::normal_distribution<float> step(0.0f, 0.08f);
  std::normal_distribution<float> turn(0.0f, 0.15f);
  std::uniform_real_distribution<float> yaw(-3.2f, 3.2f);
  constexpr int batch_size = 203;
  constexpr int time_steps = 27;
  size_t num_trajectories = 0, num_collisions = 0, num_fails = 0;

  struct Config
  {
    bool track_unknown;
    double inflation_radius;
    bool use_prefilter;
    bool keepout;
  };
  // An inflation radius below the circumscribed radius makes every point need a footprint check
  for (const Config & config : {Config{false, 2.6, true, true}, Config{true, 2.6, true, false},
      Config{false, 1.0, true, false}, Config{false, 2.6, false, false}})
  {
    std::map<std::string, std::string> keepout_inflation;
    if (config.keepout) {
      keepout_inflation = {
        {"filters", "[keepout_inflation]"},
        {"keepout_inflation.plugin", "nav2_costmap_2d::InflationLayer"},
        {"keepout_inflation.inflation_radius", "2.6"},
        {"keepout_inflation.cost_scaling_factor", "4.0"}};
    }
    auto test_costmap = makeCostmap(
      "score_costmap_" + std::to_string(num_trajectories), config.track_unknown, 0.1,
      config.inflation_radius, keepout_inflation);
    CriticSetup setup;
    configureCritic(setup, test_costmap.costmap_ros, config.use_prefilter);
    auto & costmap = *test_costmap.costmap_ros->getCostmap();
    std::uniform_real_distribution<float> pos_x(
      static_cast<float>(costmap.getOriginX() - 0.5),
      static_cast<float>(costmap.getOriginX() + costmap.getSizeInMetersX() + 0.5));
    std::uniform_real_distribution<float> pos_y(
      static_cast<float>(costmap.getOriginY() - 0.5),
      static_cast<float>(costmap.getOriginY() + costmap.getSizeInMetersY() + 0.5));

    for (const auto & footprint : testFootprints()) {
      if (footprint.size() < 3) {
        continue;  // not a valid costmap footprint
      }
      test_costmap.costmap_ros->setRobotFootprint(footprint);
      for (int map = 0; map < 9; ++map) {
        fillRandomCostmap(*test_costmap.costmap_ros, rng, map % 2 == 1, config.keepout);
        if (map == 8) {
          // every trajectory in collision
          std::memset(
            costmap.getCharMap(), nav2_costmap_2d::LETHAL_OBSTACLE,
            costmap.getSizeInCellsX() * costmap.getSizeInCellsY());
        }

        models::State state;
        state.local_path_length = map % 4 == 0 ? 0.1f : 5.0f;
        models::Trajectories trajectories;
        trajectories.reset(batch_size, time_steps);
        // Every third map the trajectories start close together, so that the footprint checks
        // only need a part of the map that changes from call to call
        const bool localized = map % 3 == 2;
        const float center_x = pos_x(rng), center_y = pos_y(rng);
        std::uniform_real_distribution<float> spread(-0.4f, 0.4f);
        for (int i = 0; i < batch_size; ++i) {
          float x = localized ? center_x + spread(rng) : pos_x(rng);
          float y = localized ? center_y + spread(rng) : pos_y(rng);
          float theta = yaw(rng);
          for (int j = 0; j < time_steps; ++j) {
            trajectories.x(i, j) = x;
            trajectories.y(i, j) = y;
            trajectories.yaws(i, j) = theta;
            theta += turn(rng);
            const float distance = std::abs(step(rng));
            x += distance * std::cos(theta);
            y += distance * std::sin(theta);
          }
        }
        models::Path path;
        geometry_msgs::msg::Pose goal;
        float model_dt = 0.1f;
        Eigen::ArrayXf costs = Eigen::ArrayXf::Constant(batch_size, 0.5f);
        Eigen::ArrayXf expected_costs = costs;
        CriticData data =
        {state, trajectories, path, goal, costs, std::nullopt, model_dt,
          false, nullptr, nullptr, std::nullopt, std::nullopt};
        CriticData expected_data =
        {state, trajectories, path, goal, expected_costs, std::nullopt, model_dt,
          false, nullptr, nullptr, std::nullopt, std::nullopt};

        setup.critic.score(data);
        setup.critic.referenceScore(expected_data);
        for (int i = 0; i < batch_size; ++i) {
          ASSERT_EQ(costs(i), expected_costs(i)) << "trajectory " << i << " map " << map;
          num_collisions += costs(i) > 1000.0f;
        }
        EXPECT_EQ(data.fail_flag, expected_data.fail_flag);
        num_fails += data.fail_flag;
        num_trajectories += batch_size;
      }
    }
  }
  EXPECT_GT(num_collisions, num_trajectories / 5);
  EXPECT_LT(num_collisions, num_trajectories * 4 / 5);
  EXPECT_GE(num_fails, 12u);
  std::cout << num_trajectories << " trajectories, " << num_collisions << " in collision" <<
    std::endl;
}

TEST(CostCriticTest, PrefilterNeverSkipsACollision)
{
  std::mt19937 rng(2024);
  std::uniform_real_distribution<float> yaw(-3.5f, 3.5f);
  std::uniform_real_distribution<float> offset(-0.5f, 0.5f);
  size_t num_poses = 0, num_collisions = 0, num_skipped = 0;

  // The prefilter does not depend on how the costmap is inflated: half of the maps have
  // lethal boxes that no layer inflated, and unknown cells
  for (double resolution : {0.1, 0.05}) {
    for (bool track_unknown : {false, true}) {
      auto test_costmap = makeCostmap(
        "prefilter_costmap_" + std::to_string(num_poses), track_unknown, resolution, 2.2);
      CriticSetup setup;
      configureCritic(setup, test_costmap.costmap_ros);
      auto & costmap = *test_costmap.costmap_ros->getCostmap();
      const int size_x = static_cast<int>(costmap.getSizeInCellsX());
      const int size_y = static_cast<int>(costmap.getSizeInCellsY());
      std::uniform_real_distribution<float> pos_x(
        static_cast<float>(costmap.getOriginX() - 1.0),
        static_cast<float>(costmap.getOriginX() + costmap.getSizeInMetersX() + 1.0));
      std::uniform_real_distribution<float> pos_y(
        static_cast<float>(costmap.getOriginY() - 1.0),
        static_cast<float>(costmap.getOriginY() + costmap.getSizeInMetersY() + 1.0));

      for (const auto & footprint : testFootprints()) {
        for (int map = 0; map < 12; ++map) {
          fillRandomCostmap(*test_costmap.costmap_ros, rng, map % 3 == 1, map % 2 == 1);
          if (map == 0) {
            // without obstacles only the map border makes collisions
            std::memset(
              costmap.getCharMap(), nav2_costmap_2d::FREE_SPACE,
              static_cast<size_t>(size_x) * static_cast<size_t>(size_y));
          }
          setup.critic.updateCostmapInfo();
          setup.critic.updateBlockedCellDistance(0, 0, size_x - 1, size_y - 1);
          setup.critic.updateFootprintPrefilter(footprint);
          ASSERT_TRUE(setup.critic.prefilter_active_);

          for (int n = 0; n < 2000; ++n) {
            const float x = pos_x(rng), y = pos_y(rng), theta = yaw(rng);
            const bool expected =
              setup.critic.referenceFootprintInCollision(x, y, theta, footprint);
            if (setup.critic.footprintCollisionFree(x, y, theta)) {
              ASSERT_FALSE(expected) << "pose " << x << ", " << y << ", " << theta <<
                " map " << map;
              ++num_skipped;
              // poses just beside a skipped one probe the margin of the prefilter
              for (int m = 0; m < 4; ++m) {
                const float near_x = x + offset(rng), near_y = y + offset(rng);
                const float near_theta = theta + 0.3f * offset(rng);
                if (setup.critic.footprintCollisionFree(near_x, near_y, near_theta)) {
                  ASSERT_FALSE(
                    setup.critic.referenceFootprintInCollision(
                      near_x, near_y, near_theta, footprint)) <<
                    "pose " << near_x << ", " << near_y << ", " << near_theta << " map " << map;
                }
              }
            }
            num_collisions += expected;
            ++num_poses;
          }
        }
      }
    }
  }
  std::cout << num_poses << " poses, " << num_collisions << " in collision, " << num_skipped <<
    " skipped" << std::endl;
  // the prefilter skips a good part of the free poses
  EXPECT_GT(num_skipped, (num_poses - num_collisions) / 4);
}

TEST(CostCriticTest, PrefilterCoversRasterizedOutline)
{
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> yaw(-3.5f, 3.5f);
  std::uniform_int_distribution<int> eighth_turn(-8, 8);
  std::bernoulli_distribution on_cell_border(0.3);
  std::bernoulli_distribution aligned(0.3);
  // blocked_cell_distance_ is written within this many cells of the blocked cell
  constexpr int window = 32;
  size_t num_cells = 0;

  // A single blocked cell anywhere on the footprint outline, as the footprint collision
  // checker rasterizes it, must keep the prefilter from clearing the pose
  for (double resolution : {0.1, 0.05}) {
    auto test_costmap = makeCostmap(
      "outline_costmap_" + std::to_string(static_cast<int>(resolution * 100)), false,
      resolution);
    CriticSetup setup;
    configureCritic(setup, test_costmap.costmap_ros);
    auto & costmap = *test_costmap.costmap_ros->getCostmap();
    const int size_x = static_cast<int>(costmap.getSizeInCellsX());
    const int size_y = static_cast<int>(costmap.getSizeInCellsY());
    const double origin_x = costmap.getOriginX(), origin_y = costmap.getOriginY();
    std::memset(
      costmap.getCharMap(), nav2_costmap_2d::FREE_SPACE,
      static_cast<size_t>(size_x) * static_cast<size_t>(size_y));
    setup.critic.updateCostmapInfo();
    auto & distance = setup.critic.blocked_cell_distance_;
    distance.assign(static_cast<size_t>(size_x) * static_cast<size_t>(size_y), 255u);
    // poses far enough from the map border for the prefilter to clear them
    std::uniform_real_distribution<float> pos_x(
      static_cast<float>(origin_x + 4.5), static_cast<float>(origin_x + 7.5));
    std::uniform_real_distribution<float> pos_y(
      static_cast<float>(origin_y + 4.5), static_cast<float>(origin_y + 5.5));
    std::uniform_int_distribution<int> cell(0, 9);

    for (const auto & footprint : testFootprints()) {
      setup.critic.updateFootprintPrefilter(footprint);
      ASSERT_TRUE(setup.critic.prefilter_active_);
      for (uint8_t min_distance : setup.critic.prefilter_min_distance_) {
        ASSERT_LE(min_distance, window);
      }

      for (int n = 0; n < 120; ++n) {
        float x = pos_x(rng), y = pos_y(rng), theta = yaw(rng);
        if (on_cell_border(rng)) {
          x = static_cast<float>(origin_x + 5.0 + resolution * cell(rng));
          y = static_cast<float>(origin_y + 5.0 + resolution * cell(rng));
        }
        if (aligned(rng)) {
          theta = static_cast<float>(M_PI_4 * eighth_turn(rng));
        }
        ASSERT_TRUE(setup.critic.footprintCollisionFree(x, y, theta));

        std::vector<geometry_msgs::msg::Point> oriented;
        nav2_costmap_2d::transformFootprint(x, y, theta, footprint, oriented);
        for (size_t i = 0; i < oriented.size(); ++i) {
          const auto & from = oriented[i];
          const auto & to = oriented[(i + 1) % oriented.size()];
          unsigned int from_x, from_y, to_x, to_y;
          ASSERT_TRUE(costmap.worldToMap(from.x, from.y, from_x, from_y));
          ASSERT_TRUE(costmap.worldToMap(to.x, to.y, to_x, to_y));
          for (nav2_util::LineIterator line(from_x, from_y, to_x, to_y); line.isValid();
            line.advance())
          {
            // chessboard distance to the single blocked cell
            const int blocked_x = line.getX(), blocked_y = line.getY();
            auto fill = [&](bool blocked) {
                for (int cy = std::max(0, blocked_y - window);
                  cy <= std::min(size_y - 1, blocked_y + window); ++cy)
                {
                  for (int cx = std::max(0, blocked_x - window);
                    cx <= std::min(size_x - 1, blocked_x + window); ++cx)
                  {
                    distance[cy * size_x + cx] = static_cast<uint8_t>(
                      blocked ? std::max(std::abs(cx - blocked_x), std::abs(cy - blocked_y)) : 255);
                  }
                }
              };
            fill(true);
            ASSERT_FALSE(setup.critic.footprintCollisionFree(x, y, theta)) <<
              "pose " << x << ", " << y << ", " << theta << " blocked cell " << blocked_x <<
              ", " << blocked_y;
            fill(false);
            ++num_cells;
          }
        }
      }
    }
  }
  std::cout << num_cells << " outline cells" << std::endl;
}

TEST(CostCriticTest, PrefilterActivation)
{
  std::mt19937 rng(5);
  const auto footprint = testFootprints()[0];
  auto check = [&](
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros, bool use_prefilter,
    const nav2_costmap_2d::Footprint & test_footprint, bool with_unknown) {
      CriticSetup setup;
      configureCritic(setup, costmap_ros, use_prefilter);
      fillRandomCostmap(*costmap_ros, rng, with_unknown);
      setup.critic.updateCostmapInfo();
      setup.critic.updateFootprintPrefilter(test_footprint);
      return setup.critic.prefilter_active_;
    };

  auto costmap = makeCostmap("activation_costmap", false, 0.1, 2.0).costmap_ros;
  EXPECT_TRUE(check(costmap, true, footprint, false));
  // unknown cells count as blocked cells
  EXPECT_TRUE(check(costmap, true, footprint, true));
  // it does not depend on an inflation layer
  EXPECT_TRUE(
    check(
      makeCostmap(
        "no_inflation_costmap", false, 0.1, 2.0,
        {{"plugins", "[obstacle_layer]"},
          {"obstacle_layer.plugin", "nav2_costmap_2d::ObstacleLayer"}}).costmap_ros, true,
      footprint, false));
  // disabled by parameter
  EXPECT_FALSE(check(costmap, false, footprint, false));
  // degenerate footprint
  EXPECT_FALSE(check(costmap, true, {footprint[0]}, false));
}

TEST(CostCriticTest, NonFiniteTrajectoriesCollide)
{
  std::mt19937 rng(11);
  constexpr int batch_size = 70;
  constexpr int time_steps = 9;
  const float not_a_number = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  auto test_costmap = makeCostmap("non_finite_costmap", false, 0.1, 2.6);
  CriticSetup setup;
  configureCritic(setup, test_costmap.costmap_ros);
  auto & costmap = *test_costmap.costmap_ros->getCostmap();
  std::uniform_real_distribution<float> pos_x(
    static_cast<float>(costmap.getOriginX() + 1.0),
    static_cast<float>(costmap.getOriginX() + costmap.getSizeInMetersX() - 1.0));
  std::uniform_real_distribution<float> pos_y(
    static_cast<float>(costmap.getOriginY() + 1.0),
    static_cast<float>(costmap.getOriginY() + costmap.getSizeInMetersY() - 1.0));
  std::uniform_real_distribution<float> yaw(-3.2f, 3.2f);

  // Trajectories with a point that is not finite are in collision, the other trajectories are
  // scored as without them. In the last round no point has a finite position.
  for (int round = 0; round < 4; ++round) {
    fillRandomCostmap(*test_costmap.costmap_ros, rng, false);
    models::State state;
    state.local_path_length = 5.0f;
    models::Trajectories trajectories, finite_trajectories;
    trajectories.reset(batch_size, time_steps);
    for (int i = 0; i < batch_size; ++i) {
      const float x = pos_x(rng), y = pos_y(rng), theta = yaw(rng);
      for (int j = 0; j < time_steps; ++j) {
        trajectories.x(i, j) = x + 0.05f * static_cast<float>(j) * std::cos(theta);
        trajectories.y(i, j) = y + 0.05f * static_cast<float>(j) * std::sin(theta);
        trajectories.yaws(i, j) = theta;
      }
    }
    finite_trajectories = trajectories;
    std::vector<bool> finite(batch_size, true);
    for (int i = 0; i < batch_size; ++i) {
      if (round == 3) {
        trajectories.x.row(i).setConstant(not_a_number);
        finite[i] = false;
      } else if (i % 7 == 3) {
        const float value = i % 3 == 0 ? not_a_number : (i % 3 == 1 ? infinity : -infinity);
        // an even time step, as the critic scores every second point
        const int step = 2 * (i % ((time_steps + 1) / 2));
        (i % 2 == 0 ? trajectories.x : trajectories.y)(i, step) = value;
        finite[i] = false;
      }
    }

    models::Path path;
    geometry_msgs::msg::Pose goal;
    float model_dt = 0.1f;
    Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(batch_size);
    Eigen::ArrayXf expected_costs = costs;
    CriticData data =
    {state, trajectories, path, goal, costs, std::nullopt, model_dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};
    CriticData expected_data =
    {state, finite_trajectories, path, goal, expected_costs, std::nullopt, model_dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};
    setup.critic.score(data);
    setup.critic.referenceScore(expected_data);

    // a trajectory in collision gets the collision cost scaled by the critic weight
    const float collision_cost = expected_costs.maxCoeff();
    for (int i = 0; i < batch_size; ++i) {
      if (finite[i]) {
        EXPECT_EQ(costs(i), expected_costs(i)) << "trajectory " << i << " round " << round;
      } else {
        EXPECT_GT(costs(i), 1000.0f) << "trajectory " << i << " round " << round;
        if (collision_cost > 1000.0f) {
          EXPECT_EQ(costs(i), collision_cost) << "trajectory " << i << " round " << round;
        }
      }
    }
    if (round == 3) {
      EXPECT_TRUE(data.fail_flag);
    }
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(0, nullptr);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
