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

#include <cmath>
#include <chrono>
#include "nav2_mppi_controller/critics/cost_critic.hpp"
#include "nav2_core/controller_exceptions.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mppi::critics
{

void CostCritic::initialize()
{
  auto getParentParam = parameters_handler_->getParamGetter(parent_name_);
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(consider_footprint_, "consider_footprint", false);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 3.81f);
  getParam(critical_cost_, "critical_cost", 300.0f);
  getParam(near_collision_cost_, "near_collision_cost", 253);
  getParam(collision_cost_, "collision_cost", 1000000.0f);
  getParam(near_goal_distance_, "near_goal_distance", 0.5f);
  getParam(inflation_layer_name_, "inflation_layer_name", std::string(""));
  getParam(trajectory_point_step_, "trajectory_point_step", 2);

  // Normalized by cost value to put in same regime as other weights
  weight_ /= 254.0f;

  // Normalize weight when parameter is changed dynamically as well
  auto weightDynamicCb = [&](
    const rclcpp::Parameter & weight, rcl_interfaces::msg::SetParametersResult & /*result*/) {
      weight_ = weight.as_double() / 254.0f;
    };
  parameters_handler_->addParamCallback(name_ + ".cost_weight", weightDynamicCb);

  collision_checker_.setCostmap(costmap_);
  possible_collision_cost_ = findCircumscribedCost(costmap_ros_);

  if (possible_collision_cost_ < 1.0f) {
    RCLCPP_ERROR(
      logger_,
      "Inflation layer either not found or inflation is not set sufficiently for "
      "optimized non-circular collision checking capabilities. It is HIGHLY recommended to set"
      " the inflation radius to be at MINIMUM half of the robot's largest cross-section. See "
      "github.com/ros-planning/navigation2/tree/main/nav2_smac_planner#potential-fields"
      " for full instructions. This will substantially impact run-time performance.");
  }

  if (costmap_ros_->getUseRadius() == consider_footprint_) {
    RCLCPP_WARN(
      logger_,
      "Inconsistent configuration in collision checking. Please verify the robot's shape settings "
      "in both the costmap and the cost critic.");
    if (costmap_ros_->getUseRadius()) {
      throw nav2_core::ControllerException(
              "Considering footprint in CostCritic but no robot footprint provided in the "
              "costmap (robot radius used instead). Disable considering footprint.");
    }
  }

  if (near_collision_cost_ > 253) {
    RCLCPP_WARN(logger_, "Near collision cost is set higher than INSCRIBED_INFLATED_OBSTACLE");
  }

  RCLCPP_INFO(
    logger_,
    "InflationCostCritic instantiated with %d power and %f / %f weights. "
    "Critic will collision check based on %s cost.",
    power_, critical_cost_, weight_, consider_footprint_ ?
    "footprint" : "circular");

#ifdef _OPENMP
  RCLCPP_INFO(logger_, "OpenMP enabled with max %d threads", omp_get_max_threads());
#else
  RCLCPP_WARN(logger_, "OpenMP NOT enabled - running single-threaded");
#endif

  // Initialize timing statistics
  stats_count_ = 0;
  stats_sum_ = 0;
  stats_min_ = std::numeric_limits<long>::max();
  stats_max_ = 0;
  profile_setup_sum_ = 0;
  profile_loop_sum_ = 0;
  profile_power_sum_ = 0;
}

float CostCritic::findCircumscribedCost(
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap)
{
  double result = -1.0;
  const double circum_radius = costmap->getLayeredCostmap()->getCircumscribedRadius();
  if (static_cast<float>(circum_radius) == circumscribed_radius_) {
    // early return if footprint size is unchanged
    return circumscribed_cost_;
  }

  // check if the costmap has an inflation layer
  const auto inflation_layer = nav2_costmap_2d::InflationLayer::getInflationLayer(
    costmap,
    inflation_layer_name_);
  if (inflation_layer != nullptr) {
    const double resolution = costmap->getCostmap()->getResolution();
    double inflation_radius = inflation_layer->getInflationRadius();
    if (inflation_radius < circum_radius) {
      RCLCPP_ERROR(
        rclcpp::get_logger("computeCircumscribedCost"),
        "The inflation radius (%f) is smaller than the circumscribed radius (%f) "
        "If this is an SE2-collision checking plugin, it cannot use costmap potential "
        "field to speed up collision checking by only checking the full footprint "
        "when robot is within possibly-inscribed radius of an obstacle. This may "
        "significantly slow down planning times!",
        inflation_radius, circum_radius);
      result = 0.0;
      return result;
    }
    result = inflation_layer->computeCost(circum_radius / resolution);
  } else {
    RCLCPP_WARN(
      logger_,
      "No inflation layer found in costmap configuration. "
      "If this is an SE2-collision checking plugin, it cannot use costmap potential "
      "field to speed up collision checking by only checking the full footprint "
      "when robot is within possibly-inscribed radius of an obstacle. This may "
      "significantly slow down planning times and not avoid anything but absolute collisions!");
  }

  circumscribed_radius_ = static_cast<float>(circum_radius);
  circumscribed_cost_ = static_cast<float>(result);

  return circumscribed_cost_;
}

void CostCritic::score(CriticData & data)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  if (!enabled_) {
    return;
  }

  // Setup cost information for various parts of the critic
  is_tracking_unknown_ = costmap_ros_->getLayeredCostmap()->isTrackingUnknown();
  auto * costmap = collision_checker_.getCostmap();
  origin_x_ = static_cast<float>(costmap->getOriginX());
  origin_y_ = static_cast<float>(costmap->getOriginY());
  resolution_ = static_cast<float>(costmap->getResolution());
  size_x_ = costmap->getSizeInCellsX();
  size_y_ = costmap->getSizeInCellsY();

  if (consider_footprint_) {
    // footprint may have changed since initialization if user has dynamic footprints
    possible_collision_cost_ = findCircumscribedCost(costmap_ros_);
  }

  // If near the goal, don't apply the preferential term since the goal is near obstacles
  bool near_goal = false;
  if (data.state.local_path_length < near_goal_distance_) {
    near_goal = true;
  }

  Eigen::ArrayXf repulsive_cost(data.costs.rows());
  repulsive_cost.setZero();
  bool all_trajectories_collide = true;

  int strided_traj_cols = floor((data.trajectories.x.cols() - 1) / trajectory_point_step_) + 1;
  int strided_traj_rows = data.trajectories.x.rows();
  int outer_stride = strided_traj_rows * trajectory_point_step_;

  const auto traj_x = Eigen::Map<const Eigen::ArrayXXf, 0,
      Eigen::Stride<-1, -1>>(
    data.trajectories.x.data(), strided_traj_rows, strided_traj_cols,
    Eigen::Stride<-1, -1>(outer_stride, 1));
  const auto traj_y = Eigen::Map<const Eigen::ArrayXXf, 0,
      Eigen::Stride<-1, -1>>(
    data.trajectories.y.data(), strided_traj_rows, strided_traj_cols,
    Eigen::Stride<-1, -1>(outer_stride, 1));
  const auto traj_yaw = Eigen::Map<const Eigen::ArrayXXf, 0,
      Eigen::Stride<-1, -1>>(
    data.trajectories.yaws.data(), strided_traj_rows, strided_traj_cols,
    Eigen::Stride<-1, -1>(outer_stride, 1));

  auto setup_end = std::chrono::high_resolution_clock::now();
  long setup_time = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - start_time).count();

#ifdef _OPENMP
  int num_threads_used = 1;
  int max_threads = omp_get_max_threads();

  if (stats_count_ == 0) {
    RCLCPP_INFO(logger_, "OpenMP: max=%d, rows=%d, in_parallel=%d, limit=%d",
                max_threads, strided_traj_rows, omp_in_parallel(), omp_get_thread_limit());
  }

  #pragma omp parallel for schedule(dynamic, 16) num_threads(6)
#endif
  for (int i = 0; i < strided_traj_rows; ++i) {
#ifdef _OPENMP
    if (i == 0) {
      num_threads_used = omp_get_num_threads();
    }
#endif
    bool trajectory_collide = false;
    float pose_cost = 0.0f;
    float & traj_cost = repulsive_cost(i);

    for (int j = 0; j < strided_traj_cols; j++) {
      float Tx = traj_x(i, j);
      float Ty = traj_y(i, j);
      unsigned int x_i = 0u, y_i = 0u;

      // The getCost doesn't use orientation
      // The footprintCostAtPose will always return "INSCRIBED" if footprint is over it
      // So the center point has more information than the footprint
      if (!worldToMapFloat(Tx, Ty, x_i, y_i)) {
        pose_cost = 255.0f;  // NO_INFORMATION in float
      } else {
        pose_cost = static_cast<float>(costmap->getCost(getIndex(x_i, y_i)));
        if (pose_cost < 1.0f) {
          continue;  // In free space
        }
      }

      if (inCollision(pose_cost, Tx, Ty, traj_yaw(i, j))) {
        traj_cost = collision_cost_;
        trajectory_collide = true;
        break;
      }

      // Let near-collision trajectory points be punished severely
      // Note that we collision check based on the footprint actual,
      // but score based on the center-point cost regardless
      if (pose_cost >= static_cast<float>(near_collision_cost_)) {
        traj_cost += critical_cost_;
      } else if (!near_goal) {  // Generally prefer trajectories further from obstacles
        traj_cost += pose_cost;
      }
    }

#ifdef _OPENMP
    if (!trajectory_collide) {
      #pragma omp atomic write
      all_trajectories_collide = false;
    }
#else
    all_trajectories_collide &= trajectory_collide;
#endif
  }

  auto loop_end = std::chrono::high_resolution_clock::now();
  long loop_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - setup_end).count();

  if (power_ > 1u) {
    data.costs += (repulsive_cost *
      (weight_ / static_cast<float>(strided_traj_cols))).pow(power_);
  } else {
    data.costs += repulsive_cost * (weight_ / static_cast<float>(strided_traj_cols));
  }

  auto power_end = std::chrono::high_resolution_clock::now();
  long power_time = std::chrono::duration_cast<std::chrono::microseconds>(power_end - loop_end).count();

  data.fail_flag = all_trajectories_collide;

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  long exec_time = duration.count();

  // Update running statistics
  stats_count_++;
  stats_sum_ += exec_time;
  if (exec_time < stats_min_) stats_min_ = exec_time;
  if (exec_time > stats_max_) stats_max_ = exec_time;

  // Accumulate profiling data
  profile_setup_sum_ += setup_time;
  profile_loop_sum_ += loop_time;
  profile_power_sum_ += power_time;

  // Report statistics every 100 executions
  if (stats_count_ % 100 == 0) {
    double avg = static_cast<double>(stats_sum_) / stats_count_;
    double avg_setup = static_cast<double>(profile_setup_sum_) / stats_count_;
    double avg_loop = static_cast<double>(profile_loop_sum_) / stats_count_;
    double avg_power = static_cast<double>(profile_power_sum_) / stats_count_;
    double setup_pct = (avg_setup / avg) * 100.0;
    double loop_pct = (avg_loop / avg) * 100.0;
    double power_pct = (avg_power / avg) * 100.0;

#ifdef _OPENMP
    RCLCPP_INFO(
      logger_,
      "CostCritic timing (%zu calls, %dx%d, %d threads): total=%.1f µs [min=%ld, max=%ld] | "
      "setup=%.1f µs (%.1f%%) | loop=%.1f µs (%.1f%%) | power=%.1f µs (%.1f%%)",
      stats_count_, strided_traj_rows, strided_traj_cols, num_threads_used,
      avg, stats_min_, stats_max_,
      avg_setup, setup_pct, avg_loop, loop_pct, avg_power, power_pct);
#else
    RCLCPP_INFO(
      logger_,
      "CostCritic timing (%zu calls, %dx%d, NO OMP): total=%.1f µs [min=%ld, max=%ld] | "
      "setup=%.1f µs (%.1f%%) | loop=%.1f µs (%.1f%%) | power=%.1f µs (%.1f%%)",
      stats_count_, strided_traj_rows, strided_traj_cols,
      avg, stats_min_, stats_max_,
      avg_setup, setup_pct, avg_loop, loop_pct, avg_power, power_pct);
#endif
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::CostCritic,
  mppi::critics::CriticFunction)
