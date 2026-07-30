// Copyright (c) 2026 Open Navigation LLC
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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLE_BYPASS_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLE_BYPASS_CRITIC_HPP_

#include <optional>
#include <string>

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/path.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace mppi::critics
{

/**
 * @class mppi::critics::ObstacleBypassCritic
 * @brief Critic objective function for steering around dynamic obstacles
 * blocking the path. Uses the costmap to determine which side of the
 * obstacle is best to bypass and how far to offset from the path.
 */
class ObstacleBypassCritic : public CriticFunction
{
public:
  /**
   * @brief Initialize critic parameters and runtime state.
   */
  void initialize() override;

  /**
   * @brief Score trajectories based on obstacle bypass objective.
   * @param data Critic data containing trajectories, path, and context.
   */
  void score(CriticData & data) override;

protected:
  struct BypassResult
  {
    float target_x;
    float target_y;
    int sign;  // +1 left, -1 right
  };

  /**
   * @brief Determine the best side and target to bypass an obstacle
   * @param path The pruned local path.
   * @param robot_x,robot_y Robot position.
   * @param obstacle_idx Path index of the obstacle
   * @param free_idx Path index of the last free point before the obstacle, or 0 if none.
   * @param target_idx Path index of the forward-looking target.
   * @param prev_sign Previously chosen side (+1/-1/0).
   * @return The resolved target and side, or nullopt if no usable side was found.
   */
  std::optional<BypassResult> computeBypassTarget(
    const models::Path & path,
    float robot_x, float robot_y,
    size_t obstacle_idx, size_t free_idx, size_t target_idx,
    int prev_sign);

  /**
   * @brief Log a one-line bypass status at DEBUG level, once per transition.
   */
  void reportStatus(const std::string & status);

  /**
   * @brief Report the status, deactivate the bypass and clear the remembered side.
   */
  void deactivate(const std::string & status, bool clear_side = true);

  /**
   * @brief Publish a pose.
   */
  void publishPose(
    const nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr & pub,
    double x, double y, double yaw);

  /**
   * @brief Publish the robot->corridor reachability check segment for debugging
   * (green if clear, red if it hit a lethal cell).
   */
  void publishCheckLine(float x0, float y0, float x1, float y1, bool blocked);

  /**
   * @brief Delete the reachability check line marker.
   */
  void clearCheckLine();

  size_t target_offset_from_furthest_{0};
  size_t resume_offset_{0};
  float threshold_to_consider_{0};
  float min_distance_occupancy_check_{0};
  float max_path_occupancy_ratio_{0};
  float bypass_offset_dist_{0};
  // Last side chosen (+1 left, -1 right, 0 none). reset when the obstacle is passed.
  int last_bypass_sign_{0};
  unsigned int power_{0};
  float weight_{0};
  bool bypass_active_{false};

  bool visualize_furthest_point_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr furthest_point_pub_;

  bool visualize_occupancy_check_distance_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr occupancy_check_dist_pub_;

  bool visualize_target_point_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_point_pub_;

  bool visualize_check_line_{false};
  nav2::Publisher<visualization_msgs::msg::Marker>::SharedPtr check_line_pub_;
  bool check_line_shown_{false};

  std::string last_status_;
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLE_BYPASS_CRITIC_HPP_
