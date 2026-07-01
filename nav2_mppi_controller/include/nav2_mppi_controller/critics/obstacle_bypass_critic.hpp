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

#include <string>

#include "nav2_mppi_controller/critic_function.hpp"
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
  /**
   * @brief Determine the best side and distance to bypass an obstacle using costmap
   * @param path_x X position on path at obstacle
   * @param path_y Y position on path at obstacle
   * @param path_yaw Yaw of path tangent at obstacle
   * @param robot_x Robot X position (start of the reachability line check)
   * @param robot_y Robot Y position (start of the reachability line check)
   * @param free_x X of the last free path point before the obstacle
   * @param free_y Y of the last free path point before the obstacle
   * @param free_perp_x X of the unit (left) perpendicular at the last free point
   * @param free_perp_y Y of the unit (left) perpendicular at the last free point
   * @param target_base_x X of the forward-looking target base point on the path
   * @param target_base_y Y of the forward-looking target base point on the path
   * @param target_perp_x X of the unit perpendicular at the target base point
   * @param target_perp_y Y of the unit perpendicular at the target base point
   * @param check_reachability When false, skip the reachability line check
   *   (used when there is no valid free-point anchor before the obstacle)
   * @param prev_sign Previously chosen side (+1 left, -1 right, 0 none). The
   *   chosen side stays on prev_sign as long as that side still has free space,
   *   to avoid flip-flopping (hysteresis); otherwise the closer side is used.
   * @param[out] signed_offset Signed offset distance (+ left, - right)
   * @return true if a valid bypass target was found
   */
  bool determineBestBypassSide(
    float path_x, float path_y, float path_yaw,
    float robot_x, float robot_y,
    float free_x, float free_y, float free_perp_x, float free_perp_y,
    float target_base_x, float target_base_y,
    float target_perp_x, float target_perp_y,
    bool check_reachability,
    float prev_sign,
    float & signed_offset);

  /**
   * @brief Log a one-line bypass status, but only when it differs from the last
   * logged one, so the message is readable and does not spam at control rate.
   * @param status Human-readable status / reason string
   */
  void reportStatus(const std::string & status);

  /**
   * @brief Publish the lateral reachability check segment for debugging.
   * @param x0,y0 Segment start (last free path point)
   * @param x1,y1 Segment end (offset endpoint, or the blocking cell if blocked)
   * @param blocked Whether the sweep hit a lethal cell (colors the line red)
   * @param id Marker id (distinguishes the preferred vs alternate side)
   */
  void publishCheckLine(float x0, float y0, float x1, float y1, bool blocked, int id);

  size_t target_offset_from_furthest_{0};
  float threshold_to_consider_{0};
  float min_distance_occupancy_check_{0};
  float max_path_occupancy_ratio_{0};
  float bypass_offset_dist_{0};
  // Side chosen on the last active cycle (+1 left, -1 right, 0 none); kept across
  // the transient "not enough path traversed" gate to stabilize the side, reset
  // when the obstacle is resolved/passed.
  float last_bypass_sign_{0.0f};
  unsigned int power_{0};
  float weight_{0};
  bool bypass_active_{false};

  bool visualize_furthest_point_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr furthest_point_pub_;

  bool visualize_occupancy_check_distance_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr occupancy_check_dist_pub_;

  bool visualize_target_point_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_point_pub_;

  bool visualize_blocked_point_{false};
  nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr blocked_point_pub_;

  bool visualize_check_line_{false};
  nav2::Publisher<visualization_msgs::msg::Marker>::SharedPtr check_line_pub_;

  // Last status logged by reportStatus(); used to suppress repeated messages.
  std::string last_status_;
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLE_BYPASS_CRITIC_HPP_
