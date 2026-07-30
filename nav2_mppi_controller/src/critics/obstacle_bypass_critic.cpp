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

#include <algorithm>
#include <cmath>
#include <string>

#include "nav2_mppi_controller/critics/obstacle_bypass_critic.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/line_iterator.hpp"

namespace mppi::critics
{

void ObstacleBypassCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 4.667f);
  getParam(min_distance_occupancy_check_, "min_distance_occupancy_check", 2.0f);
  getParam(max_path_occupancy_ratio_, "max_path_occupancy_ratio", 0.07f);
  getParam(target_offset_from_furthest_, "target_offset_from_furthest", 20);
  getParam(resume_offset_, "resume_offset", 20);
  getParam(threshold_to_consider_, "threshold_to_consider", 0.5f);
  getParam(bypass_offset_dist_, "bypass_offset_dist", 1.0f);

  getParam(visualize_furthest_point_, "visualize_furthest_point", false);
  getParam(visualize_occupancy_check_distance_, "visualize_occupancy_check_distance", false);
  getParam(visualize_target_point_, "visualize_target_point", false);
  getParam(visualize_check_line_, "visualize_check_line", false);

  if (auto node = parent_.lock()) {
    auto make_pose_pub =
      [&](bool enabled, const char * topic)
      -> nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr {
        if (!enabled) {return nullptr;}
        auto pub = node->create_publisher<geometry_msgs::msg::PoseStamped>(topic, 1);
        pub->on_activate();
        return pub;
      };
    furthest_point_pub_ =
      make_pose_pub(visualize_furthest_point_,
        "/critics/ObstacleBypassCritic/furthest_reached_path_point");
    occupancy_check_dist_pub_ =
      make_pose_pub(visualize_occupancy_check_distance_,
        "/critics/ObstacleBypassCritic/occupancy_check_end_point");
    target_point_pub_ =
      make_pose_pub(visualize_target_point_,
        "/critics/ObstacleBypassCritic/target_point");
    if (visualize_check_line_) {
      check_line_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(
          "/critics/ObstacleBypassCritic/reachability_check_line", 1);
      check_line_pub_->on_activate();
    }
  }

  RCLCPP_INFO(
    logger_, "ObstacleBypassCritic instantiated with %d power and %f weight", power_, weight_);
}

void ObstacleBypassCritic::reportStatus(const std::string & status)
{
  if (status != last_status_) {
    RCLCPP_DEBUG(logger_, "ObstacleBypassCritic: %s", status.c_str());
    last_status_ = status;
  }
}

void ObstacleBypassCritic::deactivate(const std::string & status, bool clear_side)
{
  reportStatus(status);
  clearCheckLine();
  bypass_active_ = false;
  if (clear_side) {
    last_bypass_sign_ = 0.0f;
  }
}

void ObstacleBypassCritic::publishPose(
  const nav2::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr & pub,
  double x, double y, double yaw)
{
  if (!pub || pub->get_subscription_count() == 0) {
    return;
  }
  auto msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
  msg->header.frame_id = costmap_ros_->getGlobalFrameID();
  msg->header.stamp = clock_->now();
  msg->pose.position.x = x;
  msg->pose.position.y = y;
  tf2::Quaternion quat;
  quat.setRPY(0.0, 0.0, yaw);
  msg->pose.orientation = tf2::toMsg(quat);
  pub->publish(std::move(msg));
}

void ObstacleBypassCritic::publishCheckLine(
  float x0, float y0, float x1, float y1, bool blocked)
{
  if (!check_line_pub_ || check_line_pub_->get_subscription_count() == 0) {
    return;
  }
  auto marker = std::make_unique<visualization_msgs::msg::Marker>();
  marker->header.frame_id = costmap_ros_->getGlobalFrameID();
  marker->header.stamp = clock_->now();
  marker->ns = "bypass_reachability_check";
  marker->id = 0;
  marker->type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker->action = visualization_msgs::msg::Marker::ADD;
  marker->scale.x = 0.03;
  marker->color.a = 1.0f;
  marker->color.r = blocked ? 1.0f : 0.0f;
  marker->color.g = blocked ? 0.0f : 1.0f;
  marker->color.b = blocked ? 0.0f : 1.0f;
  marker->pose.orientation.w = 1.0;
  geometry_msgs::msg::Point start, end;
  start.x = x0; start.y = y0;
  end.x = x1; end.y = y1;
  marker->points.push_back(start);
  marker->points.push_back(end);
  check_line_pub_->publish(std::move(marker));
  check_line_shown_ = true;
}

void ObstacleBypassCritic::clearCheckLine()
{
  if (!check_line_shown_) {
    return;
  }
  check_line_shown_ = false;
  if (!check_line_pub_ || check_line_pub_->get_subscription_count() == 0) {
    return;
  }
  auto marker = std::make_unique<visualization_msgs::msg::Marker>();
  marker->header.frame_id = costmap_ros_->getGlobalFrameID();
  marker->header.stamp = clock_->now();
  marker->ns = "bypass_reachability_check";
  marker->action = visualization_msgs::msg::Marker::DELETEALL;
  check_line_pub_->publish(std::move(marker));
}

std::optional<ObstacleBypassCritic::BypassResult> ObstacleBypassCritic::computeBypassTarget(
  const models::Path & path, float robot_x, float robot_y,
  size_t obstacle_idx, size_t last_free_idx, size_t target_idx, int prev_sign)
{
  const float resolution = static_cast<float>(costmap_->getResolution());
  const bool tracking_unknown = costmap_ros_->getLayeredCostmap()->isTrackingUnknown();
  const int max_steps = static_cast<int>(
    std::max(costmap_->getSizeInCellsX(), costmap_->getSizeInCellsY()));
  const size_t last_path_idx = static_cast<size_t>(path.x.size()) - 1;
  unsigned int mx, my;

  auto isNonLethal = [&](unsigned char c) {
      return c < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE &&
             (c != nav2_costmap_2d::NO_INFORMATION || tracking_unknown);
    };

  // Point on the path at idx plus its unit left-normal; false if the local tangent is degenerate.
  auto frameAt = [&](size_t idx, float & px, float & py, float & nx, float & ny) -> bool {
      const size_t next = std::min(idx + 1, last_path_idx);
      px = path.x(idx);
      py = path.y(idx);
      const float tx = path.x(next) - px;
      const float ty = path.y(next) - py;
      const float tangent_len = sqrtf(tx * tx + ty * ty);
      if (tangent_len < 1e-6f) {return false;}
      nx = -ty / tangent_len;
      ny = tx / tangent_len;
      return true;
    };

  float obs_x, obs_y, obs_nx, obs_ny;
  if (!frameAt(obstacle_idx, obs_x, obs_y, obs_nx, obs_ny)) {
    reportStatus("INACTIVE: degenerate path tangent at the obstacle");
    return std::nullopt;
  }
  float target_base_x, target_base_y, target_nx, target_ny;
  if (!frameAt(target_idx, target_base_x, target_base_y, target_nx, target_ny)) {
    reportStatus("INACTIVE: degenerate path tangent at the forward target");
    return std::nullopt;
  }
  // last_free_idx == 0 means the block starts at the first path point:
  // skip the reachability line check.
  float free_x = 0.0f, free_y = 0.0f, free_nx = 0.0f, free_ny = 0.0f;
  const bool check_reachability =
    last_free_idx > 0 && frameAt(last_free_idx, free_x, free_y, free_nx, free_ny);

  // Scan perpendicular to the path to find the first non-lethal cell on each side.
  auto scanSide = [&](float sign) -> int {
      for (int s = 1; s <= max_steps; ++s) {
        const float wx = obs_x + sign * s * resolution * obs_nx;
        const float wy = obs_y + sign * s * resolution * obs_ny;
        if (!costmap_->worldToMap(wx, wy, mx, my)) {
          return max_steps + 1;
        } else if (isNonLethal(costmap_->getCost(mx, my))) {
          return s;
        }
      }
      return max_steps + 1;
    };

  // A side is usable when:
  // - the target cell is in the map and non-lethal
  // - the target is reachable: the straight line from the robot to the offset point beside the last
  // free path point is clear of lethal cells
  auto isSideReachable = [&](float offset) -> bool {
      const float target_x = target_base_x + offset * target_nx;
      const float target_y = target_base_y + offset * target_ny;
      unsigned int target_mx, target_my;
      // check target cell is valid
      if (!costmap_->worldToMap(target_x, target_y, target_mx, target_my) ||
        !isNonLethal(costmap_->getCost(target_mx, target_my)))
      {
        return false;
      }

      if (!check_reachability) {
        clearCheckLine();
        return true;
      }

      unsigned int robot_mx, robot_my;
      if (!costmap_->worldToMap(robot_x, robot_y, robot_mx, robot_my)) {
        return true;  // Robot off the costmap: cannot run the line check.
      }
      // line endpoint: Offset point beside the last free path point, on the candidate side.
      const float end_x = free_x + offset * free_nx;
      const float end_y = free_y + offset * free_ny;
      unsigned int end_mx, end_my;
      // check line endpoint is valid
      if (!costmap_->worldToMap(end_x, end_y, end_mx,
        end_my) || !isNonLethal(costmap_->getCost(end_mx, end_my)))
      {
        publishCheckLine(robot_x, robot_y, end_x, end_y, true);
        return false;
      }

      float line_blocked_x = end_x, line_blocked_y = end_y;
      bool blocked = false;
      for (nav2_util::LineIterator line(robot_mx, robot_my, end_mx, end_my); line.isValid();
        line.advance())
      {
        if (!isNonLethal(costmap_->getCost(line.getX(), line.getY()))) {
          blocked = true;
          double line_blocked_x_d, line_blocked_y_d;
          costmap_->mapToWorld(line.getX(), line.getY(), line_blocked_x_d, line_blocked_y_d);
          line_blocked_x = static_cast<float>(line_blocked_x_d);
          line_blocked_y = static_cast<float>(line_blocked_y_d);
          break;
        }
      }
      publishCheckLine(robot_x, robot_y, line_blocked_x, line_blocked_y, blocked);
      return !blocked;
    };

  const int first_free_left = scanSide(1.0f);
  const int first_free_right = scanSide(-1.0f);
  if (first_free_left > max_steps && first_free_right > max_steps) {
    reportStatus("INACTIVE: no free space on either side of the obstacle");
    return std::nullopt;
  }

  // Side hysteresis: keep the previous side while it still has free space, otherwise take the
  // side whose free space is closer (ties to left). Signed offset: + left, - right.
  int sign;
  if (prev_sign > 0 && first_free_left <= max_steps) {
    sign = 1;
  } else if (prev_sign < 0 && first_free_right <= max_steps) {
    sign = -1;
  } else {
    sign = (first_free_left <= first_free_right) ? 1 : -1;
  }

  // Try the chosen side, then the other one.
  for (int attempt = 0; attempt < 2; ++attempt, sign = -sign) {
    const int first_free = (sign > 0) ? first_free_left : first_free_right;
    if (first_free > max_steps) {
      continue;
    }
    const float offset = sign * (first_free * resolution + bypass_offset_dist_);
    if (isSideReachable(offset)) {
      // Forward-looking target point offset from the path in the direction of the bypass
      return BypassResult{target_base_x + offset * target_nx, target_base_y + offset * target_ny,
        sign};
    }
  }

  reportStatus("INACTIVE: no reachable bypass side");
  return std::nullopt;
}

void ObstacleBypassCritic::score(CriticData & data)
{
  if (!enabled_ || data.state.local_path_length < threshold_to_consider_) {
    deactivate("Disabled");
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);
  const size_t furthest_reached_path_point = *data.furthest_reached_path_point;
  const size_t path_segments_count = data.path.x.size() - 1;

  if (furthest_reached_path_point > 0) {
    publishPose(
      furthest_point_pub_, data.path.x(furthest_reached_path_point),
      data.path.y(furthest_reached_path_point), data.path.yaws(furthest_reached_path_point));
  }

  // Furthest path index to check for occupancy: within min_distance_occupancy_check_ of the
  // start, or up to the furthest reached point.
  size_t occupancy_check_distance_idx = 0;
  float path_dist = 0.0f;
  for (unsigned int i = 1; i < path_segments_count; i++) {
    const float dx = data.path.x(i) - data.path.x(i - 1);
    const float dy = data.path.y(i) - data.path.y(i - 1);
    path_dist += sqrtf(dx * dx + dy * dy);
    if (path_dist <= min_distance_occupancy_check_ || i < furthest_reached_path_point) {
      occupancy_check_distance_idx = (i + 1 < path_segments_count) ? i + 1 : i;
    }
  }
  if (occupancy_check_distance_idx == 0) {
    deactivate("INACTIVE: no occupancy window ahead");
    return;
  }
  publishPose(
    occupancy_check_dist_pub_, data.path.x(occupancy_check_distance_idx),
    data.path.y(occupancy_check_distance_idx), data.path.yaws(occupancy_check_distance_idx));

  // Check if obstacles are blocking significant proportions of the local path
  // If path is blocked, incentivize turning in the shorter direction around the obstacle
  utils::setPathCostsIfNotSet(data, costmap_ros_);
  std::vector<bool> & path_pts_valid = *data.path_pts_valid;
  float invalid_ctr = 0.0f;
  for (size_t i = 0; i < occupancy_check_distance_idx; i++) {
    if (!path_pts_valid[i]) {invalid_ctr += 1.0f;}
  }
  const float occupancy_ratio = invalid_ctr / static_cast<float>(occupancy_check_distance_idx);
  const bool path_blocked = occupancy_ratio > max_path_occupancy_ratio_ && invalid_ctr > 2.0f;

  // Once bypass is active, require the ratio to drop well below the
  // threshold before deactivating to prevent oscillation
  if (!path_blocked) {
    if (!bypass_active_ || occupancy_ratio < max_path_occupancy_ratio_ * 0.5f) {
      deactivate("INACTIVE: path ahead is clear");
      return;
    }
  }

  // Find the first blocked path point
  size_t blocked_idx = 0;
  for (size_t j = 0; j < occupancy_check_distance_idx; j++) {
    if (!path_pts_valid[j]) {blocked_idx = j; break;}
  }

  // Find first valid path point past the blocked region
  size_t resume_idx = blocked_idx;
  for (; resume_idx < path_pts_valid.size(); resume_idx++) {
    if (path_pts_valid[resume_idx]) {break;}
  }

  // If blocked until the end of the path, don't activate bypass
  if (resume_idx >= path_pts_valid.size()) {
    deactivate("INACTIVE: path blocked to the end of the path");
    return;
  }

  // Don't activate bypass if the first valid path point past the blocked region is already near
  if (resume_idx < resume_offset_) {
    deactivate("INACTIVE: obstacle nearly passed");
    return;
  }

  // Midpoint of blocked region to score against. Note that the path is being continuously
  // pruned, so the blocked_idx is updated and adjusted forward as the robot moves
  const size_t obstacle_idx = (blocked_idx + resume_idx) / 2;

  // Last free path point before the obstacle for the reachability line check
  const size_t last_free_idx = (blocked_idx > 1) ? blocked_idx - 1 : 0;

  const size_t target_idx = std::min(
    furthest_reached_path_point + target_offset_from_furthest_, path_segments_count - 1);

  const geometry_msgs::msg::Pose & robot_pose = data.state.pose.pose;
  const auto result = computeBypassTarget(
    data.path, static_cast<float>(robot_pose.position.x), static_cast<float>(robot_pose.position.y),
    obstacle_idx, last_free_idx, target_idx, last_bypass_sign_);
  if (!result) {
    bypass_active_ = false;  // reason already reported by computeBypassTarget()
    last_bypass_sign_ = 0.0f;
    return;
  }

  // Forward-looking target point offset from the path in the direction of the bypass
  const float target_x = result->target_x;
  const float target_y = result->target_y;

  // Don't apply while the robot is moving away from the target
  // This keeps it off when reversing away from an obstacle
  {
    const auto & q = robot_pose.orientation;
    const float cyaw = 1.0f - 2.0f * static_cast<float>(q.y * q.y + q.z * q.z);
    const float syaw = 2.0f * static_cast<float>(q.x * q.y + q.w * q.z);
    const float vx = static_cast<float>(data.state.robot_speed.linear.x);
    const float vy = static_cast<float>(data.state.robot_speed.linear.y);
    const float world_vx = vx * cyaw - vy * syaw;
    const float world_vy = vx * syaw + vy * cyaw;
    const float to_tx = target_x - static_cast<float>(robot_pose.position.x);
    const float to_ty = target_y - static_cast<float>(robot_pose.position.y);
    constexpr float speed_deadband_sq = 0.05f * 0.05f;  // ignore odometry noise at standstill
    if ((world_vx * world_vx + world_vy * world_vy) > speed_deadband_sq &&
      (world_vx * to_tx + world_vy * to_ty) < 0.0f)
    {
      // keep last_bypass_sign_ to restore the side on resume
      deactivate("INACTIVE: moving away from the bypass target", false);
      return;
    }
  }

  // The critic is scored against the target to incentivize trajectories to steer around
  // the obstacle in the direction with the least disruption to path tracking.
  const int last_idx = data.trajectories.y.cols() - 1;
  const auto diff_x = target_x - data.trajectories.x.col(last_idx);
  const auto diff_y = target_y - data.trajectories.y.col(last_idx);
  if (power_ > 1u) {
    data.costs += (((diff_x.square() + diff_y.square()).sqrt()) * weight_).pow(power_);
  } else {
    data.costs += ((diff_x.square() + diff_y.square()).sqrt()) * weight_;
  }

  bypass_active_ = true;
  last_bypass_sign_ = result->sign;
  reportStatus(result->sign > 0.0f ? "ACTIVE: bypassing left" : "ACTIVE: bypassing right");
  publishPose(target_point_pub_, target_x, target_y, data.path.yaws(target_idx));
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::ObstacleBypassCritic,
  mppi::critics::CriticFunction)
