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

#include <algorithm>
#include <cmath>

#include "nav2_mppi_controller/critics/path_align_critic.hpp"

namespace mppi::critics
{

void PathAlignCritic::initialize()
{
  auto getParentParam = parameters_handler_->getParamGetter(parent_name_);
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 10.0f);
  getParam(min_distance_occupancy_check_, "min_distance_occupancy_check", 2.0f);
  getParam(max_path_occupancy_ratio_, "max_path_occupancy_ratio", 0.07f);
  getParam(offset_from_furthest_, "offset_from_furthest", 20);
  getParam(trajectory_point_step_, "trajectory_point_step", 4);
  getParam(
    threshold_to_consider_,
    "threshold_to_consider", 0.5f);
  getParam(use_path_orientations_, "use_path_orientations", false);
  getParam(visualize_furthest_point_, "visualize_furthest_point", false);

  if (visualize_furthest_point_) {
    auto node = parent_.lock();
    if (node) {
      furthest_point_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "/critics/PathAlignCritic/furthest_reached_path_point", 1);
      furthest_point_pub_->on_activate();
    }
  }

  getParam(visualize_occupancy_check_distance_, "visualize_occupancy_check_distance", false);

  if (visualize_occupancy_check_distance_) {
    auto node = parent_.lock();
    if (node) {
      occupancy_check_dist_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "/critics/PathAlignCritic/occupancy_check_end_point", 1);
      occupancy_check_dist_pub_->on_activate();
    }
  }

  RCLCPP_INFO(
    logger_,
    "ReferenceTrajectoryCritic instantiated with %d power and %f weight",
    power_, weight_);
}

void PathAlignCritic::score(CriticData & data)
{
  if (!enabled_ || data.state.local_path_length < threshold_to_consider_) {
    return;
  }

  // Don't apply when first getting bearing w.r.t. the path
  utils::setPathFurthestPointIfNotSet(data);
  const size_t furthest_reached_path_point = *data.furthest_reached_path_point;

  const auto now = clock_->now();
  // Visualize furthest reached pose if enabled
  if (visualize_furthest_point_ && furthest_reached_path_point > 0 &&
    furthest_point_pub_->get_subscription_count() > 0)
  {
    auto furthest_point = std::make_unique<geometry_msgs::msg::PoseStamped>();
    furthest_point->header.frame_id = costmap_ros_->getGlobalFrameID();
    furthest_point->header.stamp = now;
    furthest_point->pose.position.x = data.path.x(furthest_reached_path_point);
    furthest_point->pose.position.y = data.path.y(furthest_reached_path_point);
    furthest_point->pose.position.z = 0.0;
    tf2::Quaternion quat;
    quat.setRPY(0.0, 0.0, data.path.yaws(furthest_reached_path_point));
    furthest_point->pose.orientation = tf2::toMsg(quat);
    furthest_point_pub_->publish(std::move(furthest_point));
  }

  if (furthest_reached_path_point < offset_from_furthest_) {
    return;
  }

  // Find integrated distance in the path and find the first path IDX further than
  //  max(min_distance_occupancy_check_, furthest_reached_path_point)
  const size_t path_segments_count = data.path.x.size() - 1;
  size_t occupancy_check_distance_idx = 0;
  std::vector<float> & path_integrated_distances = path_integrated_distances_;
  std::vector<utils::Pose2D> & path = path_;
  path_integrated_distances.assign(path_segments_count, 0.0f);
  path.resize(path_segments_count);
  float dx = 0.0f, dy = 0.0f;
  for (unsigned int i = 1; i != path_segments_count; i++) {
    auto & pose = path[i - 1];
    pose.x = data.path.x(i - 1);
    pose.y = data.path.y(i - 1);
    pose.theta = data.path.yaws(i - 1);

    dx = data.path.x(i) - pose.x;
    dy = data.path.y(i) - pose.y;
    path_integrated_distances[i] = path_integrated_distances[i - 1] + sqrtf(dx * dx + dy * dy);

    if (path_integrated_distances[i] <= min_distance_occupancy_check_ ||
      i < furthest_reached_path_point)
    {
      occupancy_check_distance_idx = (i + 1 < path_segments_count) ? i + 1 : i;
    }
  }

  // Visualize occupancy check distance if enabled
  if (visualize_occupancy_check_distance_ &&
    occupancy_check_dist_pub_->get_subscription_count() > 0)
  {
    auto occupancy_check_point = std::make_unique<geometry_msgs::msg::PoseStamped>();
    occupancy_check_point->header.frame_id = costmap_ros_->getGlobalFrameID();
    occupancy_check_point->header.stamp = now;
    occupancy_check_point->pose.position.x = data.path.x(occupancy_check_distance_idx);
    occupancy_check_point->pose.position.y = data.path.y(occupancy_check_distance_idx);
    occupancy_check_point->pose.position.z = 0.0;
    tf2::Quaternion quat;
    quat.setRPY(0.0, 0.0, data.path.yaws(occupancy_check_distance_idx));
    occupancy_check_point->pose.orientation = tf2::toMsg(quat);
    occupancy_check_dist_pub_->publish(std::move(occupancy_check_point));
  }

  // Don't apply when dynamic obstacles are blocking significant proportions of the local path
  const float occupancy_check_distance_idx_flt = static_cast<float>(occupancy_check_distance_idx);
  utils::setPathCostsIfNotSet(data, costmap_ros_);
  const std::vector<bool> & path_pts_valid_bits = *data.path_pts_valid;
  float invalid_ctr = 0.0f;
  for (size_t i = 0; i < occupancy_check_distance_idx; i++) {
    if (!path_pts_valid_bits[i]) {invalid_ctr += 1.0f;}
    if (invalid_ctr / occupancy_check_distance_idx_flt > max_path_occupancy_ratio_ &&
      invalid_ctr > 2.0f)
    {
      // Path is blocked, defer to ObstacleBypassCritic if desired to go around dynamically
      return;
    }
  }

  // Byte copy of the validity flags, cheaper to look up per sample than std::vector<bool>
  path_pts_valid_.assign(path_pts_valid_bits.begin(), path_pts_valid_bits.end());
  const std::vector<uint8_t> & path_pts_valid = path_pts_valid_;

  // Finish populating the path vector
  auto & final_pose = path[path_segments_count - 1];
  final_pose.x = data.path.x(path_segments_count - 1);
  final_pose.y = data.path.y(path_segments_count - 1);
  final_pose.theta = data.path.yaws(path_segments_count - 1);

  // Strided trajectory points (every trajectory_point_step_ time step). The trajectories are
  // column-major, so point (t, p) is at t + p * outer_stride.
  const Eigen::Index batch_size = data.trajectories.x.rows();
  const Eigen::Index strided_traj_cols =
    (data.trajectories.x.cols() - 1) / trajectory_point_step_ + 1;
  const Eigen::Index outer_stride = batch_size * trajectory_point_step_;
  const float * const traj_x = data.trajectories.x.data();
  const float * const traj_y = data.trajectories.y.data();
  const float * const traj_yaws = data.trajectories.yaws.data();

  Eigen::ArrayXf & cost = cost_;
  cost.setZero(data.costs.rows());

  const utils::Pose2D * const path_data = path.data();
  const uint8_t * const path_pts_valid_data = path_pts_valid.data();
  const bool use_path_orientations = use_path_orientations_;

  // Trajectories are processed in blocks small enough for their integrated distances to stay
  // in cache. Column p of traj_integrated_distances_ holds the distance up to strided point p.
  constexpr Eigen::Index block_size = 128;
  traj_integrated_distances_.resize(block_size, strided_traj_cols);
  traj_integrated_distances_.col(0).setZero();
  for (Eigen::Index t0 = 0; t0 < batch_size; t0 += block_size) {
    const Eigen::Index block_rows = std::min(block_size, batch_size - t0);

    // Column by column, so that this loop vectorizes
    for (Eigen::Index p = 1; p < strided_traj_cols; p++) {
      const float * x0 = traj_x + (p - 1) * outer_stride + t0;
      const float * y0 = traj_y + (p - 1) * outer_stride + t0;
      const float * x1 = x0 + outer_stride;
      const float * y1 = y0 + outer_stride;
      const float * dist0 = &traj_integrated_distances_(0, p - 1);
      float * dist1 = &traj_integrated_distances_(0, p);
      for (Eigen::Index b = 0; b < block_rows; b++) {
        const float seg_dx = x1[b] - x0[b];
        const float seg_dy = y1[b] - y0[b];
        dist1[b] = dist0[b] + std::sqrt(seg_dx * seg_dx + seg_dy * seg_dy);
      }
    }

    for (Eigen::Index b = 0; b < block_rows; b++) {
      const Eigen::Index t = t0 + b;
      float summed_path_dist = 0.0f;
      unsigned int num_samples = 0u;
      unsigned int path_pt = 0u;
      // At each (strided) traj point, find the path point whose integrated distance along
      // the path is closest to the trajectory point's integrated distance along the trajectory.
      // if that path point is not in collision, compute the Euclidean distance between the
      // matching path pt & traj pt the total cost is the average of those distances across
      // the trajectory
      for (Eigen::Index p = 1; p < strided_traj_cols; p++) {
        path_pt = utils::findClosestPathPt(
          path_integrated_distances, traj_integrated_distances_(b, p), path_pt);

        // The nearest path point to align to needs to be not in collision, else
        // let the obstacle critic take over in this region due to dynamic obstacles
        if (path_pts_valid_data[path_pt]) {
          const auto & pose = path_data[path_pt];
          const Eigen::Index idx = t + p * outer_stride;
          dx = pose.x - traj_x[idx];
          dy = pose.y - traj_y[idx];
          num_samples++;
          if (use_path_orientations) {
            const float dyaw = utils::shortestAngularDistance(pose.theta, traj_yaws[idx]);
            summed_path_dist += sqrtf(dx * dx + dy * dy + dyaw * dyaw);
          } else {
            summed_path_dist += sqrtf(dx * dx + dy * dy);
          }
        }
      }
      if (num_samples > 0u) {
        cost(t) = summed_path_dist / static_cast<float>(num_samples);
      } else {
        cost(t) = 0.0f;
      }
    }
  }

  // Scaled in place, so that the product is rounded before it is added to the costs
  cost *= weight_;
  if (power_ > 1u) {
    data.costs += cost.pow(power_);
  } else {
    data.costs += cost;
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::PathAlignCritic,
  mppi::critics::CriticFunction)
