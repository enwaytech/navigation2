// Copyright (c) 2026 Enway GmbH, Adi Vardi
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

#include "nav2_mppi_controller/critics/cost_speed_limit_critic.hpp"

namespace mppi::critics
{

void
CostSpeedLimitCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(onset_distance_, "onset_distance", 1.0f);
  getParam(min_speed_, "min_speed", 0.6f);
  getParam(max_speed_, "max_speed", 2.0f);
  getParam(weight_, "cost_weight", 5.0f);
  getParam(power_, "cost_power", 1);
  // getParam(trajectory_point_step_, "trajectory_point_step", 2);
  getParam(inflation_layer_name_, "inflation_layer_name", std::string(""));

  constexpr float kMinPositive = 1e-3f;
  if (onset_distance_ <= 0.0f) {
    RCLCPP_WARN_STREAM(
      logger_,
      "onset_distance must be > 0, got " << onset_distance_ << " — clamping to " <<
        kMinPositive);
    onset_distance_ = kMinPositive;
  }
  if (min_speed_ <= 0.0f) {
    RCLCPP_WARN_STREAM(
      logger_,
      "min_speed must be > 0, got " << min_speed_ << " — clamping to " <<
        kMinPositive);
    min_speed_ = kMinPositive;
  }
  if (max_speed_ <= 0.0f) {
    RCLCPP_WARN_STREAM(
      logger_,
      "max_speed must be > 0, got " << max_speed_ << " — clamping to " <<
        kMinPositive);
    max_speed_ = kMinPositive;
  }
  if (min_speed_ > max_speed_) {
    RCLCPP_WARN_STREAM(
      logger_,
      "min_speed (" << min_speed_ << ") must be <= max_speed (" << max_speed_ <<
        ") — clamping min_speed to max_speed");
    min_speed_ = max_speed_;
  }

  collision_checker_.setCostmap(costmap_);

  inflation_layer_ = nav2_costmap_2d::InflationLayerInterface::getInflationLayer(
    costmap_ros_,
    inflation_layer_name_);

  if (inflation_layer_ == nullptr)
  {
    RCLCPP_WARN(logger_, "No inflation layer found in costmap configuration. CostSpeedLimitCritic is disabled!");
    return;
  }

  RCLCPP_INFO_STREAM(logger_,
                     "CostSpeedLimitCritic instantiated with"
                     " onset_distance=" << onset_distance_ <<
                     ", min_speed=" << min_speed_ <<
                     ", max_speed=" << max_speed_ <<
                     ", weight=" << weight_);
}

void
CostSpeedLimitCritic::score(CriticData& data)
{
  if (!enabled_ || inflation_layer_ == nullptr)
  {
    return;
  }

  // version 1: only apply at base_link
  const geometry_msgs::msg::Pose & robot_pose = data.state.pose.pose;
  const float rx = static_cast<float>(robot_pose.position.x);
  const float ry = static_cast<float>(robot_pose.position.y);

  const float pose_cost = costAtPose(rx, ry);
  if (pose_cost < 1.0f || pose_cost == static_cast<float>(nav2_costmap_2d::NO_INFORMATION)) {
    return;  // In free space
  }

  const float dist_to_obj = distanceToObstacle(pose_cost);
  if (dist_to_obj > onset_distance_)
  {
    return;
  }

  const float dist_ratio = std::clamp(dist_to_obj / onset_distance_, 0.f, 1.f);
  const float allowed_vx = min_speed_ + dist_ratio * (max_speed_ - min_speed_);

  const auto violation = (data.state.vx.abs() - allowed_vx).max(0.0f);
  const auto per_traj = (violation * data.model_dt).rowwise().sum().eval();

  if (power_ > 1u) {
    data.costs += (per_traj * weight_).pow(power_).eval();
  } else {
    data.costs += (per_traj * weight_).eval();
  }

  /*
  // version 2: penalize over strided trajectories

  int strided_traj_cols = floor((data.trajectories.x.cols() - 1) / trajectory_point_step_) + 1;
  int strided_traj_rows = data.trajectories.x.rows();
  int outer_stride = strided_traj_rows * trajectory_point_step_;

  const auto traj_x = Eigen::Map<const Eigen::ArrayXXf, 0, Eigen::Stride<-1, -1>>(
      data.trajectories.x.data(), strided_traj_rows, strided_traj_cols, Eigen::Stride<-1, -1>(outer_stride, 1));
  const auto traj_y = Eigen::Map<const Eigen::ArrayXXf, 0, Eigen::Stride<-1, -1>>(
      data.trajectories.y.data(), strided_traj_rows, strided_traj_cols, Eigen::Stride<-1, -1>(outer_stride, 1));
  // const auto traj_yaw = Eigen::Map<const Eigen::ArrayXXf, 0,
  //     Eigen::Stride<-1, -1>>(
  //   data.trajectories.yaws.data(), strided_traj_rows, strided_traj_cols,
  //   Eigen::Stride<-1, -1>(outer_stride, 1));

  for (int i = 0; i < strided_traj_rows; ++i)
  {
    bool trajectory_collide = false;
    float pose_cost = 0.0f;

    for (int j = 0; j < strided_traj_cols; j++)
    {
      const float pose_cost = costAtPose(traj.x(i, j), traj.y(i, j));
      if (pose_cost < 1.0f) {
        continue;  // In free space
      }

      // TODO check obstacleCritic to see how they do it and what other setup we need
      const float dist_to_obj = distanceToObstacle(pose_cost);
      // TODO add inscribed radius or footprint side offset, to be the same as CP.
      // OR, already add the signed offset to the onset_dist in the params
      if (dist_to_obj > onset_distance_)
      {
        continue;
      }
      const auto allowed_speed = min_speed_ + dist_to_obj * (max_speed_ - min_speed_);

      // TODO penalize if state.vx at point is > allowed speed
      // TODO find a nicer way to do it with Eigen

      if (pose_cost > onset_dis_cost)
      {
        // TODO compute allowed speed for the cost
        // TODO actually cost is not linear, so should reverse engineer the formula in
        // const auto allowed_speed = min_speed_ +
        // (1.0f - wz_ratio) * max_speed_ + wz_ratio * min_speed_;

        // TODO here we penalize speed
        // TODO should I penalize per point over speed or for the
      }
  }
      */

}

float
CostSpeedLimitCritic::costAtPose(float x, float y)
{
  unsigned int x_i, y_i;
  if (!collision_checker_.worldToMap(x, y, x_i, y_i)) {
    return static_cast<float>(nav2_costmap_2d::NO_INFORMATION);
  } else {
    return static_cast<float>(collision_checker_.pointCost(x_i, y_i));
  }
}

float
CostSpeedLimitCritic::distanceToObstacle(const float cost)
{
  const float scale_factor = static_cast<float>(inflation_layer_->getCostScalingFactor());
  const float inscribed_radius = costmap_ros_->getLayeredCostmap()->getInscribedRadius();
  constexpr float constant_log = log(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 1);
  float dist_to_obj = inscribed_radius - (log(cost) - constant_log) / scale_factor;

  // TODO divide by resolution?
  // const double resolution = costmap->getCostmap()->getResolution();

  // the distance is computed at base_link cost so
  // substract the inscribed radius to get the closest distance to the object
  dist_to_obj -= inscribed_radius;

  return dist_to_obj;
}

} // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mppi::critics::CostSpeedLimitCritic, mppi::critics::CriticFunction)
