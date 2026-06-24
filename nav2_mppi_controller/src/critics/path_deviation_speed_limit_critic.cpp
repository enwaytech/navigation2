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

#include "nav2_mppi_controller/critics/path_deviation_speed_limit_critic.hpp"

namespace mppi::critics
{

void PathDeviationSpeedLimitCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(min_speed_, "min_speed", 0.6f);
  getParam(max_speed_, "max_speed", 2.0f);
  getParam(min_deviation_, "min_deviation", 0.5f);
  getParam(max_deviation_, "max_deviation", 1.5f);
  getParam(weight_, "cost_weight", 5.0f);
  getParam(power_, "cost_power", 1);

  constexpr float kMinPositive = 1e-3f;
  if (max_deviation_ <= 0.0f) {
    RCLCPP_WARN_STREAM(
      logger_,
      "max_deviation must be > 0, got " << max_deviation_ << " — clamping to " <<
        kMinPositive);
    max_deviation_ = kMinPositive;
  }
  if (min_speed_ > max_speed_) {
    RCLCPP_WARN_STREAM(
      logger_,
      "min_speed (" << min_speed_ << ") must be <= max_speed (" << max_speed_ <<
        ") — clamping min_speed to max_speed");
    min_speed_ = max_speed_;
  }

  if (min_deviation_ > max_deviation_) {
    RCLCPP_WARN_STREAM(
      logger_,
      "min_deviation (" << min_deviation_ << ") must be <= max_deviation (" << max_deviation_ <<
        ") — clamping min_deviation to max_deviation");
    min_deviation_ = max_deviation_;
  }

  RCLCPP_INFO_STREAM(
    logger_,
    "PathDeviationSpeedLimitCritic instantiated with"
      " min_deviation=" << min_deviation_ <<
      ", max_deviation=" << max_deviation_ <<
      ", max_speed=" << max_speed_ <<
      ", min_speed=" << min_speed_ <<
      ", weight=" << weight_);
}

void PathDeviationSpeedLimitCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);
  const size_t path_end = *data.furthest_reached_path_point + 1;

  // TODO
  const geometry_msgs::msg::Pose & robot_pose = data.state.pose.pose;
  const float rx = static_cast<float>(robot_pose.position.x);
  const float ry = static_cast<float>(robot_pose.position.y);

  float min_dist_sq = std::numeric_limits<float>::max();
  for (size_t p = 0; p < path_end; ++p) {
    const float dx = rx - data.path.x(p);
    const float dy = ry - data.path.y(p);
    min_dist_sq = std::min(min_dist_sq, dx * dx + dy * dy);
  }
  const float min_dist = std::sqrt(min_dist_sq);

  // no penalty below min_deviation_
  if (min_dist < min_deviation_) {
    return;
  }

  const float dev_ratio =
    std::clamp((min_dist - min_deviation_) / (max_deviation_ - min_deviation_), 0.f, 1.f);
  const float allowed_vx = max_speed_ - dev_ratio * (max_speed_ - min_speed_);

  const auto violation = (data.state.vx - allowed_vx).max(0.0f);
  const auto per_traj = (violation * data.model_dt).rowwise().sum().eval();

  if (power_ > 1u) {
    data.costs += (per_traj * weight_).pow(power_).eval();
  } else {
    data.costs += (per_traj * weight_).eval();
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::PathDeviationSpeedLimitCritic,
  mppi::critics::CriticFunction)
