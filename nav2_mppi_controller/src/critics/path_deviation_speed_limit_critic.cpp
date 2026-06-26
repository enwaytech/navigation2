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
  getParam(min_deviation_, "min_deviation", 0.5f);
  getParam(weight_, "cost_weight", 5.0f);
  getParam(power_, "cost_power", 1);

  constexpr float kMinPositive = 1e-3f;
  if (min_deviation_ <= 0.0f) {
    RCLCPP_WARN_STREAM(
      logger_,
      "min_deviation must be > 0, got " << min_deviation_ << " — clamping to " <<
        kMinPositive);
    min_deviation_ = kMinPositive;
  }

  RCLCPP_INFO_STREAM(
    logger_,
    "PathDeviationSpeedLimitCritic instantiated with"
      " min_deviation=" << min_deviation_ <<
      ", min_speed=" << min_speed_ <<
      ", weight=" << weight_);
}

void PathDeviationSpeedLimitCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  const geometry_msgs::msg::Pose & robot_pose = data.state.pose.pose;
  const float rx = static_cast<float>(robot_pose.position.x);
  const float ry = static_cast<float>(robot_pose.position.y);

  // Path is pruned each iteration — the nearest path point is always index 0.
  const float ref_dx = rx - data.path.x(0);
  const float ref_dy = ry - data.path.y(0);

  // Compute path tangent from consecutive points; fall back to stored yaw if degenerate.
  float ux, uy;
  if (data.path.x.size() >= 2) {
    const float tx = data.path.x(1) - data.path.x(0);
    const float ty = data.path.y(1) - data.path.y(0);
    const float tlen = std::sqrt(tx * tx + ty * ty);
    if (tlen >= 1e-6f) {
      ux = tx / tlen;
      uy = ty / tlen;
    } else {
      ux = std::cos(data.path.yaws(0));
      uy = std::sin(data.path.yaws(0));
    }
  } else {
    ux = std::cos(data.path.yaws(0));
    uy = std::sin(data.path.yaws(0));
  }

  // Cross-track error: d · n, where n = (-uy, ux) is the path normal.
  const float cross_track_err = std::abs(-ref_dx * uy + ref_dy * ux);

  // no penalty below min_deviation_
  if (cross_track_err < min_deviation_) {
    return;
  }

  const auto violation = (data.state.vx.abs() - min_speed_).max(0.0f);
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
