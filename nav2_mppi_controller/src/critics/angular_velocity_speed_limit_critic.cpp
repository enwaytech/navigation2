// Copyright (c) 2026 Enway GmbH, Georg Flick
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

#include "nav2_mppi_controller/critics/angular_velocity_speed_limit_critic.hpp"

namespace mppi::critics
{

void AngularVelocitySpeedLimitCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(min_angular_velocity_, "min_angular_velocity", 0.0f);
  getParam(max_angular_velocity_, "max_angular_velocity", 1.0f);
  getParam(max_speed_, "max_speed", 2.0f);
  getParam(min_speed_, "min_speed", 0.6f);
  getParam(weight_, "cost_weight", 10.0f);
  getParam(power_, "cost_power", 1);

  constexpr float kMinPositive = 1e-3f;
  if (max_angular_velocity_ <= 0.0f) {
    RCLCPP_WARN(
      logger_,
      "max_angular_velocity must be > 0, got %f — clamping to %f",
      max_angular_velocity_, kMinPositive);
    max_angular_velocity_ = kMinPositive;
  }
  if (min_speed_ > max_speed_) {
    RCLCPP_WARN(
      logger_,
      "min_speed (%f) must be <= max_speed (%f) — clamping min_speed to max_speed",
      min_speed_, max_speed_);
    min_speed_ = max_speed_;
  }

  RCLCPP_INFO(
    logger_,
    "AngularVelocitySpeedLimitCritic instantiated with min_wz=%f, max_wz=%f, "
    "max_speed=%f, min_speed=%f, weight=%f",
    min_angular_velocity_, max_angular_velocity_, max_speed_, min_speed_, weight_);
}

void AngularVelocitySpeedLimitCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }
  auto & state = data.state;
  auto & costs = data.costs;

  const auto wz = state.wz.abs();
  const auto speed = (state.vx.square() + state.vy.square()).sqrt();
  const auto wz_ratio = (wz / max_angular_velocity_).min(1.0f);
  const auto allowed_speed = (1.0f - wz_ratio) * max_speed_ + wz_ratio * min_speed_;
  const auto speed_violation = (speed - allowed_speed).max(0.0f);
  const auto mask = (wz >= min_angular_velocity_).cast<float>();

  const auto per_traj = (mask * speed_violation * data.model_dt).rowwise().sum().eval();
  if (power_ > 1u) {
    costs += (per_traj * weight_).pow(power_).eval();
  } else {
    costs += (per_traj * weight_).eval();
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::AngularVelocitySpeedLimitCritic,
  mppi::critics::CriticFunction)
