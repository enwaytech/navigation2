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

#include <cmath>

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
    RCLCPP_WARN_STREAM(
      logger_,
      "max_angular_velocity must be > 0, got " << max_angular_velocity_ << " — clamping to " <<
        kMinPositive);
    max_angular_velocity_ = kMinPositive;
  }
  if (min_speed_ > max_speed_) {
    RCLCPP_WARN_STREAM(
      logger_,
      "min_speed (" << min_speed_ << ") must be <= max_speed (" << max_speed_ <<
        ") — clamping min_speed to max_speed");
    min_speed_ = max_speed_;
  }

  RCLCPP_INFO_STREAM(
    logger_,
    "AngularVelocitySpeedLimitCritic instantiated with min_wz=" << min_angular_velocity_
                                                                << ", max_wz=" <<
      max_angular_velocity_
                                                                << ", max_speed=" << max_speed_
                                                                << ", min_speed=" << min_speed_
                                                                << ", weight=" << weight_);
}

void AngularVelocitySpeedLimitCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }
  auto & state = data.state;
  auto & costs = data.costs;
  const Eigen::Index n_rows = state.vx.rows();
  const Eigen::Index n_cols = state.vx.cols();

  const float model_dt = data.model_dt;
  const float min_wz = min_angular_velocity_;
  const float max_wz = max_angular_velocity_;
  const float min_speed = min_speed_;
  const float max_speed = max_speed_;

  // Column by column, so that every pass runs over contiguous memory and vectorizes
  per_traj_.setZero(n_rows);
  speed_.resize(n_rows);
  mask_.resize(n_rows);
  step_cost_.resize(n_rows);
  float * speed = speed_.data();
  float * mask = mask_.data();
  for (Eigen::Index c = 0; c < n_cols; ++c) {
    const float * vx = state.vx.col(c).data();
    const float * vy = state.vy.col(c).data();
    const float * wz = state.wz.col(c).data();
    // Plain loop: Eigen's vectorized sqrt is approximate on x86
    for (Eigen::Index i = 0; i < n_rows; ++i) {
      speed[i] = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
      // No penalty below min_angular_velocity
      mask[i] = std::abs(wz[i]) >= min_wz ? 1.0f : 0.0f;
    }
    const auto wz_ratio = (state.wz.col(c).abs() / max_wz).min(1.0f);
    const auto allowed_speed = (1.0f - wz_ratio) * max_speed + wz_ratio * min_speed;
    step_cost_ = mask_ * (speed_ - allowed_speed).max(0.0f) * model_dt;
    per_traj_ += step_cost_;
  }

  // Scaled in place, so that the product is rounded before it is added to the costs
  per_traj_ *= weight_;
  if (power_ > 1u) {
    costs += per_traj_.pow(power_);
  } else {
    costs += per_traj_;
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::AngularVelocitySpeedLimitCritic,
  mppi::critics::CriticFunction)
