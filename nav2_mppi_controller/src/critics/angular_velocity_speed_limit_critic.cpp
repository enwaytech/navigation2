#include "nav2_mppi_controller/critics/angular_velocity_speed_limit_critic.hpp"

namespace mppi::critics
{

void AngularVelocitySpeedLimitCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(min_angular_velocity_, "min_angular_velocity", 0.0f);
  getParam(max_angular_velocity_, "max_angular_velocity", 1.0f);
  getParam(max_speed_, "max_speed", 2.0f);
  getParam(min_speed_ratio_, "min_speed_ratio", 0.3f);
  getParam(weight_, "cost_weight", 10.0f);
  getParam(power_, "cost_power", 1);
  getParam(punish_ackermann_constraints_, "punish_ackermann_constraints", false);

  RCLCPP_INFO(
    logger_,
    "AngularVelocitySpeedLimitCritic instantiated with min_wz=%f, max_wz=%f, max_speed=%f, min_ratio=%f, weight=%f, punish_ackermann_constraints=%s",
    min_angular_velocity_, max_angular_velocity_, max_speed_, min_speed_ratio_, weight_, punish_ackermann_constraints_ ? "true" : "false");
}

void AngularVelocitySpeedLimitCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }
  auto & state = data.state;
  auto & costs = data.costs;

  const size_t batch_size = state.vx.rows();
  const size_t time_steps = state.vx.cols();

  for (size_t i = 0; i < batch_size; ++i) {
    float max_violation = 0.0f;
    float wz_violation = 0.0f;
    for (size_t t = 0; t < time_steps; ++t) {
      float vx = state.vx(i, t);
      float wz = std::abs(state.wz(i, t));

      const auto wz_constrained = std::abs(vx) / 1.0f;
      if (std::abs(wz) > std::abs(wz_constrained)) {
        wz_violation = 10000.0f;
      }

      if (wz < min_angular_velocity_) {
        continue;
      }

      // Calculate allowed speed based on angular velocity
      // Linear interpolation: max_speed when wz=0, to min_speed_ratio*max_speed when wz=max_angular_velocity
      float wz_ratio = std::min(wz / max_angular_velocity_, 1.0f);
      float allowed_speed = max_speed_ * (1.0f - wz_ratio * (1.0f - min_speed_ratio_));

      float speed_violation = std::abs(vx) - allowed_speed;
      if (speed_violation > 0.0f) {
        max_violation = std::max(max_violation, speed_violation);
      }
    }
    if (max_violation > 0.0f) {
      costs[i] += weight_ * std::pow(max_violation, power_);
    }
    if (wz_violation > 0.0f && punish_ackermann_constraints_) {
      costs[i] += wz_violation;
    }
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::AngularVelocitySpeedLimitCritic,
  mppi::critics::CriticFunction)
