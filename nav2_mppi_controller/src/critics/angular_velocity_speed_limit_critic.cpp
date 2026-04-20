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
  getParam(power_, "cost_power", 2);
  getParam(punish_ackermann_constraints_, "punish_ackermann_constraints", false);
  getParam(min_turning_radius_, "min_turning_radius", 1.0f);
  getParam(ackermann_violation_cost_, "ackermann_violation_cost", 10000.0f);

  constexpr float kMinPositive = 1e-3f;
  if (max_angular_velocity_ <= 0.0f) {
    RCLCPP_WARN(
      logger_,
      "max_angular_velocity must be > 0, got %f — clamping to %f",
      max_angular_velocity_, kMinPositive);
    max_angular_velocity_ = kMinPositive;
  }
  if (min_turning_radius_ <= 0.0f) {
    RCLCPP_WARN(
      logger_,
      "min_turning_radius must be > 0, got %f — clamping to %f",
      min_turning_radius_, kMinPositive);
    min_turning_radius_ = kMinPositive;
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
    "max_speed=%f, min_speed=%f, min_turning_radius=%f, weight=%f, "
    "punish_ackermann_constraints=%s, ackermann_violation_cost=%f",
    min_angular_velocity_, max_angular_velocity_, max_speed_, min_speed_,
    min_turning_radius_, weight_,
    punish_ackermann_constraints_ ? "true" : "false", ackermann_violation_cost_);
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

      const auto wz_constrained = std::abs(vx) / min_turning_radius_;
      if (wz > wz_constrained) {
        wz_violation = ackermann_violation_cost_;
      }

      if (wz < min_angular_velocity_) {
        continue;
      }

      // Linearly interpolate allowed speed from max_speed at wz=0
      // to min_speed at wz>=max_angular_velocity.
      float wz_ratio = std::min(wz / max_angular_velocity_, 1.0f);
      float allowed_speed = (1.0f - wz_ratio) * max_speed_ + wz_ratio * min_speed_;

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
