#include "nav2_mppi_controller/critics/angular_velocity_speed_limit_critic.hpp"

namespace mppi::critics
{

void AngularVelocitySpeedLimitCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(angular_velocity_threshold_, "angular_velocity_threshold", 0.5f);
  getParam(max_speed_at_threshold_, "max_speed_at_threshold", 1.0f);
  getParam(weight_, "cost_weight", 10.0f);
  getParam(power_, "cost_power", 2);

  RCLCPP_INFO(
    logger_,
    "AngularVelocitySpeedLimitCritic instantiated with wz_threshold=%f, max_speed=%f, weight=%f",
    angular_velocity_threshold_, max_speed_at_threshold_, weight_);
}

void AngularVelocitySpeedLimitCritic::score(CriticData & data)
{
  auto & state = data.state;
  auto & costs = data.costs;

  const size_t batch_size = state.vx.rows();
  const size_t time_steps = state.vx.cols();

  for (size_t i = 0; i < batch_size; ++i) {
    float max_violation = 0.0f;

    for (size_t t = 0; t < time_steps; ++t) {
      float vx = state.vx(i, t);
      float wz = std::abs(state.wz(i, t));

      // If angular velocity exceeds threshold, check speed limit
      if (wz > angular_velocity_threshold_) {
        float speed_violation = std::abs(vx) - max_speed_at_threshold_;
        if (speed_violation > 0.0f) {
          max_violation = std::max(max_violation, speed_violation);
        }
      }
    }

    if (max_violation > 0.0f) {
      costs[i] += weight_ * std::pow(max_violation, power_);
    }
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::AngularVelocitySpeedLimitCritic,
  mppi::critics::CriticFunction)
