// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2025 Open Navigation LLC
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

#ifndef NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_
#define NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <algorithm>

#include "nav2_mppi_controller/models/control_sequence.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/constraints.hpp"

#include "nav2_mppi_controller/tools/parameters_handler.hpp"

namespace mppi
{

// Forward declaration of utils method, since utils.hpp can't be included here due
// to recursive inclusion.
namespace utils
{
float clamp(const float lower_bound, const float upper_bound, const float input);
}

/**
 * @class mppi::MotionModel
 * @brief Abstract motion model for modeling a vehicle
 */
class MotionModel
{
public:
  /**
    * @brief Constructor for mppi::MotionModel
    */
  MotionModel() = default;

  /**
    * @brief Destructor for mppi::MotionModel
    */
  virtual ~MotionModel() = default;

  /**
    * @brief Initialize motion model on bringup and set required variables
    * @param control_constraints Constraints on control
    * @param model_dt duration of a time step
    */
  void initialize(const models::ControlConstraints & control_constraints, float model_dt)
  {
    control_constraints_ = control_constraints;
    model_dt_ = model_dt;
  }

  /**
   * @brief With input velocities, find the vehicle's output velocities
   * @param state Contains control velocities to use to populate vehicle velocities
   */
  virtual void predict(models::State & state)
  {
    const bool is_holo = isHolonomic();
    float max_delta_vx = model_dt_ * control_constraints_.ax_max;
    float min_delta_vx = model_dt_ * control_constraints_.ax_min;
    float max_delta_vy = model_dt_ * control_constraints_.ay_max;
    float min_delta_vy = model_dt_ * control_constraints_.ay_min;
    float max_delta_wz = model_dt_ * control_constraints_.az_max;

    unsigned int n_cols = state.vx.cols();

    for (unsigned int i = 1; i < n_cols; i++) {
      auto lower_bound_vx = (state.vx.col(i - 1) >
        0).select(state.vx.col(i - 1) + min_delta_vx,
        state.vx.col(i - 1) - max_delta_vx);
      auto upper_bound_vx = (state.vx.col(i - 1) >
        0).select(state.vx.col(i - 1) + max_delta_vx,
        state.vx.col(i - 1) - min_delta_vx);

      state.cvx.col(i - 1) = state.cvx.col(i - 1)
        .cwiseMax(lower_bound_vx)
        .cwiseMin(upper_bound_vx);
      state.vx.col(i) = state.cvx.col(i - 1);

      state.cwz.col(i - 1) = state.cwz.col(i - 1)
        .cwiseMax(state.wz.col(i - 1) - max_delta_wz)
        .cwiseMin(state.wz.col(i - 1) + max_delta_wz);
      state.wz.col(i) = state.cwz.col(i - 1);

      if (is_holo) {
        auto lower_bound_vy = (state.vy.col(i - 1) >
          0).select(state.vy.col(i - 1) + min_delta_vy,
          state.vy.col(i - 1) - max_delta_vy);
        auto upper_bound_vy = (state.vy.col(i - 1) >
          0).select(state.vy.col(i - 1) + max_delta_vy,
          state.vy.col(i - 1) - min_delta_vy);

        state.cvy.col(i - 1) = state.cvy.col(i - 1)
          .cwiseMax(lower_bound_vy)
          .cwiseMin(upper_bound_vy);
        state.vy.col(i) = state.cvy.col(i - 1);
      }
    }
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  virtual bool isHolonomic() = 0;

  /**
   * @brief Apply hard vehicle constraints to a control sequence
   * @param control_sequence Control sequence to apply constraints to
   */
  virtual void applyConstraints(models::ControlSequence & /*control_sequence*/) {}

protected:
  float model_dt_{0.0};
  models::ControlConstraints control_constraints_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f};
};

/**
 * @class mppi::AckermannMotionModel
 * @brief Ackermann motion model
 */
class AckermannMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::AckermannMotionModel
    */
  explicit AckermannMotionModel(ParametersHandler * param_handler, const std::string & name)
  {
    auto getParam = param_handler->getParamGetter(name + ".AckermannConstraints");
    getParam(min_turning_r_, "min_turning_r", 0.2);
    getParam(max_steering_angle_, "max_steering_angle", 0.52);  // ~30 degrees default
    getParam(max_steering_rate_, "max_steering_rate", 1.57);    // ~90 deg/s default
    getParam(enable_steering_dynamics_, "enable_steering_dynamics", false);
    getParam(wheelbase_, "wheelbase", 1.64);  // Distance between front and rear axles
    getParam(steering_ratio_, "steering_ratio", 0.6545);  // k: rear_angle = k * front_angle
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() override
  {
    return false;
  }

  /**
   * @brief Predict with steering actuator dynamics
   * @param state State to propagate with steering dynamics
   */
  void predict(models::State & state) override
  {
    if (!enable_steering_dynamics_) {
      // Use default prediction without steering dynamics
      MotionModel::predict(state);
      
      // But still need to propagate steering angle and compute wz from it
      // Otherwise steering_angle stays at 0 for all future timesteps
      if (state.steering_angle.size() > 0) {
        const unsigned int n_cols = state.vx.cols();
        const unsigned int n_rows = state.vx.rows();
        
        for (unsigned int i = 1; i < n_cols; i++) {
          for (unsigned int j = 0; j < n_rows; j++) {
            // Compute steering angle from commanded wz and vx
            float & cwz_cmd = state.cwz(j, i - 1);
            float vx_current = state.vx(j, i);
            float steering_angle = 0.0f;
            
            if (std::abs(vx_current) > 1e-3 && std::abs(cwz_cmd) > 1e-6) {
              // wz_normalized = wz * wheelbase / v = tan(df) - tan(k * df)
              float wz_normalized = (cwz_cmd * wheelbase_) / vx_current;
              
              // Use approximation or solver depending on magnitude
              if (std::abs(wz_normalized) < 0.1) {
                steering_angle = wz_normalized / (1.0f - steering_ratio_);
              } else {
                steering_angle = wz_normalized / (1.0f - steering_ratio_);
                for (int iter = 0; iter < 5; ++iter) {
                  float tan_df = std::tan(steering_angle);
                  float tan_k_df = std::tan(steering_ratio_ * steering_angle);
                  float f = tan_df - tan_k_df - wz_normalized;
                  float cos_df = std::cos(steering_angle);
                  float cos_k_df = std::cos(steering_ratio_ * steering_angle);
                  float df_prime = (1.0f / (cos_df * cos_df)) -
                                   (steering_ratio_ / (cos_k_df * cos_k_df));
                  if (std::abs(df_prime) > 1e-6) {
                    steering_angle -= f / df_prime;
                  }
                }
              }
            }
            
            // Clamp to physical limits
            steering_angle = utils::clamp(-max_steering_angle_, max_steering_angle_, steering_angle);
            state.steering_angle(j, i) = steering_angle;
            
            // Recompute wz from clamped steering angle for consistency
            if (std::abs(vx_current) > 1e-3 && std::abs(steering_angle) > 1e-6) {
              float tan_front = std::tan(steering_angle);
              float tan_rear = std::tan(steering_ratio_ * steering_angle);
              state.wz(j, i) = (vx_current / wheelbase_) * (tan_front - tan_rear);
            }
          }
        }
      }
      return;
    }

    const float max_delta_vx = model_dt_ * control_constraints_.ax_max;
    const float min_delta_vx = model_dt_ * control_constraints_.ax_min;
    const float max_delta_steering = model_dt_ * max_steering_rate_;
    const unsigned int n_cols = state.vx.cols();
    const unsigned int n_rows = state.vx.rows();

    for (unsigned int i = 1; i < n_cols; i++) {
      for (unsigned int j = 0; j < n_rows; j++) {
        // Apply velocity constraints
        float vx_last = state.vx(j, i - 1);
        float & cvx_cmd = state.cvx(j, i - 1);
        float lower_bound_vx = vx_last > 0 ? vx_last + min_delta_vx : vx_last - max_delta_vx;
        float upper_bound_vx = vx_last > 0 ? vx_last + max_delta_vx : vx_last - min_delta_vx;
        cvx_cmd = utils::clamp(lower_bound_vx, upper_bound_vx, cvx_cmd);
        cvx_cmd = utils::clamp(control_constraints_.vx_min, control_constraints_.vx_max, cvx_cmd);
        state.vx(j, i) = cvx_cmd;

        // Compute desired front steering angle from commanded wz
        // Using four-wheel steering kinematics:
        // wz = (v / wheelbase) * (tan(front_angle) - tan(k * front_angle))
        // Solve for front_angle given wz and v
        float & cwz_cmd = state.cwz(j, i - 1);
        float vx_current = state.vx(j, i);
        float desired_steering_angle = 0.0f;

        if (std::abs(vx_current) > 1e-3 && std::abs(cwz_cmd) > 1e-6) {
          // wz_normalized = wz * wheelbase / v = tan(df) - tan(k * df)
          float wz_normalized = (cwz_cmd * wheelbase_) / vx_current;

          // For small angles or when k is close to 1, use approximation
          // Otherwise use iterative solver (Newton-Raphson)
          if (std::abs(wz_normalized) < 0.1) {
            // Small angle approximation: tan(x) ≈ x
            // wz_normalized ≈ df - k*df = df(1 - k)
            desired_steering_angle = wz_normalized / (1.0f - steering_ratio_);
          } else {
            // Newton-Raphson to solve: f(df) = tan(df) - tan(k*df) - wz_norm = 0
            desired_steering_angle = wz_normalized / (1.0f - steering_ratio_); // initial guess
            for (int iter = 0; iter < 5; ++iter) {
              float tan_df = std::tan(desired_steering_angle);
              float tan_k_df = std::tan(steering_ratio_ * desired_steering_angle);
              float f = tan_df - tan_k_df - wz_normalized;

              float cos_df = std::cos(desired_steering_angle);
              float cos_k_df = std::cos(steering_ratio_ * desired_steering_angle);
              float df_prime = (1.0f / (cos_df * cos_df)) -
                               (steering_ratio_ / (cos_k_df * cos_k_df));

              if (std::abs(df_prime) > 1e-6) {
                desired_steering_angle -= f / df_prime;
              }
            }
          }
        }

        // Apply steering angle limits
        desired_steering_angle = utils::clamp(
          -max_steering_angle_, max_steering_angle_, desired_steering_angle);

        // Apply steering rate limits (actuator dynamics)
        float current_steering = state.steering_angle(j, i - 1);
        float steering_delta = desired_steering_angle - current_steering;
        steering_delta = utils::clamp(
          -max_delta_steering, max_delta_steering, steering_delta);
        float actual_front_steering = current_steering + steering_delta;
        state.steering_angle(j, i) = actual_front_steering;

        // Convert actual front steering angle to wz using exact kinematics
        if (std::abs(vx_current) > 1e-3 && std::abs(actual_front_steering) > 1e-6) {
          float tan_front = std::tan(actual_front_steering);
          float tan_rear = std::tan(steering_ratio_ * actual_front_steering);
          float actual_wz = (vx_current / wheelbase_) * (tan_front - tan_rear);
          state.wz(j, i) = actual_wz;
        } else {
          state.wz(j, i) = 0.0f;
        }
      }
    }
  }

  /**
   * @brief Apply hard vehicle constraints to a control sequence
   * @param control_sequence Control sequence to apply constraints to
   */
  void applyConstraints(models::ControlSequence & control_sequence) override
  {
    const auto wz_constrained = control_sequence.vx.abs() / min_turning_r_;
    control_sequence.wz = control_sequence.wz
      .max((-wz_constrained))
      .min(wz_constrained);
  }

  /**
   * @brief Get minimum turning radius of ackermann drive
   * @return Minimum turning radius
   */
  float getMinTurningRadius() {return min_turning_r_;}

private:
  float min_turning_r_{0};
  float max_steering_angle_{0.52f};      // Maximum front steering angle (rad)
  float max_steering_rate_{3.0f};       // Maximum steering rate (rad/s)
  bool enable_steering_dynamics_{false}; // Enable steering actuator dynamics modeling
  float wheelbase_{1.64f};               // Distance between front and rear axles (m)
  float steering_ratio_{0.6545f};        // k: rear_center_angle = k * front_center_angle
};

/**
 * @class mppi::DiffDriveMotionModel
 * @brief Differential drive motion model
 */
class DiffDriveMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::DiffDriveMotionModel
    */
  DiffDriveMotionModel() = default;

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() override
  {
    return false;
  }
};

/**
 * @class mppi::OmniMotionModel
 * @brief Omnidirectional motion model
 */
class OmniMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::OmniMotionModel
    */
  OmniMotionModel() = default;

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() override
  {
    return true;
  }
};

}  // namespace mppi

#endif  // NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_
