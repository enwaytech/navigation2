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
        0).select(
        state.vx.col(i - 1) + min_delta_vx,
        state.vx.col(i - 1) - max_delta_vx);
      auto upper_bound_vx = (state.vx.col(i - 1) >
        0).select(
        state.vx.col(i - 1) + max_delta_vx,
        state.vx.col(i - 1) - min_delta_vx);

      state.cvx.col(i - 1) = state.cvx.col(i - 1)
        .cwiseMax(lower_bound_vx)
        .cwiseMin(upper_bound_vx)
        .cwiseMax(control_constraints_.vx_min)
        .cwiseMin(control_constraints_.vx_max);
      state.vx.col(i) = state.cvx.col(i - 1);

      state.cwz.col(i - 1) = state.cwz.col(i - 1)
        .cwiseMax(state.wz.col(i - 1) - max_delta_wz)
        .cwiseMin(state.wz.col(i - 1) + max_delta_wz);
      state.wz.col(i) = state.cwz.col(i - 1);

      if (is_holo) {
        auto lower_bound_vy = (state.vy.col(i - 1) >
          0).select(
          state.vy.col(i - 1) + min_delta_vy,
          state.vy.col(i - 1) - max_delta_vy);
        auto upper_bound_vy = (state.vy.col(i - 1) >
          0).select(
          state.vy.col(i - 1) + max_delta_vy,
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
    // Optional steering-rate-aware constraint. When both wheel_base and
    // max_steering_angle_velocity are positive, |Δwz/dt| is clamped
    // per-timestep based on the achievable steering angle velocity:
    //   |dwz/dt|_max = max_steering_angle_velocity * |vx| / wheel_base
    // (small-angle approximation; exact form would include sec²(δ))
    getParam(wheel_base_, "wheel_base", 0.0f);
    getParam(max_steering_angle_velocity_, "max_steering_angle_velocity", 0.0f);
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
   * @brief Predict next state, with steering-rate-aware wz rate limit.
   * Replaces the base class az_max-based wz constraint with a state-dependent
   * limit derived from the steering actuator's max angular velocity.
   * Falls back to the base implementation if either param is unset.
   */
  void predict(models::State & state) override
  {
    if (wheel_base_ <= 0.0f || max_steering_angle_velocity_ <= 0.0f) {
      MotionModel::predict(state);
      return;
    }

    float max_delta_vx = model_dt_ * control_constraints_.ax_max;
    float min_delta_vx = model_dt_ * control_constraints_.ax_min;

    unsigned int n_cols = state.vx.cols();

    for (unsigned int i = 1; i < n_cols; i++) {
      auto lower_bound_vx = (state.vx.col(i - 1) > 0).select(
        state.vx.col(i - 1) + min_delta_vx,
        state.vx.col(i - 1) - max_delta_vx);
      auto upper_bound_vx = (state.vx.col(i - 1) > 0).select(
        state.vx.col(i - 1) + max_delta_vx,
        state.vx.col(i - 1) - min_delta_vx);

      state.cvx.col(i - 1) = state.cvx.col(i - 1)
        .cwiseMax(lower_bound_vx)
        .cwiseMin(upper_bound_vx)
        .cwiseMax(control_constraints_.vx_min)
        .cwiseMin(control_constraints_.vx_max);
      state.vx.col(i) = state.cvx.col(i - 1);

      // State-dependent wz rate limit including sec²(δ) factor:
      //   |dwz/dt|_max = (|vx|/L) × sec²(δ) × max_steering_velocity
      //   sec²(δ) = 1 + (wz × L / vx)²    (since tan(δ) = wz × L / vx)
      // This is tight at δ=0 and loosens with steering angle, matching the real
      // achievable wz acceleration.
      const float k = max_steering_angle_velocity_ * model_dt_ / wheel_base_;
      auto vx_abs = state.vx.col(i - 1).abs();
      auto vx_safe = vx_abs.cwiseMax(0.01f);  // floor to avoid div-by-zero
      auto wzL_over_vx = state.wz.col(i - 1) * wheel_base_ / vx_safe;
      auto sec_sq = 1.0f + wzL_over_vx.square();
      auto max_delta_wz_vec = (vx_abs * sec_sq * k).eval();

      state.cwz.col(i - 1) = state.cwz.col(i - 1)
        .cwiseMax(state.wz.col(i - 1) - max_delta_wz_vec)
        .cwiseMin(state.wz.col(i - 1) + max_delta_wz_vec);
      state.wz.col(i) = state.cwz.col(i - 1);
    }
  }

  /**
   * @brief Apply hard vehicle constraints to a control sequence
   * @param control_sequence Control sequence to apply constraints to
   */
  void applyConstraints(models::ControlSequence & control_sequence) override
  {
    // Geometric constraint: |wz| <= |vx| / min_turning_r
    const auto wz_constrained = control_sequence.vx.abs() / min_turning_r_;
    control_sequence.wz = control_sequence.wz
      .max((-wz_constrained))
      .min(wz_constrained);

    // State-dependent rate limit with sec²(δ) factor:
    //   |Δwz| <= |vx| × (1 + (wz×L/vx)²) × max_steering_velocity × dt / L
    if (wheel_base_ > 0.0f && max_steering_angle_velocity_ > 0.0f) {
      const float k = max_steering_angle_velocity_ * model_dt_ / wheel_base_;
      for (Eigen::Index i = 1; i < control_sequence.wz.size(); ++i) {
        const float vx_abs = std::abs(control_sequence.vx(i - 1));
        const float vx_safe = std::max(vx_abs, 0.01f);
        const float wzL_over_vx = control_sequence.wz(i - 1) * wheel_base_ / vx_safe;
        const float sec_sq = 1.0f + wzL_over_vx * wzL_over_vx;
        const float max_dwz = vx_abs * sec_sq * k;
        const float lo = control_sequence.wz(i - 1) - max_dwz;
        const float hi = control_sequence.wz(i - 1) + max_dwz;
        control_sequence.wz(i) = std::clamp(control_sequence.wz(i), lo, hi);
      }
    }
  }

  /**
   * @brief Get minimum turning radius of ackermann drive
   * @return Minimum turning radius
   */
  float getMinTurningRadius() {return min_turning_r_;}

private:
  float min_turning_r_{0};
  float wheel_base_{0};
  float max_steering_angle_velocity_{0};
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
