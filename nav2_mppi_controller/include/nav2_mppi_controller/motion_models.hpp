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
#include <geometry_msgs/msg/twist.hpp>

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
  void initialize(const models::ControlConstraints & control_constraints, float model_dt,
                  bool use_adaptive_lag, float smoothing_alpha, float min_tau, float max_tau)
  {
    control_constraints_ = control_constraints;
    model_dt_ = model_dt;
    use_adaptive_lag_ = use_adaptive_lag;
    smoothing_alpha_ = smoothing_alpha;
    min_tau_ = min_tau;
    max_tau_ = max_tau;
  }

 void updateTau(
    const geometry_msgs::msg::Twist & actual_speed,
    const geometry_msgs::msg::Twist & cmd_speed,
    float dt)
  {

    std::cout << "Update Tau called" << std::endl;
    if (!use_adaptive_lag_) {
      return;
    }

    const float epsilon = 1e-6f; // Avoid division by zero

    // Estimate tau for vx
    float vx_actual = actual_speed.linear.x;
    float vx_cmd = cmd_speed.linear.x;
    float vx_diff = vx_actual - last_vx_;
    if (std::abs(vx_diff) > epsilon && std::abs(vx_cmd - last_vx_) > epsilon) {
      float tau_est = dt * (vx_cmd - last_vx_) / vx_diff;
      tau_vx_ = smoothing_alpha_ * std::clamp(tau_est, min_tau_, max_tau_) +
                (1.0f - smoothing_alpha_) * tau_vx_;
    }
    last_vx_ = vx_actual;

    // Estimate tau for wz
    float wz_actual = actual_speed.angular.z;
    float wz_cmd = cmd_speed.angular.z;
    float wz_diff = wz_actual - last_wz_;
    if (std::abs(wz_diff) > epsilon && std::abs(wz_cmd - last_wz_) > epsilon) {
      float tau_est = dt * (wz_cmd - last_wz_) / wz_diff;
      tau_wz_ = smoothing_alpha_ * std::clamp(tau_est, min_tau_, max_tau_) +
                (1.0f - smoothing_alpha_) * tau_wz_;
    }
    last_wz_ = wz_actual;

    // Estimate tau for vy (if holonomic)
    if (isHolonomic()) {
      float vy_actual = actual_speed.linear.y;
      float vy_cmd = cmd_speed.linear.y;
      float vy_diff = vy_actual - last_vy_;
      if (std::abs(vy_diff) > epsilon && std::abs(vy_cmd - last_vy_) > epsilon) {
        float tau_est = dt * (vy_cmd - last_vy_) / vy_diff;
        tau_vy_ = smoothing_alpha_ * std::clamp(tau_est, min_tau_, max_tau_) +
                  (1.0f - smoothing_alpha_) * tau_vy_;
      }
      last_vy_ = vy_actual;
    }
    std::cout << "Tau vx: " << tau_vx_ << " Tau vy: " << tau_vy_ << " Tau wz: " << tau_wz_ << std::endl;
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

    unsigned int n_rows = state.vx.rows();
    unsigned int n_cols = state.vx.cols();
    // Default layout in eigen is column-major, hence accessing elements in
    // column-major fashion to utilize L1 cache as much as possible
    for (unsigned int i = 1; i != n_cols; i++) {
      for (unsigned int j = 0; j != n_rows; j++) {
        float vx_last = state.vx(j, i - 1);
        float & cvx_cmd = state.cvx(j, i - 1);
        float min_vx_acc = vx_last + min_delta_vx;
        float max_vx_acc = vx_last + max_delta_vx;
        float cvx_feasible = utils::clamp(min_vx_acc, max_vx_acc, cvx_cmd);

        // float vx_new = use_adaptive_lag_ ?
        //   vx_last + model_dt_ * (cvx_feasible - vx_last) / tau_vx_ :
        //   cvx_feasible;
        state.vx(j, i) = utils::clamp(control_constraints_.vx_min, control_constraints_.vx_max, cvx_feasible);

        float wz_last = state.wz(j, i - 1);
        float & cwz_cmd = state.cwz(j, i - 1);
        float min_wz_acc = wz_last - max_delta_wz;
        float max_wz_acc = wz_last + max_delta_wz;
        float cwz_feasible = utils::clamp(min_wz_acc, max_wz_acc, cwz_cmd);

        // float wz_new = use_adaptive_lag_ ?
        //   wz_last + model_dt_ * (cwz_feasible - wz_last) / tau_wz_ :
        //   cwz_feasible;
        state.wz(j, i) = utils::clamp(-control_constraints_.wz, control_constraints_.wz, cwz_feasible);

        if (is_holo) {
          float vy_last = state.vy(j, i - 1);
          float & cvy_cmd = state.cvy(j, i - 1);
          float min_vy_acc = vy_last + min_delta_vy;
          float max_vy_acc = vy_last + max_delta_vy;
          float cvy_feasible = utils::clamp(min_vy_acc, max_vy_acc, cvy_cmd);

          // float vy_new = use_adaptive_lag_ ?
          //   vy_last + model_dt_ * (cvy_feasible - vy_last) / tau_vy_ :
          //   cvy_feasible;
          state.vy(j, i) = utils::clamp(-control_constraints_.vy, control_constraints_.vy, cvy_feasible);
        }
      }
    }
    // for (unsigned int i = 1; i < n_cols; i++) {
    //   auto lower_bound_vx = (state.vx.col(i - 1) >
    //     0).select(state.vx.col(i - 1) + min_delta_vx,
    //     state.vx.col(i - 1) - max_delta_vx);
    //   auto upper_bound_vx = (state.vx.col(i - 1) >
    //     0).select(state.vx.col(i - 1) + max_delta_vx,
    //     state.vx.col(i - 1) - min_delta_vx);

    //   state.cvx.col(i - 1) = state.cvx.col(i - 1)
    //     .cwiseMax(lower_bound_vx)
    //     .cwiseMin(upper_bound_vx);
    //   state.vx.col(i) = state.cvx.col(i - 1);

    //   state.cwz.col(i - 1) = state.cwz.col(i - 1)
    //     .cwiseMax(state.wz.col(i - 1) - max_delta_wz)
    //     .cwiseMin(state.wz.col(i - 1) + max_delta_wz);
    //   state.wz.col(i) = state.cwz.col(i - 1);

    //   if (is_holo) {
    //     auto lower_bound_vy = (state.vy.col(i - 1) >
    //       0).select(state.vy.col(i - 1) + min_delta_vy,
    //       state.vy.col(i - 1) - max_delta_vy);
    //     auto upper_bound_vy = (state.vy.col(i - 1) >
    //       0).select(state.vy.col(i - 1) + max_delta_vy,
    //       state.vy.col(i - 1) - min_delta_vy);

    //     state.cvy.col(i - 1) = state.cvy.col(i - 1)
    //       .cwiseMax(lower_bound_vy)
    //       .cwiseMin(upper_bound_vy);
    //     state.vy.col(i) = state.cvy.col(i - 1);
    //   }
    // }
  }

  // Getter methods for tau values
  float getTauVx() const { return tau_vx_; }
  float getTauVy() const { return tau_vy_; }
  float getTauWz() const { return tau_wz_; }

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
  models::ControlConstraints control_constraints_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  bool use_adaptive_lag_{false};
  float smoothing_alpha_{0.2f};
  float min_tau_{0.01f};
  float max_tau_{0.5f};
  float tau_vx_{1e-6f};
  float tau_vy_{1e-6f};
  float tau_wz_{1e-6f};
  float last_vx_{0.0f};
  float last_vy_{0.0f};
  float last_wz_{0.0f};
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
