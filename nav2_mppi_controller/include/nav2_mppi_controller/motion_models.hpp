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
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

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
  void initialize(
    const models::ControlConstraints & control_constraints, float model_dt,
    float model_delay_vx, float model_delay_vy, float model_delay_wz)
  {
    control_constraints_ = control_constraints;
    model_dt_ = model_dt;
    model_delay_vx_ = model_delay_vx;
    model_delay_vy_ = model_delay_vy;
    model_delay_wz_ = model_delay_wz;

    cmd_history_vx_.resize(offsetSteps(model_delay_vx_), 0.0f);
    cmd_history_vy_.resize(offsetSteps(model_delay_vy_), 0.0f);
    cmd_history_wz_.resize(offsetSteps(model_delay_wz_), 0.0f);
  }

  /**
    * @brief Push the most recently published command to the per-axis history
    *        ring buffers. Called once per controller cycle from the optimizer.
    */
  void pushCommandHistory(float vx, float vy, float wz)
  {
    pushOne(cmd_history_vx_, vx);
    pushOne(cmd_history_vy_, vy);
    pushOne(cmd_history_wz_, wz);
  }

  /**
    * @brief Zero the ring buffers
    */
  void clearCommandHistory()
  {
    std::fill(cmd_history_vx_.begin(), cmd_history_vx_.end(), 0.0f);
    std::fill(cmd_history_vy_.begin(), cmd_history_vy_.end(), 0.0f);
    std::fill(cmd_history_wz_.begin(), cmd_history_wz_.end(), 0.0f);
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

    const unsigned int offset_vx = std::floor((model_delay_vx_ / model_dt_) + 0.5);
    const unsigned int offset_vy = std::floor((model_delay_vy_ / model_dt_) + 0.5);
    const unsigned int offset_wz = std::floor((model_delay_wz_ / model_dt_) + 0.5);

    // The input delay is applied in the same pass: the bounded control of column i - 1
    // becomes the velocity of column i - 1 + shift, and columns [1, offset) replay the
    // command history.
    const unsigned int shift_vx = std::max(offset_vx, 1u);
    const unsigned int shift_vy = std::max(offset_vy, 1u);
    const unsigned int shift_wz = std::max(offset_wz, 1u);
    fillDelayWindow(state.vx, offset_vx, cmd_history_vx_);
    fillDelayWindow(state.wz, offset_wz, cmd_history_wz_);
    if (is_holo) {
      fillDelayWindow(state.vy, offset_vy, cmd_history_vy_);
    }

    // Bounds the controls of column i - 1 against the velocity of column i - 1 before the
    // delay shift (the initial velocity or the previous bounded control) and writes them to
    // the shifted velocity column
    auto bound_axis = [n_cols](
      Eigen::ArrayXXf & velocities, Eigen::ArrayXXf & controls, unsigned int i,
      unsigned int shift, float min_delta, float max_delta) {
        const float * prev = i == 1 ? velocities.col(0).data() : controls.col(i - 2).data();
        float * velocity = i - 1 + shift < n_cols ? velocities.col(i - 1 + shift).data() : nullptr;
        boundControls(
          controls.col(i - 1).data(), prev, velocity, controls.rows(), min_delta, max_delta);
      };

    for (unsigned int i = 1; i < n_cols; i++) {
      bound_axis(state.vx, state.cvx, i, shift_vx, min_delta_vx, max_delta_vx);
      // Symmetric bounds: p + (-d) and p - (-d) equal p - d and p + d exactly
      bound_axis(state.wz, state.cwz, i, shift_wz, -max_delta_wz, max_delta_wz);
      if (is_holo) {
        bound_axis(state.vy, state.cvy, i, shift_vy, min_delta_vy, max_delta_vy);
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
  /**
    * @brief Bound controls by the acceleration limits relative to the previous velocities
    * @param control Controls to bound in place
    * @param prev Previous velocities
    * @param velocity Optional destination for a copy of the bounded controls
    * @param size Number of elements
    * @param min_delta Lower velocity change limit when the previous velocity is positive
    * @param max_delta Upper velocity change limit when the previous velocity is positive
    */
  static void boundControls(
    float * control, const float * prev, float * velocity, Eigen::Index size,
    float min_delta, float max_delta)
  {
    // Lower bound p + min_delta if p > 0 else p - max_delta, upper bound p + max_delta if
    // p > 0 else p - min_delta. Adding a negated delta is exact, so only the delta is
    // selected, with a bit mask: compilers turn a conditional floating point addition into
    // a branch, which does not vectorize.
    auto to_bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
      };
    auto to_float = [](uint32_t u) {
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
      };
    const uint32_t lower_pos = to_bits(min_delta);
    const uint32_t lower_neg = to_bits(-max_delta);
    const uint32_t upper_pos = to_bits(max_delta);
    const uint32_t upper_neg = to_bits(-min_delta);
    auto bound = [&](float c, float p) {
        const uint32_t positive = 0u - static_cast<uint32_t>(p > 0);
        const float lower = p + to_float((lower_pos & positive) | (lower_neg & ~positive));
        const float upper = p + to_float((upper_pos & positive) | (upper_neg & ~positive));
        return std::min(std::max(c, lower), upper);
      };
    if (velocity != nullptr) {
      for (Eigen::Index r = 0; r != size; r++) {
        const float bounded = bound(control[r], prev[r]);
        control[r] = bounded;
        velocity[r] = bounded;
      }
    } else {
      for (Eigen::Index r = 0; r != size; r++) {
        control[r] = bound(control[r], prev[r]);
      }
    }
  }

  /**
    * @brief Fill the delay window columns [1, offset) with the in-flight commands
    *        from the history ring buffer.
    */
  static void fillDelayWindow(
    Eigen::ArrayXXf & velocities, unsigned int offset, const std::vector<float> & history)
  {
    const unsigned int end = std::min(offset, static_cast<unsigned int>(velocities.cols()));
    for (unsigned int j = 1; j < end; ++j) {
      velocities.col(j).setConstant(history[j]);
    }
  }

  /**
    * @brief Convert a delay in seconds to an offset in number of rollout steps, rounding to the nearest step.
    */
  std::size_t offsetSteps(float delay) const
  {
    if (delay <= 0.0f || model_dt_ <= 0.0f) {
      return 0u;
    }
    return static_cast<std::size_t>(std::floor(delay / model_dt_ + 0.5f));
  }

  /**
    * @brief Push a value to the back of a ring buffer and rotate the elements.
    */
  static void pushOne(std::vector<float> & buf, float v)
  {
    if (buf.empty()) {return;}
    std::rotate(buf.begin(), buf.begin() + 1, buf.end());
    buf.back() = v;
  }

  float model_dt_{0.0};
  float model_delay_vx_{0.0};
  float model_delay_vy_{0.0};
  float model_delay_wz_{0.0};

  // Per-axis ring buffer of recently published commands
  std::vector<float> cmd_history_vx_;
  std::vector<float> cmd_history_vy_;
  std::vector<float> cmd_history_wz_;
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
