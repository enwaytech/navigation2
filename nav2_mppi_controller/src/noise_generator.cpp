// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include "nav2_mppi_controller/tools/noise_generator.hpp"

#include <cmath>
#include <memory>
#include <mutex>

namespace mppi
{

namespace
{

/**
 * @brief value * multiplier mod (2^31 - 1) for value < 2^31 and multiplier < 2^31 - 1. The modulo
 * uses 2^31 = 1 (mod 2^31 - 1) instead of a division.
 */
template<uint64_t multiplier>
inline uint64_t mulModMersenne31(uint64_t value)
{
  const uint64_t product = value * multiplier;
  const uint64_t folded = (product & 0x7fffffffu) + (product >> 31);
  return folded >= 0x7fffffffu ? folded - 0x7fffffffu : folded;
}

/**
 * @brief Next value of std::minstd_rand0: value * 16807 mod (2^31 - 1)
 */
inline uint64_t minstdRand0Next(uint64_t value)
{
  return mulModMersenne31<16807u>(value);
}

/**
 * @brief Value after the next one of std::minstd_rand0: value * 16807^2 mod (2^31 - 1)
 */
inline uint64_t minstdRand0NextNext(uint64_t value)
{
  return mulModMersenne31<282475249u>(value);
}

/**
 * @brief std::generate_canonical<float, 24>() of a std::minstd_rand0 value, as libstdc++ computes it
 */
inline float canonicalFloat(uint64_t value)
{
  float result = static_cast<float>(value - 1u) / 2147483648.0f;
  if (result >= 1.0f) {
    result = std::nextafter(1.0f, 0.0f);
  }
  return result;
}

}  // namespace

void NoiseGenerator::initialize(
  mppi::models::OptimizerSettings & settings, bool is_holonomic,
  const std::string & name, ParametersHandler * param_handler)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  active_ = true;

  ndistribution_vx_ = NormalDistribution{0.0f, settings_.sampling_std.vx};
  ndistribution_vy_ = NormalDistribution{0.0f, settings_.sampling_std.vy};
  ndistribution_wz_ = NormalDistribution{0.0f, settings_.sampling_std.wz};

  auto getParam = param_handler->getParamGetter(name);
  getParam(regenerate_noises_, "regenerate_noises", false);

  if (regenerate_noises_) {
    noise_thread_ = std::thread(std::bind(&NoiseGenerator::noiseThread, this));
  } else {
    generateNoisedControls();
  }
}

void NoiseGenerator::shutdown()
{
  active_ = false;
  ready_ = true;
  noise_cond_.notify_all();
  if (noise_thread_.joinable()) {
    noise_thread_.join();
  }
}

void NoiseGenerator::generateNextNoises()
{
  // Trigger the thread to run in parallel to this iteration
  // to generate the next iteration's noises (if applicable).
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    ready_ = true;
  }
  noise_cond_.notify_all();
}

void NoiseGenerator::setNoisedControls(
  models::State & state,
  const models::ControlSequence & control_sequence)
{
  std::unique_lock<std::mutex> guard(noise_lock_);

  state.cvx = noises_vx_.rowwise() + control_sequence.vx.transpose();
  state.cvy = noises_vy_.rowwise() + control_sequence.vy.transpose();
  state.cwz = noises_wz_.rowwise() + control_sequence.wz.transpose();
}

void NoiseGenerator::reset(mppi::models::OptimizerSettings & settings, bool is_holonomic)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;

  // Recompute the noises on reset, initialization, and fallback
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noises_vx_.setZero(settings_.batch_size, settings_.time_steps);
    noises_vy_.setZero(settings_.batch_size, settings_.time_steps);
    noises_wz_.setZero(settings_.batch_size, settings_.time_steps);
    ready_ = true;
  }

  if (regenerate_noises_) {
    noise_cond_.notify_all();
  } else {
    generateNoisedControls();
  }
}

void NoiseGenerator::noiseThread()
{
  do {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noise_cond_.wait(guard, [this]() {return ready_;});
    ready_ = false;
    generateNoisedControls();
  } while (active_);
}

void NoiseGenerator::generateNoisedControls()
{
  auto & s = settings_;
  noises_vx_.resize(s.batch_size, s.time_steps);
  sampleNormal(noises_vx_, ndistribution_vx_);
  noises_wz_.resize(s.batch_size, s.time_steps);
  sampleNormal(noises_wz_, ndistribution_wz_);
  if (is_holonomic_) {
    noises_vy_.resize(s.batch_size, s.time_steps);
    sampleNormal(noises_vy_, ndistribution_vy_);
  }
}

void NoiseGenerator::sampleNormal(Eigen::ArrayXXf & noises, NormalDistribution & distribution)
{
  float * out = noises.data();
  const Eigen::Index size = noises.size();
  Eigen::Index i = 0;
  if (size > 0 && distribution.saved_available) {
    distribution.saved_available = false;
    out[i++] = distribution.saved * distribution.stddev + distribution.mean;
  }

  // Marsaglia polar method of libstdc++'s std::normal_distribution
  uint64_t state = generator_state_;
  while (i < size) {
    float x, y, r2;
    do {
      // Both values from the current state, so that the two generator steps do not wait on
      // each other
      const uint64_t state_x = minstdRand0Next(state);
      state = minstdRand0NextNext(state);
      x = 2.0f * canonicalFloat(state_x) - 1.0f;
      y = 2.0f * canonicalFloat(state) - 1.0f;
      // As the compiled libstdc++ code: x * x rounded, y * y fused
      r2 = std::fma(y, y, x * x);
    } while (r2 > 1.0f || r2 == 0.0f);

    const float mult = std::sqrt(-2.0f * std::log(r2) / r2);
    out[i++] = (y * mult) * distribution.stddev + distribution.mean;
    const float saved = x * mult;
    if (i < size) {
      out[i++] = saved * distribution.stddev + distribution.mean;
    } else {
      distribution.saved = saved;
      distribution.saved_available = true;
    }
  }
  generator_state_ = state;
}

}  // namespace mppi
