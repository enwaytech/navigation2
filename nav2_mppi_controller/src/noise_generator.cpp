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

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>

namespace mppi
{

namespace
{

inline uint32_t rotl(uint32_t x, int k)
{
  return (x << k) | (x >> (32 - k));
}

inline uint64_t splitmix64(uint64_t & state)
{
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// 2^-31 and 2^-32, to map 31 random bits to a uniform float in (0, 1]
constexpr float kUniformScale = 4.656612873077392578125e-10f;
constexpr float kUniformOffset = 2.3283064365386962890625e-10f;
// pi / 2 / 2^24, to map a signed 24 bit integer to an angle in [-pi/4, pi/4)
constexpr float kAngleScale = static_cast<float>(3.14159265358979323846 / 2.0 / 16777216.0);
// Single precision minimax polynomials of sin and cos on [-pi/4, pi/4] (Cephes sinf, cosf)
constexpr float kSin1 = -1.6666654611e-1f;
constexpr float kSin2 = 8.3321608736e-3f;
constexpr float kSin3 = -1.9515295891e-4f;
constexpr float kCos1 = 4.166664568298827e-2f;
constexpr float kCos2 = -1.388731625493765e-3f;
constexpr float kCos3 = 2.443315711809948e-5f;

}  // namespace

NormalSampler::NormalSampler(uint64_t seed)
{
  this->seed(seed);
}

void NormalSampler::seed(uint64_t seed)
{
  uint64_t state = seed;
  for (int l = 0; l < kLanes; ++l) {
    uint64_t a = 0, b = 0;
    while (a == 0 && b == 0) {
      a = splitmix64(state);
      b = splitmix64(state);
    }
    s0_[l] = static_cast<uint32_t>(a);
    s1_[l] = static_cast<uint32_t>(a >> 32);
    s2_[l] = static_cast<uint32_t>(b);
    s3_[l] = static_cast<uint32_t>(b >> 32);
  }
}

void NormalSampler::nextBits(uint32_t * out, int count)
{
  // xoshiro128** advanced on kLanes independent states at once
  auto s0 = s0_;
  auto s1 = s1_;
  auto s2 = s2_;
  auto s3 = s3_;
  for (int i = 0; i < count; i += kLanes) {
    for (int l = 0; l < kLanes; ++l) {
      out[i + l] = rotl(s1[l] * 5u, 7) * 9u;
      const uint32_t t = s1[l] << 9;
      s2[l] ^= s0[l];
      s3[l] ^= s1[l];
      s1[l] ^= s2[l];
      s0[l] ^= s3[l];
      s2[l] ^= t;
      s3[l] = rotl(s3[l], 11);
    }
  }
  s0_ = s0;
  s1_ = s1;
  s2_ = s2;
  s3_ = s3;
}

void NormalSampler::generateBlock(float * first, float * second, float stddev)
{
  alignas(64) uint32_t radius_bits[kBlockPairs];
  alignas(64) uint32_t angle_bits[kBlockPairs];
  nextBits(radius_bits, kBlockPairs);
  nextBits(angle_bits, kBlockPairs);

  // Box-Muller radius sqrt(-2 ln(u)) with u in (0, 1], so log(0) cannot occur, and the
  // clamp keeps the square root argument non-negative at u = 1
  Eigen::Array<float, kBlockPairs, 1> radius;
  for (int k = 0; k < kBlockPairs; ++k) {
    radius[k] = static_cast<float>(static_cast<int32_t>(radius_bits[k] >> 1)) * kUniformScale +
      kUniformOffset;
  }
  radius = (-2.0f * radius.log()).cwiseMax(0.0f).sqrt() * stddev;

  // Angle uniform in [-pi/4, pi/4) from the top 24 bits. Bit 0 swaps cos and sin, which
  // reflects the angle onto [pi/4, 3pi/4), bit 1 negates both, which rotates it by pi:
  // together the angle is uniform on the full circle. The swap and the sign are applied
  // with bit masks so that the loop vectorizes without branches.
  for (int k = 0; k < kBlockPairs; ++k) {
    const uint32_t bits = angle_bits[k];
    const float x = static_cast<float>(static_cast<int32_t>(bits >> 8) - (1 << 23)) * kAngleScale;
    const float z = x * x;
    const float sin_x = ((kSin3 * z + kSin2) * z + kSin1) * z * x + x;
    const float cos_x = ((kCos3 * z + kCos2) * z + kCos1) * z * z - 0.5f * z + 1.0f;
    uint32_t cos_bits, sin_bits;
    std::memcpy(&cos_bits, &cos_x, sizeof(float));
    std::memcpy(&sin_bits, &sin_x, sizeof(float));
    const uint32_t swap = (cos_bits ^ sin_bits) & (0u - (bits & 1u));
    const uint32_t sign = (bits & 2u) << 30;
    cos_bits ^= swap ^ sign;
    sin_bits ^= swap ^ sign;
    float a, b;
    std::memcpy(&a, &cos_bits, sizeof(float));
    std::memcpy(&b, &sin_bits, sizeof(float));
    first[k] = radius[k] * a;
    second[k] = radius[k] * b;
  }
}

void NormalSampler::fill(float * data, Eigen::Index size, float stddev)
{
  constexpr Eigen::Index block_size = 2 * kBlockPairs;
  Eigen::Index done = 0;
  for (; done + block_size <= size; done += block_size) {
    generateBlock(data + done, data + done + kBlockPairs, stddev);
  }
  if (done < size) {
    alignas(64) float tail[block_size];
    generateBlock(tail, tail + kBlockPairs, stddev);
    std::copy(tail, tail + (size - done), data + done);
  }
}

void NoiseGenerator::initialize(
  mppi::models::OptimizerSettings & settings, bool is_holonomic,
  const std::string & name, ParametersHandler * param_handler)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  active_ = true;

  sampling_std_ = settings_.sampling_std;

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
  state.cwz = noises_wz_.rowwise() + control_sequence.wz.transpose();
  // cvy is only read by holonomic motion models
  if (is_holonomic_) {
    state.cvy = noises_vy_.rowwise() + control_sequence.vy.transpose();
  }
}

void NoiseGenerator::reset(mppi::models::OptimizerSettings & settings, bool is_holonomic)
{
  // Recompute the noises on reset, initialization, and fallback. They are drawn here and not by
  // the noise thread: the optimizer may use them right after the reset, and would otherwise
  // find the zeros or the new noises depending on which thread is faster.
  std::unique_lock<std::mutex> guard(noise_lock_);
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  noises_vx_.setZero(settings_.batch_size, settings_.time_steps);
  noises_vy_.setZero(settings_.batch_size, settings_.time_steps);
  noises_wz_.setZero(settings_.batch_size, settings_.time_steps);
  generateNoisedControls();
  // A pending request to the noise thread is served by this
  ready_ = false;
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
  // resize() keeps the existing buffers when the shape is unchanged
  noises_vx_.resize(s.batch_size, s.time_steps);
  noises_wz_.resize(s.batch_size, s.time_steps);
  sampler_.fill(noises_vx_.data(), noises_vx_.size(), sampling_std_.vx);
  sampler_.fill(noises_wz_.data(), noises_wz_.size(), sampling_std_.wz);
  if (is_holonomic_) {
    noises_vy_.resize(s.batch_size, s.time_steps);
    sampler_.fill(noises_vy_.data(), noises_vy_.size(), sampling_std_.vy);
  }
}

}  // namespace mppi
