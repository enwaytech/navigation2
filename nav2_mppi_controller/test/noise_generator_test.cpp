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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav2_mppi_controller/tools/noise_generator.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"
#include "nav2_mppi_controller/models/optimizer_settings.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/control_sequence.hpp"

// Tests noise generator object

using namespace mppi;  // NOLINT

TEST(NoiseGeneratorTest, NoiseGeneratorLifecycle)
{
  // Tests shuts down internal thread cleanly
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;

  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  std::string name = "test";
  ParametersHandler handler(node, name);

  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);
  generator.shutdown();
}

TEST(NoiseGeneratorTest, NoiseGeneratorMain)
{
  // Tests shuts down internal thread cleanly
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(true));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  // Populate a potential control sequence
  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(25);
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i;
    control_sequence.vy(i) = i;
    control_sequence.wz(i) = i;
  }

  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  // Request an update with no noise yet generated, should result in identical outputs
  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);  // sets initial sizing and draws the noises
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // save initial state
  auto initial_cvx_0 = state.cvx(0);
  auto initial_cvy_0 = state.cvy(0);
  auto initial_cwz_0 = state.cwz(0);
  auto initial_cvx_9 = state.cvx(0, 9);
  auto initial_cvy_9 = state.cvy(0, 9);
  auto initial_cwz_9 = state.cwz(0, 9);

  EXPECT_NE(state.cvx(0), 0);
  EXPECT_EQ(state.cvy(0), 0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0), 0);
  EXPECT_NE(state.cvx(0, 9), 9);
  EXPECT_EQ(state.cvy(0, 9), 0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), 9);

  EXPECT_NEAR(state.cvx(0), 0, 0.3);
  EXPECT_NEAR(state.cwz(0), 0, 0.3);
  EXPECT_NEAR(state.cvx(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cwz(0, 9), 9, 0.3);

  // Request an update with noise requested
  generator.generateNextNoises();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // Ensure the state has changed after generating new noises
  EXPECT_NE(state.cvx(0), initial_cvx_0);
  EXPECT_EQ(state.cvy(0), initial_cvy_0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0), initial_cwz_0);
  EXPECT_NE(state.cvx(0, 9), initial_cvx_9);
  EXPECT_EQ(state.cvy(0, 9), initial_cvy_9);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), initial_cwz_9);


  // Test holonomic setting
  generator.reset(settings, true);  // Now holonomically
  generator.generateNextNoises();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);
  EXPECT_NE(state.cvx(0), 0);
  EXPECT_NE(state.cvy(0), 0);  // Now populated in non-holonomic
  EXPECT_NE(state.cwz(0), 0);
  EXPECT_NE(state.cvx(0, 9), 9);
  EXPECT_NE(state.cvy(0, 9), 9);  // Now populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), 9);

  EXPECT_NEAR(state.cvx(0), 0, 0.3);
  EXPECT_NEAR(state.cvy(0), 0, 0.3);
  EXPECT_NEAR(state.cwz(0), 0, 0.3);
  EXPECT_NEAR(state.cvx(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cvy(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cwz(0, 9), 9, 0.3);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, ResetDrawsNoisesBeforeReturning)
{
  // The optimizer uses the noises right after a reset, without waiting for the noise thread
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(true));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(25);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  generator.initialize(settings, true, "test_name", &handler);
  for (int i = 0; i != 20; i++) {
    // The noise thread is busy with a request, as after an optimization
    generator.generateNextNoises();
    generator.reset(settings, true);
    generator.setNoisedControls(state, control_sequence);
    EXPECT_GT(state.cvx.abs().maxCoeff(), 0.0f);
    EXPECT_GT(state.cvy.abs().maxCoeff(), 0.0f);
    EXPECT_GT(state.cwz.abs().maxCoeff(), 0.0f);
    EXPECT_NEAR(state.cvx.mean(), 0.0f, 0.02f);
  }

  generator.shutdown();
}

TEST(NoiseGeneratorTest, NoiseGeneratorMainNoRegenerate)
{
  // This time with no regeneration of noises
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  // Populate a potential control sequence
  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(25);
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i;
    control_sequence.vy(i) = i;
    control_sequence.wz(i) = i;
  }

  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  // Request an update with no noise yet generated, should result in identical outputs
  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);  // sets initial sizing and draws the noises
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // save initial state
  auto initial_cvx_0 = state.cvx(0);
  auto initial_cvy_0 = state.cvy(0);
  auto initial_cwz_0 = state.cwz(0);
  auto initial_cvx_9 = state.cvx(0, 9);
  auto initial_cvy_9 = state.cvy(0, 9);
  auto initial_cwz_9 = state.cwz(0, 9);

  EXPECT_NE(state.cvx(0), 0);
  EXPECT_EQ(state.cvy(0), 0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0), 0);
  EXPECT_NE(state.cvx(0, 9), 9);
  EXPECT_EQ(state.cvy(0, 9), 0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), 9);

  EXPECT_NEAR(state.cvx(0), 0, 0.3);
  EXPECT_NEAR(state.cwz(0), 0, 0.3);
  EXPECT_NEAR(state.cvx(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cwz(0, 9), 9, 0.3);

  // this doesn't work if regenerate_noises is false
  generator.generateNextNoises();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // Ensure the state has changed after generating new noises
  EXPECT_EQ(state.cvx(0), initial_cvx_0);
  EXPECT_EQ(state.cvy(0), initial_cvy_0);  // Not populated in non-holonomic
  EXPECT_EQ(state.cwz(0), initial_cwz_0);
  EXPECT_EQ(state.cvx(0, 9), initial_cvx_9);
  EXPECT_EQ(state.cvy(0, 9), initial_cvy_9);  // Not populated in non-holonomic
  EXPECT_EQ(state.cwz(0, 9), initial_cwz_9);

  generator.shutdown();
}

// Statistical checks of the normal samples. Every check compares an estimate with its
// expected value using a tolerance of kSigmas standard errors of that estimate for the sample
// size, so a correct generator fails one check with probability about 2e-9 (normal
// approximation, two-sided 6 sigma). With fewer than 200 checks in this file, the chance that
// a fixed seed of a correct generator fails any of them is below 1e-6.
constexpr double kSigmas = 6.0;

double normalCdf(double x)
{
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double correlation(const float * a, const float * b, size_t n)
{
  double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sa += a[i];
    sb += b[i];
    saa += static_cast<double>(a[i]) * a[i];
    sbb += static_cast<double>(b[i]) * b[i];
    sab += static_cast<double>(a[i]) * b[i];
  }
  const double cov = sab / n - (sa / n) * (sb / n);
  return cov / std::sqrt((saa / n - (sa / n) * (sa / n)) * (sbb / n - (sb / n) * (sb / n)));
}

// Correlation of a sample with the sample lag positions later in memory
double lagCorrelation(const std::vector<float> & x, size_t lag)
{
  return correlation(x.data(), x.data() + lag, x.size() - lag);
}

// Correlation of the squares, which detects dependence that leaves the samples uncorrelated
double lagSquaredCorrelation(const std::vector<float> & x, size_t lag)
{
  std::vector<float> sq(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    sq[i] = x[i] * x[i];
  }
  return lagCorrelation(sq, lag);
}

void expectNormal(const float * x, size_t n, double stddev)
{
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    ASSERT_TRUE(std::isfinite(x[i]));
    sum += x[i];
  }
  const double mean = sum / n;
  double m2 = 0.0, m3 = 0.0, m4 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = x[i] - mean;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const double var = stddev * stddev;
  const double nd = static_cast<double>(n);

  EXPECT_NEAR(mean, 0.0, kSigmas * stddev / std::sqrt(nd));
  EXPECT_NEAR(m2, var, kSigmas * var * std::sqrt(2.0 / nd));
  EXPECT_NEAR(m3 / std::pow(m2, 1.5), 0.0, kSigmas * std::sqrt(6.0 / nd));
  EXPECT_NEAR(m4 / (m2 * m2) - 3.0, 0.0, kSigmas * std::sqrt(24.0 / nd));
}

void expectNormalCdf(const float * x, size_t n, double stddev, double max_t)
{
  // Empirical CDF in the body and both tails, and the two-sided tail fractions, up to max_t
  // standard deviations where the expected counts are large enough for the normal approximation
  const double nd = static_cast<double>(n);
  for (double t = -max_t; t <= max_t; t += 0.5) {
    const double threshold = t * stddev;
    size_t below = 0;
    for (size_t i = 0; i < n; ++i) {
      below += x[i] <= threshold;
    }
    const double p = normalCdf(t);
    EXPECT_NEAR(below / nd, p, kSigmas * std::sqrt(p * (1.0 - p) / nd)) << "t = " << t;
  }
  for (double t = 1.0; t <= max_t; t += 1.0) {
    size_t outside = 0;
    for (size_t i = 0; i < n; ++i) {
      outside += std::abs(x[i]) > t * stddev;
    }
    const double p = std::erfc(t / std::sqrt(2.0));
    EXPECT_NEAR(outside / nd, p, kSigmas * std::sqrt(p * (1.0 - p) / nd)) << "t = " << t;
  }
}

TEST(NoiseGeneratorTest, NormalSamplerDistribution)
{
  // 4M samples with seed 42 (multiple blocks and a partial last block)
  const size_t n = 4000001;
  std::vector<float> x(n);
  NormalSampler sampler(42u);
  sampler.fill(x.data(), n, 1.0f);
  expectNormal(x.data(), n, 1.0);
  expectNormalCdf(x.data(), n, 1.0, 4.0);

  // Each output position of the Box-Muller pairs is normal on its own, not only pooled:
  // the first 256 samples of every 512 sample block are the first outputs of the pairs
  std::vector<float> first, second;
  for (size_t i = 0; i < n; ++i) {
    ((i % 512) < 256 ? first : second).push_back(x[i]);
  }
  expectNormal(first.data(), first.size(), 1.0);
  expectNormal(second.data(), second.size(), 1.0);
  expectNormalCdf(first.data(), first.size(), 1.0, 3.0);
  expectNormalCdf(second.data(), second.size(), 1.0, 3.0);

  // The tails are not truncated: P(max |x| < 4.5) = (1 - 6.8e-6)^n < 1e-11
  float max_abs = 0.0f;
  for (float v : x) {
    max_abs = std::max(max_abs, std::abs(v));
  }
  EXPECT_GT(max_abs, 4.5f);

  // Scaled standard deviation
  sampler.fill(x.data(), n, 0.3f);
  expectNormal(x.data(), n, 0.3);
}

TEST(NoiseGeneratorTest, NormalSamplerIndependence)
{
  const size_t n = 4000000;
  std::vector<float> x(n);
  NormalSampler sampler(7u);
  sampler.fill(x.data(), n, 1.0f);

  // Lags 1 and 2 cover consecutive samples, 256 the two outputs of each Box-Muller pair,
  // 512 consecutive blocks, 2000 consecutive time steps of a noise array
  const double tol = kSigmas / std::sqrt(static_cast<double>(n));
  for (size_t lag : {1u, 2u, 255u, 256u, 257u, 512u, 2000u}) {
    EXPECT_NEAR(lagCorrelation(x, lag), 0.0, tol) << "lag " << lag;
    // corr(x^2, y^2) of independent normals has standard error 1 / sqrt(n) as well
    EXPECT_NEAR(lagSquaredCorrelation(x, lag), 0.0, tol) << "lag " << lag;
  }

  // Signs of the two outputs of a pair are independent
  size_t same_sign = 0;
  for (size_t b = 0; b + 512 <= n; b += 512) {
    for (size_t k = 0; k < 256; ++k) {
      same_sign += (x[b + k] > 0.0f) == (x[b + k + 256] > 0.0f);
    }
  }
  const double pairs = static_cast<double>(n / 2);
  EXPECT_NEAR(same_sign / pairs, 0.5, kSigmas * 0.5 / std::sqrt(pairs));
}

TEST(NoiseGeneratorTest, NormalSamplerDeterminismAndSizes)
{
  NormalSampler a(3u), b(3u), c(4u);
  std::vector<float> xa(1000), xb(1000), xc(1000);
  a.fill(xa.data(), 1000, 1.0f);
  b.fill(xb.data(), 1000, 1.0f);
  c.fill(xc.data(), 1000, 1.0f);
  EXPECT_EQ(xa, xb);
  EXPECT_NE(xa, xc);

  // Reseeding restarts the stream
  a.seed(3u);
  a.fill(xa.data(), 1000, 1.0f);
  EXPECT_EQ(xa, xb);

  // Partial blocks write exactly the requested samples
  for (Eigen::Index size : {0, 1, 2, 511, 512, 513, 1025}) {
    std::vector<float> buf(size + 2, 1234.0f);
    a.fill(buf.data() + 1, size, 1.0f);
    EXPECT_EQ(buf.front(), 1234.0f);
    EXPECT_EQ(buf.back(), 1234.0f);
    for (Eigen::Index i = 1; i <= size; ++i) {
      EXPECT_TRUE(std::isfinite(buf[i]));
      EXPECT_NE(buf[i], 1234.0f);
    }
  }

  // Zero standard deviation gives zero noise
  std::vector<float> zeros(600, 1.0f);
  a.fill(zeros.data(), 600, 0.0f);
  for (float v : zeros) {
    EXPECT_EQ(v, 0.0f);
  }
}

class NoiseGeneratorWrapper : public NoiseGenerator
{
public:
  using NoiseGenerator::noises_vx_;
  using NoiseGenerator::noises_vy_;
  using NoiseGenerator::noises_wz_;

  void generate()
  {
    generateNoisedControls();
  }
};

TEST(NoiseGeneratorTest, NoiseArraysStatistics)
{
  // Robot sized noise arrays: configured standard deviations, no correlation along batches,
  // along time steps, between the axes, or between consecutive generations
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  std::string name = "test";
  ParametersHandler handler(node, name);
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 2000;
  settings.time_steps = 100;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.4;

  NoiseGeneratorWrapper generator;
  generator.initialize(settings, true, "test_name", &handler);
  generator.reset(settings, true);

  const size_t n = 2000 * 100;
  ASSERT_EQ(static_cast<size_t>(generator.noises_vx_.size()), n);
  ASSERT_EQ(static_cast<size_t>(generator.noises_vy_.size()), n);
  ASSERT_EQ(static_cast<size_t>(generator.noises_wz_.size()), n);

  expectNormal(generator.noises_vx_.data(), n, 0.2);
  expectNormal(generator.noises_vy_.data(), n, 0.1);
  expectNormal(generator.noises_wz_.data(), n, 0.4);
  expectNormalCdf(generator.noises_wz_.data(), n, 0.4, 3.0);

  const double tol = kSigmas / std::sqrt(static_cast<double>(n));
  for (const Eigen::ArrayXXf * noises :
    {&generator.noises_vx_, &generator.noises_vy_, &generator.noises_wz_})
  {
    // Neighbours along a column (batch) and along a row (time step)
    const auto & a = *noises;
    const Eigen::ArrayXXf batch_next = a.bottomRows(1999);
    const Eigen::ArrayXXf batch_prev = a.topRows(1999);
    EXPECT_NEAR(correlation(batch_prev.data(), batch_next.data(), batch_prev.size()), 0.0, tol);
    const Eigen::ArrayXXf time_next = a.rightCols(99);
    const Eigen::ArrayXXf time_prev = a.leftCols(99);
    EXPECT_NEAR(correlation(time_prev.data(), time_next.data(), time_prev.size()), 0.0, tol);
  }
  EXPECT_NEAR(
    correlation(generator.noises_vx_.data(), generator.noises_wz_.data(), n), 0.0, tol);
  EXPECT_NEAR(
    correlation(generator.noises_vx_.data(), generator.noises_vy_.data(), n), 0.0, tol);
  EXPECT_NEAR(
    correlation(generator.noises_vy_.data(), generator.noises_wz_.data(), n), 0.0, tol);

  const Eigen::ArrayXXf previous_vx = generator.noises_vx_;
  generator.generate();
  expectNormal(generator.noises_vx_.data(), n, 0.2);
  EXPECT_NEAR(correlation(previous_vx.data(), generator.noises_vx_.data(), n), 0.0, tol);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, NonHolonomicLeavesVyUntouched)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  std::string name = "test";
  ParametersHandler handler(node, name);
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  NoiseGeneratorWrapper generator;
  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);
  EXPECT_TRUE((generator.noises_vy_ == 0.0f).all());
  EXPECT_FALSE((generator.noises_vx_ == 0.0f).all());
  EXPECT_FALSE((generator.noises_wz_ == 0.0f).all());

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(25);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);
  generator.setNoisedControls(state, control_sequence);
  EXPECT_TRUE((state.cvy == 0.0f).all());
  EXPECT_TRUE(state.cvx.isApprox(generator.noises_vx_));
  EXPECT_TRUE(state.cwz.isApprox(generator.noises_wz_));
  generator.shutdown();
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  rclcpp::init(0, nullptr);

  int result = RUN_ALL_TESTS();

  rclcpp::shutdown();

  return result;
}
