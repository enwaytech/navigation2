// Copyright (c) 2026, Dexory (Tony Najjar)
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

#include "nav2_costmap_2d/inflation_layer.hpp"

#include <limits>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "nav2_costmap_2d/distance_transform.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/parameter_events_filter.hpp"

PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::InflationLayer, nav2_costmap_2d::Layer)

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;
using rcl_interfaces::msg::ParameterType;

namespace nav2_costmap_2d
{

InflationLayer::InflationLayer()
: inflation_radius_(0),
  inscribed_radius_(0),
  cost_scaling_factor_(0),
  inflate_unknown_(false),
  inflate_around_unknown_(false),
  cell_inflation_radius_(0),
  num_threads_(-1),
  resolution_(0),
  last_min_x_(std::numeric_limits<double>::lowest()),
  last_min_y_(std::numeric_limits<double>::lowest()),
  last_max_x_(std::numeric_limits<double>::max()),
  last_max_y_(std::numeric_limits<double>::max())
{
  access_ = new mutex_t();
}

InflationLayer::~InflationLayer()
{
  delete access_;
}

void
InflationLayer::onInitialize()
{
  {
    auto node = node_.lock();
    if (!node) {
      throw std::runtime_error{"Failed to lock node"};
    }
    enabled_ = node->declare_or_get_parameter(name_ + "." + "enabled", true);
    inflation_radius_ = node->declare_or_get_parameter(name_ + "." + "inflation_radius", 0.55);
    cost_scaling_factor_ = node->declare_or_get_parameter(
      name_ + "." + "cost_scaling_factor", 10.0);
    inflate_unknown_ = node->declare_or_get_parameter(name_ + "." + "inflate_unknown", false);
    inflate_around_unknown_ = node->declare_or_get_parameter(
      name_ + "." + "inflate_around_unknown", false);
    num_threads_ = node->declare_or_get_parameter(
      name_ + "." + "num_threads", -1);
  }

  current_ = true;
  need_reinflation_ = false;
  matchSize();
}

void InflationLayer::activate()
{
  auto node = node_.lock();
  post_set_params_handler_ = node->add_post_set_parameters_callback(
    std::bind(
      &InflationLayer::updateParametersCallback,
      this, std::placeholders::_1));
  on_set_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(
      &InflationLayer::validateParameterUpdatesCallback,
      this, std::placeholders::_1));
}

void InflationLayer::deactivate()
{
  auto node = node_.lock();
  if (post_set_params_handler_ && node) {
    node->remove_post_set_parameters_callback(post_set_params_handler_.get());
  }
  post_set_params_handler_.reset();
  if (on_set_params_handler_ && node) {
    node->remove_on_set_parameters_callback(on_set_params_handler_.get());
  }
  on_set_params_handler_.reset();
}

void
InflationLayer::matchSize()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  nav2_costmap_2d::Costmap2D * costmap = layered_costmap_->getCostmap();
  resolution_ = costmap->getResolution();
  cell_inflation_radius_ = cellDistance(inflation_radius_);
  computeCaches();
}

void
InflationLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/, double * min_x,
  double * min_y, double * max_x, double * max_y)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (need_reinflation_) {
    // Reset last_* to "no expansion" values so the next cycle won't
    // merge with these full-map bounds (avoids double full-map update after reset)
    last_min_x_ = last_min_y_ = std::numeric_limits<double>::max();
    last_max_x_ = last_max_y_ = std::numeric_limits<double>::lowest();

    *min_x = std::numeric_limits<double>::lowest();
    *min_y = std::numeric_limits<double>::lowest();
    *max_x = std::numeric_limits<double>::max();
    *max_y = std::numeric_limits<double>::max();
    need_reinflation_ = false;
  } else {
    double tmp_min_x = last_min_x_;
    double tmp_min_y = last_min_y_;
    double tmp_max_x = last_max_x_;
    double tmp_max_y = last_max_y_;
    last_min_x_ = *min_x;
    last_min_y_ = *min_y;
    last_max_x_ = *max_x;
    last_max_y_ = *max_y;
    *min_x = std::min(tmp_min_x, *min_x) - inflation_radius_;
    *min_y = std::min(tmp_min_y, *min_y) - inflation_radius_;
    *max_x = std::max(tmp_max_x, *max_x) + inflation_radius_;
    *max_y = std::max(tmp_max_y, *max_y) + inflation_radius_;
  }
}

int
InflationLayer::getOptimalThreadCount()
{
#ifdef _OPENMP
  // If num_threads parameter is explicitly set (> 0), use it
  if (num_threads_ > 0) {
    RCLCPP_INFO_ONCE(
      logger_,
      "OpenMP: Using configured num_threads: %d",
      num_threads_);
    return num_threads_;
  }

  // Otherwise use auto-detection: half the available cores for memory-bound algorithms
  // Balances performance with memory bandwidth and safety on constrained systems
  // Respects OMP_NUM_THREADS environment variable
  int cpu_cores = omp_get_max_threads();
  int optimal = std::max(1, cpu_cores / 2);

  RCLCPP_INFO_ONCE(
    logger_,
    "OpenMP: %d cores available, using %d threads (auto)",
    cpu_cores, optimal);

  return optimal;
#else
  return 1;
#endif
}

void
InflationLayer::onFootprintChanged()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  inscribed_radius_ = layered_costmap_->getInscribedRadius();
  cell_inflation_radius_ = cellDistance(inflation_radius_);
  computeCaches();
  need_reinflation_ = true;

  if (inflation_radius_ < inscribed_radius_) {
    RCLCPP_ERROR(
      logger_,
      "The configured inflation radius (%.3f) is smaller than "
      "the computed inscribed radius (%.3f) of your footprint, "
      "it is highly recommended to set inflation radius to be at "
      "least as big as the inscribed radius to avoid collisions",
      inflation_radius_, inscribed_radius_);
  }

  RCLCPP_DEBUG(
    logger_, "InflationLayer::onFootprintChanged(): num footprint points: %zu,"
    " inscribed_radius_ = %.3f, inflation_radius_ = %.3f",
    layered_costmap_->getFootprint().size(), inscribed_radius_, inflation_radius_);
}

void
InflationLayer::applyInflation(
  unsigned char * master_array,
  const MatrixXfRM & distance_map,
  int min_i, int min_j, int max_i, int max_j,
  int roi_min_i, int roi_min_j,
  unsigned int size_x)
{
  const float cell_inflation_radius_f = static_cast<float>(cell_inflation_radius_);
  const int lut_max = static_cast<int>(cost_lut_.size() - 1);
  const unsigned char * lut_data = cost_lut_.data();
  const int lut_precision = COST_LUT_PRECISION;
  const bool inflate_unk = inflate_unknown_;

#ifdef _OPENMP
  const int num_threads = getOptimalThreadCount();
  #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
#endif
  for (int j = min_j; j < max_j; ++j) {
    const int row_offset = j * static_cast<int>(size_x);
    const int dist_row = j - roi_min_j;

    for (int i = min_i; i < max_i; ++i) {
      const float distance_cells = distance_map(dist_row, i - roi_min_i);
      if (distance_cells > cell_inflation_radius_f) {
        continue;
      }

      const unsigned int index = row_offset + i;
      const unsigned char old_cost = master_array[index];
      const unsigned int d_scaled = std::min(
        static_cast<unsigned int>(lut_max),
        static_cast<unsigned int>(distance_cells * lut_precision + 0.5f));
      const unsigned char cost = lut_data[d_scaled];

      if (old_cost == NO_INFORMATION &&
        (inflate_unk ? (cost > FREE_SPACE) : (cost >= INSCRIBED_INFLATED_OBSTACLE)))
      {
        master_array[index] = cost;
      } else {
        master_array[index] = std::max(old_cost, cost);
      }
    }
  }
}

void
InflationLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j,
  int max_i,
  int max_j)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!enabled_ || (cell_inflation_radius_ == 0)) {
    return;
  }

  unsigned char * master_array = master_grid.getCharMap();
  const unsigned int size_x = master_grid.getSizeInCellsX();
  const unsigned int size_y = master_grid.getSizeInCellsY();

  min_i = std::max(0, min_i);
  min_j = std::max(0, min_j);
  max_i = std::min(static_cast<int>(size_x), max_i);
  max_j = std::min(static_cast<int>(size_y), max_j);

  // Compute padded ROI bounds for distance transform
  const int padding = static_cast<int>(cell_inflation_radius_);
  int roi_min_i = std::max(0, min_i - padding);
  int roi_min_j = std::max(0, min_j - padding);
  int roi_max_i = std::min(static_cast<int>(size_x), max_i + padding);
  int roi_max_j = std::min(static_cast<int>(size_y), max_j + padding);

  const int roi_width = roi_max_i - roi_min_i;
  const int roi_height = roi_max_j - roi_min_j;

  // Distance map of the ROI (Felzenszwalb-Huttenlocher distance transform with obstacles = 0 and
  // free space = INF, see DistanceTransform::distanceTransform2D())
  constexpr float inf = DistanceTransform::DT_INF;
  MatrixXfRM distance_map(roi_height, roi_width);

  // Column pass. The 1D distance transform of a column of obstacles (0) and free cells (INF) is the
  // squared distance to the nearest obstacle cell of the column (INF without one): computed by a
  // forward and a backward sweep over blocks of columns, with the obstacles read from the costmap.
  constexpr int block_width = 64;
  const int num_blocks = (roi_width + block_width - 1) / block_width;
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int block = 0; block < num_blocks; ++block) {
    const int x_begin = block * block_width;
    const int x_end = std::min(x_begin + block_width, roi_width);
    // Row of the nearest obstacle cell of each column so far, -1 for none
    int nearest[block_width];

    // Distance to the nearest obstacle cell at or above
    std::fill_n(nearest, block_width, -1);
    for (int y = 0; y < roi_height; ++y) {
      const unsigned char * cells = master_array + (y + roi_min_j) * size_x + roi_min_i;
      float * row = &distance_map(y, 0);
      for (int x = x_begin; x < x_end; ++x) {
        const unsigned char cell = cells[x];
        if (cell == LETHAL_OBSTACLE || (inflate_around_unknown_ && cell == NO_INFORMATION)) {
          nearest[x - x_begin] = y;
        }
        row[x] = nearest[x - x_begin] < 0 ? inf : static_cast<float>(y - nearest[x - x_begin]);
      }
    }

    // Minimum with the distance to the nearest obstacle cell below, squared
    constexpr int no_obstacle = std::numeric_limits<int>::max();
    std::fill_n(nearest, block_width, -1);
    for (int y = roi_height - 1; y >= 0; --y) {
      float * row = &distance_map(y, 0);
      for (int x = x_begin; x < x_end; ++x) {
        const float distance_above = row[x];
        if (distance_above == 0.0f) {
          nearest[x - x_begin] = y;
        }
        int distance = distance_above == inf ? no_obstacle : static_cast<int>(distance_above);
        if (nearest[x - x_begin] >= 0) {
          distance = std::min(distance, nearest[x - x_begin] - y);
        }
        row[x] = distance == no_obstacle ? inf : static_cast<float>(distance * distance);
      }
    }
  }

  // Row pass, only needed for the rows of the update window, and the Euclidean distances of the
  // update window cells. Cells beyond the inflation radius get INF, which applyInflation() skips
  // like any distance beyond it.
  const float cell_inflation_radius_f = static_cast<float>(cell_inflation_radius_);
  // Largest squared distance whose square root is not beyond the inflation radius
  float max_distance_sq = cell_inflation_radius_f * cell_inflation_radius_f;
  while (std::sqrt(max_distance_sq) > cell_inflation_radius_f) {
    max_distance_sq = std::nextafter(max_distance_sq, 0.0f);
  }
  while (std::sqrt(std::nextafter(max_distance_sq, inf)) <= cell_inflation_radius_f) {
    max_distance_sq = std::nextafter(max_distance_sq, inf);
  }
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    std::vector<float> d(roi_width);
    std::vector<int> v(roi_width);
    std::vector<float> z(roi_width + 1);
#ifdef _OPENMP
    #pragma omp for schedule(dynamic, 16)
#endif
    for (int j = min_j; j < max_j; ++j) {
      float * row = &distance_map(j - roi_min_j, 0);
      DistanceTransform::distanceTransform1D(row, d.data(), roi_width, v.data(), z.data());
      for (int i = min_i - roi_min_i; i < max_i - roi_min_i; ++i) {
        row[i] = d[i] > max_distance_sq ? inf : std::sqrt(d[i]);
      }
    }
  }

  // Apply inflation costs
  applyInflation(
    master_array, distance_map,
    min_i, min_j, max_i, max_j,
    roi_min_i, roi_min_j, size_x);

  current_ = true;
}


void
InflationLayer::computeCaches()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (cell_inflation_radius_ == 0) {
    return;
  }

  // Generate cost lookup table for distance -> cost mapping
  const unsigned int max_dist_scaled = cell_inflation_radius_ * COST_LUT_PRECISION + 1;
  cost_lut_.resize(max_dist_scaled + 1);
  for (unsigned int d_scaled = 0; d_scaled <= max_dist_scaled; ++d_scaled) {
    const double distance = static_cast<double>(d_scaled) / COST_LUT_PRECISION;
    cost_lut_[d_scaled] = computeCost(distance);
  }
}


rcl_interfaces::msg::SetParametersResult InflationLayer::validateParameterUpdatesCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & parameter : parameters) {
    const auto & param_type = parameter.get_type();
    const auto & param_name = parameter.get_name();
    if (param_name.find(name_ + ".") != 0) {
      continue;
    }
    if (param_type == ParameterType::PARAMETER_DOUBLE) {
      if (parameter.as_double() < 0.0) {
        RCLCPP_WARN(
        logger_, "The value of parameter '%s' is incorrectly set to %f, "
        "it should be >=0. Ignoring parameter update.",
        param_name.c_str(), parameter.as_double());
        result.successful = false;
      }
    }
  }
  return result;
}

void
InflationLayer::updateParametersCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());

  bool need_cache_recompute = false;

  for (const auto & parameter : parameters) {
    const auto & param_type = parameter.get_type();
    const auto & param_name = parameter.get_name();
    if (param_name.find(name_ + ".") != 0) {
      continue;
    }

    if (param_type == ParameterType::PARAMETER_DOUBLE) {
      if (param_name == name_ + "." + "inflation_radius" &&
        inflation_radius_ != parameter.as_double())
      {
        inflation_radius_ = parameter.as_double();
        need_reinflation_ = true;
        need_cache_recompute = true;
        current_ = false;
      } else if (param_name == name_ + "." + "cost_scaling_factor" && // NOLINT
        getCostScalingFactor() != parameter.as_double())
      {
        cost_scaling_factor_ = parameter.as_double();
        need_reinflation_ = true;
        need_cache_recompute = true;
        current_ = false;
      }
    } else if (param_type == ParameterType::PARAMETER_INTEGER) {
      if (param_name == name_ + "." + "num_threads" && // NOLINT
        num_threads_ != parameter.as_int())
      {
        int new_value = parameter.as_int();
#ifdef _OPENMP
        if (new_value < -1) {
          RCLCPP_WARN(
            logger_,
            "Invalid num_threads value %d, must be -1 (auto) or > 0. Ignoring.",
            new_value);
        } else {
          int available_cores = omp_get_max_threads();
          if (new_value > available_cores) {
            RCLCPP_WARN(
              logger_,
              "num_threads=%d exceeds available cores (%d). Ignoring.",
              new_value, available_cores);
          } else {
            num_threads_ = new_value;
            RCLCPP_INFO(
              logger_,
              "Updated num_threads to %d %s",
              num_threads_,
              num_threads_ == -1 ? "(auto)" : "");
          }
        }
#else
        RCLCPP_WARN(
          logger_,
          "num_threads parameter ignored - OpenMP support not available. "
          "Inflation layer will use single thread.");
        num_threads_ = new_value;
#endif
      }
    } else if (param_type == ParameterType::PARAMETER_BOOL) {
      if (param_name == name_ + "." + "enabled" && enabled_ != parameter.as_bool()) {
        enabled_ = parameter.as_bool();
        need_reinflation_ = true;
        current_ = false;
      } else if (param_name == name_ + "." + "inflate_unknown" && // NOLINT
        inflate_unknown_ != parameter.as_bool())
      {
        inflate_unknown_ = parameter.as_bool();
        need_reinflation_ = true;
        current_ = false;
      } else if (param_name == name_ + "." + "inflate_around_unknown" && // NOLINT
        inflate_around_unknown_ != parameter.as_bool())
      {
        inflate_around_unknown_ = parameter.as_bool();
        need_reinflation_ = true;
        current_ = false;
      }
    }
  }

  if (need_cache_recompute) {
    matchSize();
  }
}

}  // namespace nav2_costmap_2d
