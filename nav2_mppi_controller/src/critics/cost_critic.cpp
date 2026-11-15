// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "nav2_mppi_controller/critics/cost_critic.hpp"
#include "nav2_costmap_2d/inflation_layer_interface.hpp"
#include "nav2_core/controller_exceptions.hpp"

namespace mppi::critics
{

void CostCritic::initialize()
{
  auto getParentParam = parameters_handler_->getParamGetter(parent_name_);
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(consider_footprint_, "consider_footprint", false);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 3.81f);
  getParam(critical_cost_, "critical_cost", 300.0f);
  getParam(near_collision_cost_, "near_collision_cost", 253);
  getParam(collision_cost_, "collision_cost", 1000000.0f);
  getParam(near_goal_distance_, "near_goal_distance", 0.5f);
  getParam(inflation_layer_name_, "inflation_layer_name", std::string(""));
  getParam(num_threads_, "num_threads", -1);
  getParam(trajectory_point_step_, "trajectory_point_step", 2);
  getParam(use_footprint_prefilter_, "use_footprint_prefilter", true);

  // Normalized by cost value to put in same regime as other weights
  weight_ /= 254.0f;

  // Normalize weight when parameter is changed dynamically as well
  auto weightDynamicCb = [&](
    const rclcpp::Parameter & weight) {
      weight_ = weight.as_double() / 254.0f;
    };
  parameters_handler_->addParamCallback(name_ + ".cost_weight", weightDynamicCb);

  collision_checker_.setCostmap(costmap_);
  possible_collision_cost_ = findCircumscribedCost(costmap_ros_);

  if (possible_collision_cost_ < 1.0f) {
    RCLCPP_ERROR(
      logger_,
      "Inflation layer either not found or inflation is not set sufficiently for "
      "optimized non-circular collision checking capabilities. It is HIGHLY recommended to set"
      " the inflation radius to be at MINIMUM half of the robot's largest cross-section. See "
      "github.com/ros-planning/navigation2/tree/main/nav2_smac_planner#potential-fields"
      " for full instructions. This will substantially impact run-time performance.");
  }

  if (costmap_ros_->getUseRadius() == consider_footprint_) {
    RCLCPP_WARN(
      logger_,
      "Inconsistent configuration in collision checking. Please verify the robot's shape settings "
      "in both the costmap and the cost critic.");
    if (costmap_ros_->getUseRadius()) {
      throw nav2_core::ControllerException(
              "Considering footprint in CostCritic but no robot footprint provided in the "
              "costmap (robot radius used instead). Disable considering footprint.");
    }
  }

  if (near_collision_cost_ > 253) {
    RCLCPP_WARN(logger_, "Near collision cost is set higher than INSCRIBED_INFLATED_OBSTACLE");
  }

  RCLCPP_INFO(
    logger_,
    "InflationCostCritic instantiated with %d power and %f / %f weights. "
    "Critic will collision check based on %s cost.",
    power_, critical_cost_, weight_, consider_footprint_ ?
    "footprint" : "circular");
}

float CostCritic::findCircumscribedCost(
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap)
{
  double result = -1.0;
  const double circum_radius = costmap->getLayeredCostmap()->getCircumscribedRadius();
  if (static_cast<float>(circum_radius) == circumscribed_radius_) {
    // early return if footprint size is unchanged
    return circumscribed_cost_;
  }

  // check if the costmap has an inflation layer
  const auto inflation_layer = nav2_costmap_2d::InflationLayerInterface::getInflationLayer(
    costmap,
    inflation_layer_name_);
  if (inflation_layer != nullptr) {
    const double resolution = costmap->getCostmap()->getResolution();
    double inflation_radius = inflation_layer->getInflationRadius();
    if (inflation_radius < circum_radius) {
      RCLCPP_ERROR(
        rclcpp::get_logger("computeCircumscribedCost"),
        "The inflation radius (%f) is smaller than the circumscribed radius (%f) "
        "If this is an SE2-collision checking plugin, it cannot use costmap potential "
        "field to speed up collision checking by only checking the full footprint "
        "when robot is within possibly-inscribed radius of an obstacle. This may "
        "significantly slow down planning times!",
        inflation_radius, circum_radius);
      result = 0.0;
      return result;
    }
    result = inflation_layer->computeCost(circum_radius / resolution);
  } else {
    RCLCPP_WARN(
      logger_,
      "No inflation layer found in costmap configuration. "
      "If this is an SE2-collision checking plugin, it cannot use costmap potential "
      "field to speed up collision checking by only checking the full footprint "
      "when robot is within possibly-inscribed radius of an obstacle. This may "
      "significantly slow down planning times and not avoid anything but absolute collisions!");
  }

  circumscribed_radius_ = static_cast<float>(circum_radius);
  circumscribed_cost_ = static_cast<float>(result);

  return circumscribed_cost_;
}

int CostCritic::getOptimalThreadCount()
{
#ifdef _OPENMP
  if (num_threads_ > 0) {
    RCLCPP_INFO_ONCE(
      logger_,
      "OpenMP: Using configured num_threads: %d",
      num_threads_);
    return num_threads_;
  }
  // Auto (-1): use half of available cores — costmap lookups are memory-bound,
  // so more threads hit diminishing returns and increase cache contention.
  // omp_get_max_threads() respects OMP_NUM_THREADS if set.
  const int cpu_cores = omp_get_max_threads();
  const int optimal = std::max(1, cpu_cores / 2);
  RCLCPP_INFO_ONCE(
    logger_,
    "OpenMP: %d cores available, using %d threads (auto)",
    cpu_cores, optimal);

  return std::max(1, optimal);
#else
  return 1;
#endif
}

void CostCritic::updateCostmapInfo()
{
  is_tracking_unknown_ = costmap_ros_->getLayeredCostmap()->isTrackingUnknown();
  auto * costmap = collision_checker_.getCostmap();
  origin_x_ = static_cast<float>(costmap->getOriginX());
  origin_y_ = static_cast<float>(costmap->getOriginY());
  resolution_ = static_cast<float>(costmap->getResolution());
  size_x_ = costmap->getSizeInCellsX();
  size_y_ = costmap->getSizeInCellsY();
  map_origin_x_ = costmap->getOriginX();
  map_origin_y_ = costmap->getOriginY();
  map_resolution_ = costmap->getResolution();
  char_map_ = costmap->getCharMap();
}

void CostCritic::updateBlockedCellDistance(
  const CandidateBounds & bounds, const nav2_costmap_2d::Footprint & footprint)
{
  // No candidate with finite coordinates
  if (!(bounds.min_x <= bounds.max_x && bounds.min_y <= bounds.max_y)) {
    return;
  }

  // Every vertex of a footprint centered in bounds is within the footprint radius of it, and
  // the prefilter reads the distances up to its reach. Two cells more cover the rounding of
  // the cells.
  double radius = prefilter_active_ ? prefilter_reach_ : 0.0;
  for (const auto & vertex : footprint) {
    radius = std::max(radius, std::hypot(vertex.x, vertex.y));
  }
  const double margin = radius + 2.0 * map_resolution_;
  auto cell = [this](double world, double origin, unsigned int size) {
      const double max_cell = static_cast<double>(size) - 1.0;
      return static_cast<int>(
        std::clamp(std::floor((world - origin) / map_resolution_), 0.0, max_cell));
    };
  updateBlockedCellDistance(
    cell(bounds.min_x - margin, map_origin_x_, size_x_),
    cell(bounds.min_y - margin, map_origin_y_, size_y_),
    cell(bounds.max_x + margin, map_origin_x_, size_x_),
    cell(bounds.max_y + margin, map_origin_y_, size_y_));
}

void CostCritic::updateBlockedCellDistance(int x_min, int y_min, int x_max, int y_max)
{
  // Two pass chessboard distance transform (Rosenfeld and Pfaltz), exact for this metric. Each
  // pass first takes the neighbors of the previous row, then the one of the same row. Region
  // sides inside of the map are bordered by virtual blocked cells, which keeps the result a
  // lower bound of the distance in the whole map.
  const size_t stride = size_x_;
  blocked_cell_distance_.resize(stride * size_y_);
  // saturating increment
  auto next = [](uint8_t d) {return static_cast<uint8_t>(std::min<uint8_t>(d, 254u) + 1u);};
  auto min3 = [](uint8_t a, uint8_t b, uint8_t c) {return std::min(std::min(a, b), c);};
  const size_t x0 = static_cast<size_t>(x_min);
  const size_t x1 = static_cast<size_t>(x_max);
  const uint8_t left_border = x_min > 0 ? 1u : 255u;
  const uint8_t right_border = x_max + 1 < static_cast<int>(size_x_) ? 1u : 255u;
  const uint8_t top_border = y_min > 0 ? 1u : 255u;
  const uint8_t bottom_border = y_max + 1 < static_cast<int>(size_y_) ? 1u : 255u;

  for (int y = y_min; y <= y_max; ++y) {
    const unsigned char * cost = char_map_ + y * stride;
    uint8_t * row = blocked_cell_distance_.data() + y * stride;
    if (y == y_min) {
      std::fill(row + x0, row + x1 + 1, top_border);
    } else {
      const uint8_t * prev = row - stride;
      row[x0] = next(x0 < x1 ? std::min(prev[x0], prev[x0 + 1]) : prev[x0]);
      for (size_t x = x0 + 1; x < x1; ++x) {
        row[x] = next(min3(prev[x - 1], prev[x], prev[x + 1]));
      }
      if (x1 > x0) {
        row[x1] = next(std::min(prev[x1 - 1], prev[x1]));
      }
    }
    if (y == y_max) {
      for (size_t x = x0; x <= x1; ++x) {
        row[x] = std::min(row[x], bottom_border);
      }
    }
    row[x0] = std::min(row[x0], left_border);
    row[x1] = std::min(row[x1], right_border);
    for (size_t x = x0; x <= x1; ++x) {
      row[x] = cost[x] >= nav2_costmap_2d::LETHAL_OBSTACLE ? 0u : row[x];
    }
    for (size_t x = x0 + 1; x <= x1; ++x) {
      row[x] = std::min(row[x], next(row[x - 1]));
    }
  }

  for (int y = y_max; y >= y_min; --y) {
    uint8_t * row = blocked_cell_distance_.data() + y * stride;
    if (y < y_max) {
      const uint8_t * below = row + stride;
      row[x0] = std::min(
        row[x0], next(x0 < x1 ? std::min(below[x0], below[x0 + 1]) : below[x0]));
      for (size_t x = x0 + 1; x < x1; ++x) {
        row[x] = std::min(row[x], next(min3(below[x - 1], below[x], below[x + 1])));
      }
      if (x1 > x0) {
        row[x1] = std::min(row[x1], next(std::min(below[x1 - 1], below[x1])));
      }
    }
    for (size_t x = x1; x-- > x0; ) {
      row[x] = std::min(row[x], next(row[x + 1]));
    }
  }
}

namespace
{

// Trajectory states between the center cost and the footprint passes
constexpr uint8_t kTrajectoryFree = 0;
constexpr uint8_t kTrajectoryCollides = 1;
constexpr uint8_t kTrajectoryCheckFootprint = 2;

// Branch free selection, as a conditional float operation is not vectorized with -ftrapping-math
inline float selectFloat(bool condition, float if_true, float if_false)
{
  uint32_t bits_true, bits_false;
  std::memcpy(&bits_true, &if_true, sizeof(float));
  std::memcpy(&bits_false, &if_false, sizeof(float));
  const uint32_t mask = 0u - static_cast<uint32_t>(condition);
  const uint32_t bits = (bits_true & mask) | (bits_false & ~mask);
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

}  // namespace

inline CostCritic::LineCost CostCritic::lineCost(int x0, int y0, int x1, int y1) const
{
  // Same cells as nav2_util::LineIterator: pixel p of the line is p steps along the major axis
  // and floor((den / 2 + p * num_add) / den) steps along the minor axis. Pixels k steps apart
  // are at chessboard distance k, so the pixels closer to a cell than its distance to the
  // nearest blocked cell are not blocked and are skipped.
  const std::ptrdiff_t row_step = static_cast<std::ptrdiff_t>(size_x_);
  const std::ptrdiff_t start = static_cast<std::ptrdiff_t>(y0) * row_step + x0;
  const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(y1) * row_step + x1;
  const uint8_t start_distance = blocked_cell_distance_[start];
  const int delta_x = std::abs(x1 - x0);
  const int delta_y = std::abs(y1 - y0);
  const int64_t den = std::max(delta_x, delta_y);
  // pixels after last_pixel are closer to the end cell than its distance to a blocked cell
  const int64_t last_pixel = den - blocked_cell_distance_[end];
  if (start_distance > last_pixel) {
    return LineCost::FREE;
  }

  bool unknown = false;
  int64_t pixel = start_distance;
  if (start_distance == 0) {
    if (char_map_[start] == nav2_costmap_2d::LETHAL_OBSTACLE) {
      return LineCost::LETHAL;
    }
    unknown = true;  // NO_INFORMATION
    pixel = 1;
  }

  const bool x_major = delta_x >= delta_y;
  const int64_t num_add = x_major ? delta_y : delta_x;
  const std::ptrdiff_t step_x = x1 >= x0 ? 1 : -1;
  const std::ptrdiff_t step_y = y1 >= y0 ? row_step : -row_step;
  const std::ptrdiff_t step_major = x_major ? step_x : step_y;
  const std::ptrdiff_t step_minor = x_major ? step_y : step_x;
  // den > 0 here, as pixel 0 is the end cell otherwise
  while (pixel <= last_pixel) {
    const int64_t minor = (den / 2 + pixel * num_add) / den;
    const std::ptrdiff_t index = start + pixel * step_major + minor * step_minor;
    const uint8_t distance = blocked_cell_distance_[index];
    if (distance == 0) {
      if (char_map_[index] == nav2_costmap_2d::LETHAL_OBSTACLE) {
        return LineCost::LETHAL;
      }
      unknown = true;  // NO_INFORMATION
      ++pixel;
    } else {
      pixel += distance;
    }
  }
  return unknown ? LineCost::UNKNOWN : LineCost::FREE;
}

bool CostCritic::footprintInCollision(
  float x, float y, float theta, const nav2_costmap_2d::Footprint & footprint)
{
  const double px = static_cast<double>(x);
  const double py = static_cast<double>(y);
  const double pt = static_cast<double>(theta);
  if (footprint.size() < 2) {
    // degenerate footprint: keep the reference implementation
    const auto cost = static_cast<unsigned char>(static_cast<float>(
        collision_checker_.footprintCostAtPose(px, py, pt, footprint)));
    return cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
           (cost == nav2_costmap_2d::NO_INFORMATION && !is_tracking_unknown_);
  }

  // FootprintCollisionChecker::footprintCost() returns LETHAL_OBSTACLE for a vertex outside of
  // the map. Otherwise it returns the cost of the first line, in order, holding a cell
  // >= LETHAL_OBSTACLE: LETHAL_OBSTACLE if that line holds one (lineCost() returns early on it),
  // NO_INFORMATION if not, as a lethal cell of a later line is then masked by the max().
  // If no line holds such a cell, the cost is below LETHAL_OBSTACLE: no collision.
  const double cos_th = cos(pt);
  const double sin_th = sin(pt);
  double wx, wy;
  unsigned int x0 = 0u, y0 = 0u, x1 = 0u, y1 = 0u;
  bool unknown_first = false;

  transformFootprintVertex(px, py, cos_th, sin_th, footprint[0], wx, wy);
  if (!worldToMapDouble(wx, wy, x0, y0)) {
    return true;
  }
  const unsigned int x_start = x0;
  const unsigned int y_start = y0;

  for (size_t i = 0; i + 1 < footprint.size(); ++i) {
    transformFootprintVertex(px, py, cos_th, sin_th, footprint[i + 1], wx, wy);
    if (!worldToMapDouble(wx, wy, x1, y1)) {
      return true;
    }
    if (!unknown_first) {
      const LineCost line = lineCost(
        static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(x1), static_cast<int>(y1));
      if (line == LineCost::LETHAL) {
        return true;
      }
      if (line == LineCost::UNKNOWN) {
        if (!is_tracking_unknown_) {
          return true;
        }
        // NO_INFORMATION is the result unless a later vertex is outside of the map
        unknown_first = true;
      }
    }
    x0 = x1;
    y0 = y1;
  }
  if (unknown_first) {
    return false;
  }

  // the line connecting the first point to the last point comes last
  const LineCost line = lineCost(
    static_cast<int>(x_start), static_cast<int>(y_start), static_cast<int>(x1),
    static_cast<int>(y1));
  return line == LineCost::LETHAL || (line == LineCost::UNKNOWN && !is_tracking_unknown_);
}

namespace
{

// Maximum number of covering circles of the footprint prefilter
constexpr int kMaxCoveringCircles = 8;

// Distance in cells between a rasterized footprint cell and the center cell of the circle
// covering it, beyond the circle radius: the cells of a Bresenham line are at most half a cell
// from the segment between the vertex cell centers, a vertex cell center is at most half a cell
// diagonal from the vertex, and the circle center is at most half a cell diagonal from its cell
// center. The last term covers floating point errors.
constexpr double kPrefilterGridSlack = 0.5 + 0.5 * M_SQRT2 + 0.5 * M_SQRT2 + 0.05;

// The prefilter is tried on kPrefilterProbe poses at a time. A prefilter call costs about a
// third of a footprint check, so when it clears fewer than a third of them it is paused for
// the next kPrefilterPause poses.
constexpr int kPrefilterProbe = 32;
constexpr int kPrefilterPause = 256;

}  // namespace

void CostCritic::computeCoveringCircles(
  const nav2_costmap_2d::Footprint & footprint, std::vector<double> & x,
  std::vector<double> & y, std::vector<double> & radius)
{
  x.clear();
  y.clear();
  radius.clear();
  if (footprint.empty()) {
    return;
  }

  double min_x = footprint[0].x, max_x = footprint[0].x;
  double min_y = footprint[0].y, max_y = footprint[0].y;
  for (const auto & vertex : footprint) {
    min_x = std::min(min_x, vertex.x);
    max_x = std::max(max_x, vertex.x);
    min_y = std::min(min_y, vertex.y);
    max_y = std::max(max_y, vertex.y);
  }
  // Circle centers are evenly spaced on the long axis of the bounding box, each covering the
  // part of the outline within its slab perpendicular to the axis
  const bool along_x = max_x - min_x >= max_y - min_y;
  auto along = [along_x](const geometry_msgs::msg::Point & p) {return along_x ? p.x : p.y;};
  const double axis_min = along_x ? min_x : min_y;
  const double length = along_x ? max_x - min_x : max_y - min_y;
  const double width = along_x ? max_y - min_y : max_x - min_x;
  const double across_center = along_x ? 0.5 * (min_y + max_y) : 0.5 * (min_x + max_x);
  const int count = std::clamp(
    static_cast<int>(std::ceil(2.0 * length / std::max(width, 1e-3))), 1, kMaxCoveringCircles);
  const double spacing = length / count;
  const double inf = std::numeric_limits<double>::infinity();

  for (int k = 0; k < count; ++k) {
    const double center = axis_min + (k + 0.5) * spacing;
    const double cx = along_x ? center : across_center;
    const double cy = along_x ? across_center : center;
    const double slab_min = k == 0 ? -inf : axis_min + k * spacing;
    const double slab_max = k == count - 1 ? inf : axis_min + (k + 1) * spacing;

    // The distance to the center is convex along an edge, so its maximum over the part of the
    // edge in the slab is at one of the ends of that part
    double r = 0.0;
    for (size_t i = 0; i < footprint.size(); ++i) {
      const auto & p = footprint[i];
      const auto & q = footprint[(i + 1) % footprint.size()];
      const double ap = along(p), aq = along(q);
      double t_min = 0.0, t_max = 1.0;
      if (ap == aq) {
        if (ap < slab_min || ap > slab_max) {
          continue;
        }
      } else {
        double t_a = (slab_min - ap) / (aq - ap);
        double t_b = (slab_max - ap) / (aq - ap);
        if (t_a > t_b) {
          std::swap(t_a, t_b);
        }
        t_min = std::max(t_min, t_a);
        t_max = std::min(t_max, t_b);
        if (t_min > t_max) {
          continue;
        }
      }
      for (const double t : {t_min, t_max}) {
        r = std::max(r, std::hypot(p.x + t * (q.x - p.x) - cx, p.y + t * (q.y - p.y) - cy));
      }
    }
    x.push_back(cx);
    y.push_back(cy);
    radius.push_back(r);
  }
}

void CostCritic::updateFootprintPrefilter(const nav2_costmap_2d::Footprint & footprint)
{
  prefilter_active_ = false;
  if (!use_footprint_prefilter_ || !consider_footprint_ || footprint.size() < 2) {
    return;
  }

  if (footprint != prefilter_footprint_) {
    computeCoveringCircles(
      footprint, prefilter_center_x_, prefilter_center_y_, prefilter_radius_);
    prefilter_footprint_ = footprint;
  }

  const size_t num_circles = prefilter_radius_.size();
  prefilter_offset_x_.resize(num_circles);
  prefilter_offset_y_.resize(num_circles);
  prefilter_min_distance_.resize(num_circles);
  float pose_border = 0.0f;
  double reach = 0.0;
  for (size_t k = 0; k < num_circles; ++k) {
    // A blocked cell covered by the circle is at most this many cells from its center cell.
    // The chessboard distance between two cells is at most their Euclidean distance, so a
    // center cell with a larger chessboard distance to the nearest blocked cell has none under
    // its circle.
    const double distance = prefilter_radius_[k] / map_resolution_ + kPrefilterGridSlack;
    const double min_distance = std::floor(distance) + 1.0;
    // beyond the saturation of blocked_cell_distance_
    if (min_distance > 254.0) {
      return;
    }
    prefilter_min_distance_[k] = static_cast<uint8_t>(min_distance);
    prefilter_offset_x_[k] = static_cast<float>(prefilter_center_x_[k] / map_resolution_);
    prefilter_offset_y_[k] = static_cast<float>(prefilter_center_y_[k] / map_resolution_);
    // A center cell ceil(distance) cells from the map border keeps the covered vertices in the
    // map. The center is at most its offset from the pose, one more cell covers the rounding.
    const double offset = std::hypot(prefilter_center_x_[k], prefilter_center_y_[k]);
    pose_border = std::max(
      pose_border, static_cast<float>(std::ceil(distance) + offset / map_resolution_ + 1.0));
    reach = std::max(reach, offset + min_distance * map_resolution_);
  }
  prefilter_pose_border_ = pose_border;
  prefilter_reach_ = reach;
  prefilter_active_ = true;
}

bool CostCritic::footprintCollisionFree(float x, float y, float theta) const
{
  // Pose in cells relative to the map origin. The errors of the float computations are tiny
  // compared to the margin of kPrefilterGridSlack.
  const float pose_x = static_cast<float>(
    (static_cast<double>(x) - map_origin_x_) / map_resolution_);
  const float pose_y = static_cast<float>(
    (static_cast<double>(y) - map_origin_y_) / map_resolution_);
  // keeps the circle centers far enough from the map border, see updateFootprintPrefilter()
  const float border = prefilter_pose_border_;
  const float max_x = static_cast<float>(size_x_) - border;
  const float max_y = static_cast<float>(size_y_) - border;
  if (!(pose_x >= border && pose_y >= border && pose_x < max_x && pose_y < max_y)) {
    return false;
  }

  const float cos_th = std::cos(theta);
  const float sin_th = std::sin(theta);
  bool collision_free = true;
  for (size_t k = 0; k < prefilter_min_distance_.size(); ++k) {
    const float cx = pose_x + prefilter_offset_x_[k] * cos_th - prefilter_offset_y_[k] * sin_th;
    const float cy = pose_y + prefilter_offset_x_[k] * sin_th + prefilter_offset_y_[k] * cos_th;
    const unsigned int index =
      getIndex(static_cast<unsigned int>(cx), static_cast<unsigned int>(cy));
    collision_free &= blocked_cell_distance_[index] >= prefilter_min_distance_[k];
  }
  return collision_free;
}

bool CostCritic::scoreCenterCosts(
  const models::Trajectories & trajectories, int block, int count, int strided_traj_cols,
  bool near_goal, Eigen::ArrayXf & repulsive_cost, CandidateBounds & bounds)
{
  // Trajectory arrays are column-major: point j of the strided trajectory i is at
  // j * outer_stride + i
  const Eigen::Index outer_stride = trajectories.x.rows() * trajectory_point_step_;
  const int first = block * kTrajectoryBlockSize;
  // local copies, as the flag stores could alias the members
  const float origin_x = origin_x_;
  const float origin_y = origin_y_;
  const float resolution = resolution_;
  const float size_x = static_cast<float>(size_x_);
  const float size_y = static_cast<float>(size_y_);
  const int row_size = static_cast<int>(size_x_);
  const unsigned char * char_map = char_map_;
  const bool check_footprint = consider_footprint_;
  // Integer thresholds: for an integer cost c, c >= ceil(t) is the same as c >= t
  const int possible_collision_cost = static_cast<int>(
    std::clamp(std::ceil(possible_collision_cost_), 0.0f, 256.0f));
  const bool always_check_footprint = possible_collision_cost_ < 1.0f;
  const bool unknown_collides = !is_tracking_unknown_;
  const int near_collision_cost = static_cast<int>(std::min(near_collision_cost_, 256u));
  const float critical_cost = critical_cost_;
  const float far_cost_weight = near_goal ? 0.0f : 1.0f;

  float traj_costs[kTrajectoryBlockSize] = {};
  uint8_t collides[kTrajectoryBlockSize] = {};
  uint8_t check_pending[kTrajectoryBlockSize] = {};
  int index[kTrajectoryBlockSize];
  uint8_t in_map[kTrajectoryBlockSize];
  uint8_t cell_cost[kTrajectoryBlockSize];

  // iterate over the trajectories backwards, as collisions are more likely towards the end
  // of the trajectory. Each sum is accumulated in the same point order as when scoring the
  // trajectories one by one; points after a collision are irrelevant. The loops are kept free
  // of branches for vectorization.
  for (int j = strided_traj_cols - 1; j >= 0; j--) {
    const float * traj_x = trajectories.x.data() + j * outer_stride + first;
    const float * traj_y = trajectories.y.data() + j * outer_stride + first;
    uint8_t * check_needed = footprintCheckNeeded(block, strided_traj_cols) +
      j * kTrajectoryBlockSize;

    // Same cell as worldToMapFloat()
    for (int k = 0; k < count; ++k) {
      const float mx = (traj_x[k] - origin_x) / resolution;
      const float my = (traj_y[k] - origin_y) / resolution;
      const bool inside = (traj_x[k] >= origin_x) & (traj_y[k] >= origin_y) & (mx < size_x) &
        (my < size_y);
      // clamped to keep the conversions defined
      const int cell_x = static_cast<int>(std::max(0.0f, std::min(mx, size_x)));
      const int cell_y = static_cast<int>(std::max(0.0f, std::min(my, size_y)));
      in_map[k] = static_cast<uint8_t>(inside);
      index[k] = inside ? cell_y * row_size + cell_x : 0;
    }
    for (int k = 0; k < count; ++k) {
      // NO_INFORMATION outside of the map
      const uint8_t cost = char_map[index[k]];
      cell_cost[k] = in_map[k] ? cost : nav2_costmap_2d::NO_INFORMATION;
    }

    // Decisions of inCollision(), with its footprint check deferred to checkFootprints().
    // The getCost doesn't use orientation
    // The footprintCostAtPose will always return "INSCRIBED" if footprint is over it
    // So the center point has more information than the footprint
    uint8_t any_check_needed = 0;
    for (int k = 0; k < count; ++k) {
      // Integer comparisons keep the loop free of branches
      const int cost = cell_cost[k];
      const bool not_free = cost >= 1;
      const bool lethal = (cost == nav2_costmap_2d::LETHAL_OBSTACLE) |
        (cost == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
      const bool check = not_free & !lethal & check_footprint &
        ((cost >= possible_collision_cost) | always_check_footprint);
      const bool unknown = not_free & !check & (cost == nav2_costmap_2d::NO_INFORMATION) &
        unknown_collides;
      collides[k] |= static_cast<uint8_t>(lethal | unknown);
      check_needed[k] = static_cast<uint8_t>(check);
      any_check_needed |= static_cast<uint8_t>(check);

      // Let near-collision trajectory points be punished severely
      // Note that we collision check based on the footprint actual,
      // but score based on the center-point cost regardless
      // Generally prefer trajectories further from obstacles, except near the goal
      const float far_cost = static_cast<float>(cost) * far_cost_weight;
      const float point_cost = selectFloat(cost >= near_collision_cost, critical_cost, far_cost);
      traj_costs[k] += selectFloat(not_free, point_cost, 0.0f);
    }

    if (any_check_needed) {
      for (int k = 0; k < count; ++k) {
        if (check_needed[k] && !collides[k]) {
          check_pending[k] = 1;
          bounds.min_x = std::min(bounds.min_x, traj_x[k]);
          bounds.max_x = std::max(bounds.max_x, traj_x[k]);
          bounds.min_y = std::min(bounds.min_y, traj_y[k]);
          bounds.max_y = std::max(bounds.max_y, traj_y[k]);
        }
      }
    }
  }

  bool footprint_check_needed = false;
  for (int k = 0; k < count; ++k) {
    repulsive_cost(first + k) = collides[k] ? collision_cost_ : traj_costs[k];
    if (collides[k]) {
      traj_state_[first + k] = kTrajectoryCollides;
    } else if (check_pending[k]) {
      traj_state_[first + k] = kTrajectoryCheckFootprint;
      footprint_check_needed = true;
    } else {
      traj_state_[first + k] = kTrajectoryFree;
    }
  }
  return footprint_check_needed;
}

bool CostCritic::checkFootprints(
  const models::Trajectories & trajectories, int block, int count,
  int strided_traj_cols, const nav2_costmap_2d::Footprint & footprint,
  Eigen::ArrayXf & repulsive_cost)
{
  const Eigen::Index outer_stride = trajectories.x.rows() * trajectory_point_step_;
  const int first = block * kTrajectoryBlockSize;
  const uint8_t * check_needed = footprintCheckNeeded(block, strided_traj_cols);
  bool all_collide = true;
  // Prefilter bookkeeping of this block: poses left in the current probe, the ones it cleared,
  // and poses left to check without it while it is paused
  int probe_left = kPrefilterProbe;
  int probe_cleared = 0;
  int pause_left = 0;
  for (int k = 0; k < count; ++k) {
    const int i = first + k;
    if (traj_state_[i] == kTrajectoryCheckFootprint) {
      for (int j = strided_traj_cols - 1; j >= 0; j--) {
        if (!check_needed[j * kTrajectoryBlockSize + k]) {
          continue;
        }
        const Eigen::Index idx = j * outer_stride + i;
        const float x = trajectories.x.data()[idx];
        const float y = trajectories.y.data()[idx];
        const float yaw = trajectories.yaws.data()[idx];
        if (prefilter_active_) {
          if (pause_left > 0) {
            --pause_left;
          } else {
            const bool cleared = footprintCollisionFree(x, y, yaw);
            probe_cleared += cleared ? 1 : 0;
            if (--probe_left == 0) {
              if (3 * probe_cleared < kPrefilterProbe) {
                pause_left = kPrefilterPause;
              }
              probe_left = kPrefilterProbe;
              probe_cleared = 0;
            }
            if (cleared) {
              continue;
            }
          }
        }
        if (footprintInCollision(x, y, yaw, footprint)) {
          traj_state_[i] = kTrajectoryCollides;
          repulsive_cost(i) = collision_cost_;
          break;
        }
      }
    }
    all_collide = all_collide && traj_state_[i] == kTrajectoryCollides;
  }
  return all_collide;
}

void CostCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  // Setup cost information for various parts of the critic
  updateCostmapInfo();

  nav2_costmap_2d::Footprint footprint;
  if (consider_footprint_) {
    // footprint may have changed since initialization if user has dynamic footprints
    possible_collision_cost_ = findCircumscribedCost(costmap_ros_);
    footprint = costmap_ros_->getRobotFootprint();
  }
  updateFootprintPrefilter(footprint);

  // If near the goal, don't apply the preferential term since the goal is near obstacles
  bool near_goal = false;
  if (data.state.local_path_length < near_goal_distance_) {
    near_goal = true;
  }

  Eigen::ArrayXf repulsive_cost(data.costs.rows());
  repulsive_cost.setZero();
  bool all_trajectories_collide = true;
  bool footprint_check_needed = false;
  CandidateBounds bounds;

  const int strided_traj_cols = floor((data.trajectories.x.cols() - 1) / trajectory_point_step_) +
    1;
  const int strided_traj_rows = data.trajectories.x.rows();
  const int num_blocks = (strided_traj_rows + kTrajectoryBlockSize - 1) / kTrajectoryBlockSize;
  const models::Trajectories & trajectories = data.trajectories;
  traj_state_.resize(static_cast<size_t>(strided_traj_rows));
  footprint_check_needed_.resize(
    static_cast<size_t>(num_blocks) * kTrajectoryBlockSize *
    static_cast<size_t>(strided_traj_cols));

  // A trajectory costs collision_cost_ if any of its points is in collision, otherwise the sum
  // of its point costs. The center costs are scored first, the footprint checks are only done
  // for the trajectories not already in collision by their center costs.
#ifdef _OPENMP
  #pragma omp parallel \
  default(none) \
  shared(repulsive_cost, near_goal, footprint, strided_traj_rows, strided_traj_cols, \
  num_blocks, trajectories, all_trajectories_collide, footprint_check_needed, bounds) \
  num_threads(getOptimalThreadCount())
#endif
  {
    CandidateBounds thread_bounds;
    bool thread_check_needed = false;
#ifdef _OPENMP
    #pragma omp for schedule(dynamic) nowait
#endif
    for (int block = 0; block < num_blocks; ++block) {
      const int count = std::min(
        strided_traj_rows - block * kTrajectoryBlockSize, +kTrajectoryBlockSize);
      thread_check_needed = scoreCenterCosts(
        trajectories, block, count, strided_traj_cols, near_goal, repulsive_cost,
        thread_bounds) || thread_check_needed;
    }
#ifdef _OPENMP
    #pragma omp critical
#endif
    {
      bounds.merge(thread_bounds);
      footprint_check_needed = footprint_check_needed || thread_check_needed;
    }
#ifdef _OPENMP
    #pragma omp barrier
    #pragma omp single
#endif
    if (footprint_check_needed) {
      updateBlockedCellDistance(bounds, footprint);
    }

#ifdef _OPENMP
    #pragma omp for schedule(dynamic) reduction(&&:all_trajectories_collide)
#endif
    for (int block = 0; block < num_blocks; ++block) {
      const int count = std::min(
        strided_traj_rows - block * kTrajectoryBlockSize, +kTrajectoryBlockSize);
      all_trajectories_collide = checkFootprints(
        trajectories, block, count, strided_traj_cols, footprint,
        repulsive_cost) && all_trajectories_collide;
    }
  }

  if (power_ > 1u) {
    data.costs += (repulsive_cost *
      (weight_ / static_cast<float>(strided_traj_cols))).pow(power_);
  } else {
    data.costs += repulsive_cost * (weight_ / static_cast<float>(strided_traj_cols));
  }

  data.fail_flag = all_trajectories_collide;
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::CostCritic,
  mppi::critics::CriticFunction)
