// Copyright (c) 2023 Robocc Brice Renaudeau
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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__COST_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__COST_CRITIC_HPP_

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_costmap_2d/inflation_layer.hpp"

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi::critics
{

/**
 * @class mppi::critics::CostCritic
 * @brief Critic objective function for avoiding obstacles using costmap's inflated cost
 */
class CostCritic : public CriticFunction
{
public:
  /**
    * @brief Initialize critic
    */
  void initialize() override;

  /**
   * @brief Evaluate cost related to obstacle avoidance
   *
   * @param costs [out] add obstacle cost values to this tensor
   */
  void score(CriticData & data) override;

protected:
  /**
    * @brief Checks if cost represents a collision
    * @param cost Point cost at pose center
    * @param x X of pose
    * @param y Y of pose
    * @param theta theta of pose
    * @return bool if in collision
    */
  inline bool inCollision(
    float cost, float x, float y, float theta,
    const nav2_costmap_2d::Footprint & footprint)
  {
    // if the center cost guarantees a collision, return before doing an expensive footprint check
    if (cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
      cost == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
    {
      return true;
    }

    // If consider_footprint_ check footprint score for collision
    if (consider_footprint_ &&
      (cost >= possible_collision_cost_ || possible_collision_cost_ < 1.0f))
    {
      return footprintInCollision(
        static_cast<double>(x), static_cast<double>(y), static_cast<double>(theta), footprint);
    }

    switch (static_cast<unsigned char>(cost)) {
      case (nav2_costmap_2d::LETHAL_OBSTACLE):
        return true;
      case (nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE):
        return consider_footprint_ ? false : true;
      case (nav2_costmap_2d::NO_INFORMATION):
        return is_tracking_unknown_ ? false : true;
    }

    return false;
  }

  /**
    * @brief Whether the footprint at a pose collides, i.e. whether the footprint cost of
    * FootprintCollisionChecker::footprintCostAtPose() is LETHAL_OBSTACLE, or NO_INFORMATION while
    * not tracking unknown space. That cost is LETHAL_OBSTACLE if a footprint vertex is off the map.
    * Otherwise the edges are rasterized in the order of the collision checker: the first edge with
    * a lethal or unknown cell decides, LETHAL_OBSTACLE if it has a lethal cell, NO_INFORMATION
    * otherwise. Edges whose bounding box holds no lethal or unknown cell (summed-area table) are
    * skipped.
    * @param x X of pose
    * @param y Y of pose
    * @param theta theta of pose
    * @param footprint Robot footprint
    * @return bool if in collision
    */
  inline bool footprintInCollision(
    double x, double y, double theta, const nav2_costmap_2d::Footprint & footprint)
  {
    constexpr size_t max_vertices = 16;
    const size_t n_vertices = footprint.size();
    if (n_vertices < 2 || n_vertices > max_vertices) {
      switch (static_cast<unsigned char>(static_cast<float>(
          collision_checker_.footprintCostAtPose(x, y, theta, footprint))))
      {
        case (nav2_costmap_2d::LETHAL_OBSTACLE):
          return true;
        case (nav2_costmap_2d::NO_INFORMATION):
          return !is_tracking_unknown_;
      }
      return false;
    }

    // Oriented footprint vertices in map cells, as in
    // FootprintCollisionChecker::footprintCostAtPose()
    const double cos_th = cos(theta);
    const double sin_th = sin(theta);
    int mx[max_vertices], my[max_vertices];
    int min_x = INT_MAX, min_y = INT_MAX, max_x = 0, max_y = 0;
    for (size_t i = 0; i < n_vertices; ++i) {
      const double wx = x + (footprint[i].x * cos_th - footprint[i].y * sin_th);
      const double wy = y + (footprint[i].x * sin_th + footprint[i].y * cos_th);
      if (wx < map_origin_x_ || wy < map_origin_y_) {
        return true;
      }
      const unsigned int cell_x = toCell(wx - map_origin_x_);
      const unsigned int cell_y = toCell(wy - map_origin_y_);
      if (cell_x >= size_x_ || cell_y >= size_y_) {
        return true;
      }
      mx[i] = static_cast<int>(cell_x);
      my[i] = static_cast<int>(cell_y);
      min_x = std::min(min_x, mx[i]);
      min_y = std::min(min_y, my[i]);
      max_x = std::max(max_x, mx[i]);
      max_y = std::max(max_y, my[i]);
    }

    // All edge cells lie within the bounding box of the vertex cells
    if (blockingCellCount(min_x, min_y, max_x, max_y) == 0) {
      return false;
    }

    // Edges i -> i + 1, closed by the edge from the first to the last vertex
    unsigned char edge_cost = nav2_costmap_2d::FREE_SPACE;
    for (size_t i = 0; i + 1 < n_vertices && edge_cost == nav2_costmap_2d::FREE_SPACE; ++i) {
      edge_cost = edgeBlockingCost(mx[i], my[i], mx[i + 1], my[i + 1]);
    }
    if (edge_cost == nav2_costmap_2d::FREE_SPACE) {
      edge_cost = edgeBlockingCost(mx[0], my[0], mx[n_vertices - 1], my[n_vertices - 1]);
    }
    switch (edge_cost) {
      case (nav2_costmap_2d::LETHAL_OBSTACLE):
        return true;
      case (nav2_costmap_2d::NO_INFORMATION):
        return !is_tracking_unknown_;
    }
    return false;
  }

  /**
    * @brief Number of lethal or unknown cells in the cell rectangle [x0, x1] x [y0, y1]
    */
  inline uint32_t blockingCellCount(int x0, int y0, int x1, int y1) const
  {
    const size_t stride = size_x_ + 1;
    const uint32_t * row0 = blocking_cells_sat_.data() + static_cast<size_t>(y0) * stride;
    const uint32_t * row1 = blocking_cells_sat_.data() + static_cast<size_t>(y1 + 1) * stride;
    return row1[x1 + 1] - row1[x0] - row0[x1 + 1] + row0[x0];
  }

  /**
    * @brief Walk the cells of the line from (x0, y0) to (x1, y1), rasterized like
    * FootprintCollisionChecker::lineCost()
    * @return LETHAL_OBSTACLE if a cell is lethal, else NO_INFORMATION if a cell is unknown,
    * else FREE_SPACE
    */
  inline unsigned char edgeBlockingCost(int x0, int y0, int x1, int y1) const
  {
    if (blockingCellCount(std::min(x0, x1), std::min(y0, y1), std::max(x0, x1),
        std::max(y0, y1)) == 0)
    {
      return nav2_costmap_2d::FREE_SPACE;
    }

    // nav2_util::LineIterator steps the major axis every cell and the minor axis whenever the
    // numerator (starting at den / 2, plus numadd per cell) reaches den: after k cells the minor
    // axis has stepped (den / 2 + k * numadd) / den times. The cells are walked in chunks, chunks
    // whose bounding box holds no lethal or unknown cell are skipped.
    const int delta_x = std::abs(x1 - x0);
    const int delta_y = std::abs(y1 - y0);
    const int x_inc = x1 >= x0 ? 1 : -1;
    const int y_inc = y1 >= y0 ? 1 : -1;
    const bool x_major = delta_x >= delta_y;
    const int den = x_major ? delta_x : delta_y;
    const int numadd = x_major ? delta_y : delta_x;
    const int row_inc = y_inc * static_cast<int>(size_x_);
    const int major_index_inc = x_major ? x_inc : row_inc;
    const int minor_index_inc = x_major ? row_inc : x_inc;
    auto minorSteps = [&](int k) {return den > 0 ? (den / 2 + k * numadd) / den : 0;};
    auto cellX = [&](int k, int minor_steps) {
        return x0 + x_inc * (x_major ? k : minor_steps);
      };
    auto cellY = [&](int k, int minor_steps) {
        return y0 + y_inc * (x_major ? minor_steps : k);
      };

    constexpr int chunk_cells = 8;
    unsigned char edge_cost = nav2_costmap_2d::FREE_SPACE;
    for (int k_begin = 0; k_begin <= den; k_begin += chunk_cells) {
      const int k_end = std::min(k_begin + chunk_cells - 1, den);
      const int minor_begin = minorSteps(k_begin);
      const int minor_end = minorSteps(k_end);
      const int cx0 = cellX(k_begin, minor_begin), cy0 = cellY(k_begin, minor_begin);
      const int cx1 = cellX(k_end, minor_end), cy1 = cellY(k_end, minor_end);
      if (blockingCellCount(std::min(cx0, cx1), std::min(cy0, cy1), std::max(cx0, cx1),
          std::max(cy0, cy1)) == 0)
      {
        continue;
      }
      int num = den > 0 ? (den / 2 + k_begin * numadd) % den : 0;
      int index = static_cast<int>(getIndex(cx0, cy0));
      for (int k = k_begin; ; ) {
        const unsigned char cost = costmap_char_map_[index];
        if (cost == nav2_costmap_2d::LETHAL_OBSTACLE) {
          return cost;
        }
        if (cost == nav2_costmap_2d::NO_INFORMATION) {
          edge_cost = cost;
        }
        if (++k > k_end) {
          break;
        }
        index += major_index_inc;
        num += numadd;
        if (num >= den) {
          num -= den;
          index += minor_index_inc;
        }
      }
    }
    return edge_cost;
  }

  /**
    * @brief static_cast<unsigned int>(offset / map_resolution_) for an offset >= 0. Multiplies with
    * the reciprocal and only divides if the quotient is within rounding error of an integer.
    */
  inline unsigned int toCell(double offset) const
  {
    const double quotient = offset * map_inv_resolution_;
    const double cell = std::floor(quotient);
    // The product deviates from the rounded division by less than 4e-16 * quotient
    if (quotient < 1e9 && quotient - cell > 1e-6 && cell + 1.0 - quotient > 1e-6) {
      return static_cast<unsigned int>(cell);
    }
    return static_cast<unsigned int>(offset / map_resolution_);
  }

  /**
    * @brief Build the summed-area table of the lethal and unknown cells of the costmap
    */
  void updateBlockingCellsTable();

  /**
    * @brief Find the min cost of the inflation decay function for which the robot MAY be
    * in collision in any orientation
    * @param costmap Costmap2DROS to get minimum inscribed cost (e.g. 128 in inflation layer documentation)
    * @return double circumscribed cost, any higher than this and need to do full footprint collision checking
    * since some element of the robot could be in collision
    */
  inline float findCircumscribedCost(std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap);

  /**
   * @brief Determine the number of threads to use for OpenMP parallelization.
   *        If num_threads_ > 0, use that value directly.
   *        If num_threads_ == -1 (auto), use half the available cores (memory-bound heuristic).
   *        If OpenMP is disabled at build time, always returns 1.
   * @return Number of threads to use
   */
  int getOptimalThreadCount();

  /**
    * @brief An implementation of worldToMap fully using floats
    * @param wx Float world X coord
    * @param wy Float world Y coord
    * @param mx unsigned int map X coord
    * @param my unsigned into map Y coord
    * @return if successful
    */
  inline bool worldToMapFloat(float wx, float wy, unsigned int & mx, unsigned int & my) const
  {
    if (wx < origin_x_ || wy < origin_y_) {
      return false;
    }

    mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
    my = static_cast<unsigned int>((wy - origin_y_) / resolution_);

    if (mx < size_x_ && my < size_y_) {
      return true;
    }
    return false;
  }

  /**
    * @brief A local implementation of getIndex
    * @param mx unsigned int map X coord
    * @param my unsigned into map Y coord
    * @return Index
    */
  inline unsigned int getIndex(unsigned int mx, unsigned int my) const
  {
    return my * size_x_ + mx;
  }

  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>
  collision_checker_{nullptr};
  float possible_collision_cost_;

  bool consider_footprint_{true};
  bool is_tracking_unknown_{true};
  float circumscribed_radius_{0.0f};
  float circumscribed_cost_{0.0f};
  float collision_cost_{0.0f};
  float critical_cost_{0.0f};
  unsigned int near_collision_cost_{253};
  float weight_{0};
  unsigned int trajectory_point_step_;

  float origin_x_, origin_y_, resolution_;
  unsigned int size_x_, size_y_;
  double map_origin_x_, map_origin_y_, map_resolution_, map_inv_resolution_;
  const unsigned char * costmap_char_map_{nullptr};
  // Summed-area table of the lethal and unknown cells, (size_x_ + 1) x (size_y_ + 1) with a zero
  // first row and column
  std::vector<uint32_t> blocking_cells_sat_;

  float near_goal_distance_;
  std::string inflation_layer_name_;

  int num_threads_{-1};
  unsigned int power_{0};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__COST_CRITIC_HPP_
