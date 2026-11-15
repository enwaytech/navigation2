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
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "geometry_msgs/msg/point.hpp"
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
    float score_cost = cost;
    if (consider_footprint_ &&
      (cost >= possible_collision_cost_ || possible_collision_cost_ < 1.0f))
    {
      score_cost = static_cast<float>(collision_checker_.footprintCostAtPose(
          static_cast<double>(x), static_cast<double>(y), static_cast<double>(theta),
          footprint));
    }

    switch (static_cast<unsigned char>(score_cost)) {
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

  /**
    * @brief Caches the costmap geometry and data pointer used by the collision checks
    */
  void updateCostmapInfo();

  // Number of trajectories processed together, point by point, by one OpenMP task
  static constexpr int kTrajectoryBlockSize = 32;

  /**
    * @brief World bounding box of the trajectory points that need a footprint check
    */
  struct CandidateBounds
  {
    float min_x{std::numeric_limits<float>::max()};
    float min_y{std::numeric_limits<float>::max()};
    float max_x{std::numeric_limits<float>::lowest()};
    float max_y{std::numeric_limits<float>::lowest()};

    void merge(const CandidateBounds & other)
    {
      min_x = std::min(min_x, other.min_x);
      min_y = std::min(min_y, other.min_y);
      max_x = std::max(max_x, other.max_x);
      max_y = std::max(max_y, other.max_y);
    }
  };

  /**
    * @brief Footprint check flags of a block of trajectories, point j of trajectory k of the
    * block at j * kTrajectoryBlockSize + k
    */
  inline uint8_t * footprintCheckNeeded(int block, int strided_traj_cols)
  {
    return footprint_check_needed_.data() +
           static_cast<size_t>(block) * kTrajectoryBlockSize *
           static_cast<size_t>(strided_traj_cols);
  }

  /**
    * @brief Center cost pass over a block of trajectories: accumulates the point costs into
    * repulsive_cost, flags collisions by center cost and the points that need a footprint check
    * @param bounds [in, out] extended by the points that need a footprint check
    * @return if any non-colliding trajectory of the block needs a footprint check
    */
  bool scoreCenterCosts(
    const models::Trajectories & trajectories, int block, int count, int strided_traj_cols,
    bool near_goal, Eigen::ArrayXf & repulsive_cost, CandidateBounds & bounds);

  /**
    * @brief Footprint check pass over a block of trajectories, run after scoreCenterCosts()
    * @return if all trajectories of the block are in collision
    */
  bool checkFootprints(
    const models::Trajectories & trajectories, int block, int count,
    int strided_traj_cols, const nav2_costmap_2d::Footprint & footprint,
    Eigen::ArrayXf & repulsive_cost);

  /**
    * @brief Footprint collision check of a pose, same result as inCollision() taking the
    * footprint branch (consider_footprint_ set) with
    * FootprintCollisionChecker::footprintCostAtPose(), but working directly on the costmap array.
    * Requires updateBlockedCellDistance() over a region holding the footprint vertices.
    * @param x X of pose
    * @param y Y of pose
    * @param theta theta of pose
    * @param footprint Robot footprint
    * @return bool if in collision
    */
  bool footprintInCollision(
    float x, float y, float theta,
    const nav2_costmap_2d::Footprint & footprint);

  /**
    * @brief Which of LETHAL_OBSTACLE and NO_INFORMATION occur on a rasterized footprint line
    */
  enum class LineCost : uint8_t
  {
    FREE,      // neither
    UNKNOWN,   // NO_INFORMATION but no LETHAL_OBSTACLE
    LETHAL     // LETHAL_OBSTACLE
  };

  /**
    * @brief Classifies the cells of the Bresenham line of nav2_util::LineIterator from
    * (x0, y0) to (x1, y1). Cells closer to the current one than its distance to the nearest
    * blocked cell are skipped.
    * @return Line cost class
    */
  LineCost lineCost(int x0, int y0, int x1, int y1) const;

  /**
    * @brief Updates the blocked cell distances over the region of the footprints centered in
    * bounds
    */
  void updateBlockedCellDistance(
    const CandidateBounds & bounds, const nav2_costmap_2d::Footprint & footprint);

  /**
    * @brief Computes for the cells of an inclusive region a lower bound of the chessboard
    * distance in cells to the nearest blocked cell (cost >= LETHAL_OBSTACLE), saturated at 255.
    * It is 0 exactly for blocked cells, and the exact distance if the region is the whole map.
    */
  void updateBlockedCellDistance(int x_min, int y_min, int x_max, int y_max);

  /**
    * @brief Footprint vertex transform of
    * nav2_costmap_2d::FootprintCollisionChecker::footprintCostAtPose()
    * @param x X of pose
    * @param y Y of pose
    * @param cos_th cosine of pose theta
    * @param sin_th sine of pose theta
    * @param vertex Footprint vertex
    * @param wx [out] X of the transformed vertex
    * @param wy [out] Y of the transformed vertex
    */
  static inline void transformFootprintVertex(
    double x, double y, double cos_th, double sin_th,
    const geometry_msgs::msg::Point & vertex, double & wx, double & wy)
  {
    wx = x + (vertex.x * cos_th - vertex.y * sin_th);
    wy = y + (vertex.x * sin_th + vertex.y * cos_th);
  }

  /**
    * @brief Prepares the footprint prefilter of the current score() call, see
    * footprintCollisionFree()
    * @param footprint Robot footprint
    */
  void updateFootprintPrefilter(const nav2_costmap_2d::Footprint & footprint);

  /**
    * @brief Circles along the long axis of the footprint bounding box that cover the footprint
    * outline: every point of the outline is within the radius of one of the circles
    * @param footprint Robot footprint
    * @param x [out] X of the circle centers in the robot frame
    * @param y [out] Y of the circle centers in the robot frame
    * @param radius [out] Circle radii
    */
  static void computeCoveringCircles(
    const nav2_costmap_2d::Footprint & footprint, std::vector<double> & x,
    std::vector<double> & y, std::vector<double> & radius);

  /**
    * @brief Conservative prefilter of the footprint check, run when prefilter_active_ is set.
    * It proves that no rasterized footprint line holds a blocked cell and that all footprint
    * vertices are in the map, so that footprintInCollision() is false, from the distance to
    * the nearest blocked cell at the centers of the covering circles. It needs
    * blocked_cell_distance_ of the current costmap around the pose.
    * @param x X of pose
    * @param y Y of pose
    * @param theta theta of pose
    * @return true if the pose is proven to be free of collision, false if unknown
    */
  bool footprintCollisionFree(float x, float y, float theta) const;

  /**
    * @brief Same result as Costmap2D::worldToMap() for finite coordinates, false otherwise
    * @return if successful
    */
  inline bool worldToMapDouble(
    double wx, double wy, unsigned int & mx, unsigned int & my) const
  {
    // The comparisons are false for NaN, and reject coordinates beyond the map before the
    // conversion to a cell index
    const double cell_x = (wx - map_origin_x_) / map_resolution_;
    const double cell_y = (wy - map_origin_y_) / map_resolution_;
    if (!(wx >= map_origin_x_ && wy >= map_origin_y_ &&
      cell_x < static_cast<double>(size_x_) && cell_y < static_cast<double>(size_y_)))
    {
      return false;
    }

    mx = static_cast<unsigned int>(cell_x);
    my = static_cast<unsigned int>(cell_y);
    return true;
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
  double map_origin_x_{0.0}, map_origin_y_{0.0}, map_resolution_{1.0};
  const unsigned char * char_map_{nullptr};

  // Scratch data of score(): distance to the nearest blocked cell, per trajectory state and per
  // trajectory point footprint check flag, see footprintCheckNeeded()
  std::vector<uint8_t> blocked_cell_distance_;
  std::vector<uint8_t> traj_state_;
  std::vector<uint8_t> footprint_check_needed_;

  // Footprint prefilter: covering circles of prefilter_footprint_ in the robot frame, and for
  // the current score() call the circle centers in cells, the distance in cells to the nearest
  // blocked cell their center cells need, the minimum distance in cells of the pose to the map
  // border, and how far from the pose it reads blocked_cell_distance_
  bool use_footprint_prefilter_{true};
  bool prefilter_active_{false};
  nav2_costmap_2d::Footprint prefilter_footprint_;
  std::vector<double> prefilter_center_x_;
  std::vector<double> prefilter_center_y_;
  std::vector<double> prefilter_radius_;
  std::vector<float> prefilter_offset_x_;
  std::vector<float> prefilter_offset_y_;
  std::vector<uint8_t> prefilter_min_distance_;
  float prefilter_pose_border_{0.0f};
  double prefilter_reach_{0.0};

  float near_goal_distance_;
  std::string inflation_layer_name_;

  int num_threads_{-1};
  unsigned int power_{0};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__COST_CRITIC_HPP_
