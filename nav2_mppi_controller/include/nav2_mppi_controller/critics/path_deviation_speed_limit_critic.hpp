// Copyright (c) 2026 Enway GmbH, Adi Vardi
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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__PATH_DEVIATION_SPEED_LIMIT_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__PATH_DEVIATION_SPEED_LIMIT_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi::critics
{

/**
 * @class mppi::critics::PathDeviationSpeedLimitCritic
 * @brief Critic objective function for limiting linear speed based on path deviation
 */
class PathDeviationSpeedLimitCritic : public CriticFunction
{
public:
  /**
    * @brief Initialize critic
    */
  void initialize() override;

  /**
   * @brief Penalize fast driving when away from path
   *
   * @param data CriticData with the sampled trajectory states; per-trajectory
   *             cost is accumulated into data.costs.
   */
  void score(CriticData & data) override;

protected:
  float min_speed_{0};
  float min_deviation_{0};
  float weight_{0};
  unsigned int power_{0};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__PATH_DEVIATION_SPEED_LIMIT_CRITIC_HPP_
