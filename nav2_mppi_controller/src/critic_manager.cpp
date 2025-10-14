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

#include "nav2_mppi_controller/critic_manager.hpp"

namespace mppi
{

void CriticManager::on_configure(
  nav2::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, ParametersHandler * param_handler)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  name_ = name;
  auto node = parent_.lock();
  logger_ = node->get_logger();
  parameters_handler_ = param_handler;

  getParams();
  loadCritics();
}

void CriticManager::getParams()
{
  auto node = parent_.lock();
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(critic_names_, "critics", std::vector<std::string>{}, ParameterType::Static);
}

void CriticManager::loadCritics()
{
  if (!loader_) {
    loader_ = std::make_unique<pluginlib::ClassLoader<critics::CriticFunction>>(
      "nav2_mppi_controller", "mppi::critics::CriticFunction");
  }

  critics_.clear();
  for (auto name : critic_names_) {
    std::string fullname = getFullName(name);
    auto instance = std::unique_ptr<critics::CriticFunction>(
      loader_->createUnmanagedInstance(fullname));
    critics_.push_back(std::move(instance));
    critics_.back()->on_configure(
      parent_, name_, name_ + "." + name, costmap_ros_,
      parameters_handler_);
    RCLCPP_INFO(logger_, "Critic loaded : %s", fullname.c_str());
  }
}

std::string CriticManager::getFullName(const std::string & name)
{
  return "mppi::critics::" + name;
}

void CriticManager::evalTrajectoriesScores(
  CriticData & data) const
{
  // std::cout << "evalTrajectoriesScores:" << std::endl;
  // std::cout << "data.state.vx: " << data.state.vx(Eigen::seq(0, 9), 0).transpose() << "\n";
  // std::cout << "data.state.cvx: " << data.state.cvx(Eigen::seq(0, 9), 0).transpose() << "\n";
  // std::cout << "data.trajectories.x: " << data.trajectories.x(Eigen::seq(0, 9), 0).transpose() << "\n";

  if (data.critic_costs.size() < critics_.size()) {
    data.critic_costs.resize(critics_.size());
  }

  unsigned int i = 0;
  for (const auto & critic : critics_) {

    if (data.fail_flag) {
      break;
    }
    const auto cost_before = data.costs;
    critic->score(data);

    data.critic_costs.at(i) = data.costs - cost_before;
    i++;
  }

  // Print individual critic costs for the best trajectory (lowest total cost)
  if (!data.critic_costs.empty() && data.costs.size() > 0) {
    Eigen::Index best_traj_idx;
    data.costs.minCoeff(&best_traj_idx);

    double total_cost = data.costs(best_traj_idx);
    RCLCPP_INFO(logger_, "Best trajectory (#%ld) individual critic costs:", best_traj_idx);

    for (size_t j = 0; j < data.critic_costs.size() && j < critics_.size(); ++j) {
      double critic_cost = data.critic_costs[j](best_traj_idx);
      double percentage = (total_cost > 0.0) ? (critic_cost / total_cost) * 100.0 : 0.0;
      RCLCPP_INFO(logger_, "  %s: %.4f (%.1f%%)",
                  critics_[j]->getName().c_str(), critic_cost, percentage);
    }
    RCLCPP_INFO(logger_, "  Total cost: %.4f", total_cost);
  }
}

}  // namespace mppi
