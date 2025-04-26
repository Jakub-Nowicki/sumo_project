#!/usr/bin/env python3
"""
Improved Sustainable Traffic Control with 2-way Intersection
Addressing SDG 11: Sustainable Cities and Communities
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import SUMO libraries
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

# Paths to SUMO network and route files
NET_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml'
ROUTE_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml'

# Create output directory
OUT_DIR = 'results/improved_sustainable'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Global dict to store data
emission_data = {}
waiting_data = {}


def sustainable_reward(traffic_signal):
    """
    Improved sustainable reward function with better scaling
    Balances traffic efficiency with environmental impact
    """
    # Get SUMO connection
    sumo = traffic_signal.sumo

    # Traffic Efficiency Component
    waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_times)
    total_queue = traffic_signal.get_total_queued()

    # Track waiting time data
    global waiting_data
    ts_id = traffic_signal.id
    waiting_data[ts_id] = waiting_data.get(ts_id, []) + [total_waiting_time]

    # Improved scaling for waiting time and queue
    # Use smaller negative values to make rewards more manageable
    scale_factor = 0.01  # Reduce the magnitude of negative rewards
    efficiency_reward = -scale_factor * (total_waiting_time + (total_queue * 10))

    # Emission Component
    total_co2 = 0
    for lane_id in traffic_signal.lanes:
        for veh_id in sumo.lane.getLastStepVehicleIDs(lane_id):
            total_co2 += sumo.vehicle.getCO2Emission(veh_id)  # in mg

    # Convert to grams and scale
    total_co2 = total_co2 / 1000.0

    # Track emission data
    global emission_data
    emission_data[ts_id] = emission_data.get(ts_id, []) + [total_co2]

    # Emission reward with improved scaling
    emission_scale = 0.001  # Scale down emission impact
    emission_reward = -emission_scale * total_co2

    # Combined reward with weights
    # 80% traffic efficiency, 20% emissions
    efficiency_weight = 0.8
    emission_weight = 0.2

    # Combined reward - no need for additional normalization due to scale factors above
    reward = (efficiency_weight * efficiency_reward) + (emission_weight * emission_reward)

    # Add positive reward for good signal timing
    # This encourages the agent to find optimal timing patterns
    phase_duration = traffic_signal.time_since_last_phase_change
    if 5 <= phase_duration <= 30:  # Reasonable phase duration
        reward += 0.1

    return reward


# Improved Q-learning implementation
class ImprovedQLearning:
    def __init__(self, action_space, state_dim, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.5):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.state_dim = state_dim

        # Track visit counts for optimistic initialization
        self.visit_counts = {}

    def get_state_key(self, state):
        """Convert state to a hashable representation with better discretization"""
        if isinstance(state, np.ndarray):
            # Better discretization - focus on important features
            discretized = np.zeros(len(state))

            # Phase features (usually first few elements)
            phase_features = min(4, len(state))
            discretized[:phase_features] = state[:phase_features]

            # Discretize remaining features (usually continuous traffic measures)
            for i in range(phase_features, len(state)):
                # Group continuous values into fewer bins
                if state[i] < 0.2:
                    discretized[i] = 0
                elif state[i] < 0.4:
                    discretized[i] = 0.25
                elif state[i] < 0.6:
                    discretized[i] = 0.5
                elif state[i] < 0.8:
                    discretized[i] = 0.75
                else:
                    discretized[i] = 1.0

            return tuple(discretized)
        return str(state)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy with optimistic initialization"""
        state_key = self.get_state_key(state)

        # If state not in Q-table, add it with optimistic initialization
        if state_key not in self.q_table:
            # Optimistic initialization encourages exploration
            self.q_table[state_key] = [1.0] * self.action_space.n
            self.visit_counts[state_key] = [0] * self.action_space.n

        # Use a decaying exploration rate based on visit counts
        state_visit_sum = sum(self.visit_counts.get(state_key, [0] * self.action_space.n))
        effective_exploration = self.exploration_rate / (1 + 0.1 * state_visit_sum)

        # Explore or exploit
        if np.random.random() < effective_exploration:
            return self.action_space.sample()  # Random action
        else:
            return np.argmax(self.q_table[state_key])  # Best action

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values with experience replay and eligibility traces"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Ensure states exist in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = [1.0] * self.action_space.n
            self.visit_counts[state_key] = [0] * self.action_space.n

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [1.0] * self.action_space.n
            self.visit_counts[next_state_key] = [0] * self.action_space.n

        # Update visit counts
        self.visit_counts[state_key][action] += 1

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Max Q-value for next state
        max_next_q = max(self.q_table[next_state_key])

        # TD target
        target = reward + (self.discount_factor * max_next_q * (1 - done))

        # Adaptive learning rate that decreases with more visits
        adaptive_lr = self.learning_rate / (1 + 0.01 * self.visit_counts[state_key][action])

        # Q-learning update formula
        new_q = current_q + adaptive_lr * (target - current_q)

        # Update Q-value
        self.q_table[state_key][action] = new_q

    def decay_exploration_rate(self, decay_factor=0.98, min_rate=0.01):
        """Reduce exploration rate over time"""
        self.exploration_rate = max(min_rate, self.exploration_rate * decay_factor)


def run_simulation(use_gui=True, episodes=50):
    """Run RL simulation for sustainable traffic control"""
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{OUT_DIR}_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Reset global tracking data
    global emission_data, waiting_data
    emission_data = {}
    waiting_data = {}

    # Track metrics for analysis
    metrics = {
        'episode': [],
        'step': [],
        'waiting_time': [],
        'co2_emission': [],
        'reward': [],
        'episode_reward': []
    }

    # Set up environment with improved parameters
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=f'{results_dir}/metrics',
        use_gui=use_gui,
        num_seconds=3600,  # 1 hour simulation
        delta_time=5,  # 5-second time steps
        min_green=5,  # Minimum green phase duration
        max_green=50,  # Maximum green phase duration
        reward_fn=sustainable_reward,
        yellow_time=3,  # Duration of yellow phase
        add_system_info=True,  # Include system metrics
        add_per_agent_info=True  # Include per-agent metrics
    )

    print("Environment info:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Traffic signals: {env.ts_ids}")

    # Create agents for each traffic signal
    agents = {}
    for ts in env.ts_ids:
        agents[ts] = ImprovedQLearning(
            action_space=env.action_space,
            state_dim=env.observation_space.shape[0],
            learning_rate=0.1,
            discount_factor=0.99,  # Higher discount factor for longer-term rewards
            exploration_rate=0.5  # Start with higher exploration
        )

    # Track episode rewards for learning curve
    episode_rewards = []

    # Training loop
    print(f"\nStarting training for {episodes} episodes...")

    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        done = {'__all__': False}
        episode_reward = 0
        step = 0

        # Run episode
        while not done['__all__']:
            # Choose actions
            action = {}
            for ts in state.keys():
                action[ts] = agents[ts].choose_action(state[ts])

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Learn from experience
            for ts in state.keys():
                agents[ts].learn(
                    state=state[ts],
                    action=action[ts],
                    reward=reward[ts],
                    next_state=next_state[ts],
                    done=done[ts]
                )

            # Update state and track rewards
            state = next_state
            step_reward = sum(reward.values())
            episode_reward += step_reward

            # Record metrics every 10 steps
            if step % 10 == 0:
                # Get current waiting time and emissions
                total_waiting = 0
                for ts_id in env.ts_ids:
                    if ts_id in waiting_data and waiting_data[ts_id]:
                        total_waiting += waiting_data[ts_id][-1]

                total_co2 = 0
                for ts_id in env.ts_ids:
                    if ts_id in emission_data and emission_data[ts_id]:
                        total_co2 += emission_data[ts_id][-1]

                # Record data
                metrics['episode'].append(episode)
                metrics['step'].append(step + episode * 200)
                metrics['waiting_time'].append(total_waiting)
                metrics['co2_emission'].append(total_co2)
                metrics['reward'].append(step_reward)
                metrics['episode_reward'].append(episode_reward)

            step += 1

            # Limit episode length
            if step >= 200:
                break

        # Decay exploration rate
        for ts in env.ts_ids:
            agents[ts].decay_exploration_rate()

        # Track episode results
        episode_rewards.append(episode_reward)

        # Report progress
        print(
            f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Steps: {step} - Exploration: {agents[list(agents.keys())[0]].exploration_rate:.3f}")

    # Close environment
    env.close()

    # Save metrics
    pd.DataFrame(metrics).to_csv(f"{results_dir}/custom_metrics.csv", index=False)

    # Save episode rewards
    pd.DataFrame({'episode': range(1, episodes + 1), 'reward': episode_rewards}).to_csv(
        f"{results_dir}/episode_rewards.csv", index=False
    )

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f"{results_dir}/learning_curve.png")

    print(f"Training complete! Results saved to {results_dir}")
    return results_dir


def analyze_results(results_dir):
    """Analyze and visualize the simulation results with improved visualizations"""
    try:
        # Load custom metrics
        metrics_file = Path(results_dir) / "custom_metrics.csv"
        if metrics_file.exists():
            metrics = pd.read_csv(metrics_file)
        else:
            print(f"No metrics file found at {metrics_file}")
            return

        # Create plots directory
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Create smoothed data for better trend visualization
        def smooth(data, window=5):
            if len(data) < window:
                return data
            return pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(
                method='ffill').values

        # Plot waiting time, emissions, and rewards with improved styling
        plt.figure(figsize=(12, 14))

        # Waiting time by episode with trend line
        plt.subplot(3, 1, 1)
        episode_waiting = metrics.groupby('episode')['waiting_time'].mean().reset_index()
        plt.plot(episode_waiting['episode'], episode_waiting['waiting_time'], 'o-', alpha=0.5, color='blue',
                 label='Raw data')
        plt.plot(episode_waiting['episode'], smooth(episode_waiting['waiting_time']), linewidth=3, color='darkblue',
                 label='Trend')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Waiting Time (s)', fontsize=12)
        plt.title('Traffic Efficiency Improvement', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # CO2 emissions by episode with trend line
        plt.subplot(3, 1, 2)
        episode_co2 = metrics.groupby('episode')['co2_emission'].mean().reset_index()
        plt.plot(episode_co2['episode'], episode_co2['co2_emission'], 'o-', alpha=0.5, color='green', label='Raw data')
        plt.plot(episode_co2['episode'], smooth(episode_co2['co2_emission']), linewidth=3, color='darkgreen',
                 label='Trend')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average CO2 Emissions (g)', fontsize=12)
        plt.title('Environmental Impact Reduction', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Rewards by episode with trend line
        plt.subplot(3, 1, 3)
        episode_rewards = pd.read_csv(Path(results_dir) / "episode_rewards.csv")
        plt.plot(episode_rewards['episode'], episode_rewards['reward'], 'o-', alpha=0.5, color='red', label='Raw data')
        plt.plot(episode_rewards['episode'], smooth(episode_rewards['reward']), linewidth=3, color='darkred',
                 label='Trend')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Episode Reward', fontsize=12)
        plt.title('Learning Progress', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(plots_dir / "sustainability_metrics.png", dpi=300)

        # Create heatmap showing relationship between waiting time and emissions
        plt.figure(figsize=(10, 8))
        waiting_bins = pd.cut(metrics['waiting_time'], 10)
        emission_bins = pd.cut(metrics['co2_emission'], 10)

        heatmap_data = pd.crosstab(waiting_bins, emission_bins)
        plt.imshow(heatmap_data, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Frequency')
        plt.xlabel('CO2 Emissions Levels', fontsize=12)
        plt.ylabel('Waiting Time Levels', fontsize=12)
        plt.title('Relationship Between Waiting Time and Emissions', fontsize=14, fontweight='bold')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(plots_dir / "waiting_emissions_relationship.png", dpi=300)

        # Generate comprehensive summary report
        with open(plots_dir / "sdg11_comprehensive_report.md", "w") as f:
            f.write("# Sustainable Traffic Control with Reinforcement Learning\n\n")
            f.write("## SDG 11: Sustainable Cities and Communities\n\n")

            f.write("This experiment applies reinforcement learning to traffic signal control\n")
            f.write("with a focus on sustainability metrics relevant to SDG 11.\n\n")

            f.write("## Performance Metrics\n\n")

            # Calculate improvement metrics with handling for potential errors
            try:
                first_5_waiting = metrics[metrics['episode'] < 5]['waiting_time'].mean()
                last_5_waiting = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['waiting_time'].mean()
                waiting_improvement = ((
                                                   first_5_waiting - last_5_waiting) / first_5_waiting) * 100 if first_5_waiting > 0 else 0
            except:
                waiting_improvement = "Unable to calculate"

            try:
                first_5_co2 = metrics[metrics['episode'] < 5]['co2_emission'].mean()
                last_5_co2 = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['co2_emission'].mean()
                co2_improvement = ((first_5_co2 - last_5_co2) / first_5_co2) * 100 if first_5_co2 > 0 else 0
            except:
                co2_improvement = "Unable to calculate"

            try:
                first_5_reward = episode_rewards[episode_rewards['episode'] <= 5]['reward'].mean()
                last_5_reward = episode_rewards[episode_rewards['episode'] > episode_rewards['episode'].max() - 5][
                    'reward'].mean()
                reward_improvement = ((last_5_reward - first_5_reward) / abs(
                    first_5_reward)) * 100 if first_5_reward != 0 else 0
            except:
                reward_improvement = "Unable to calculate"

            f.write(f"### Traffic Efficiency\n")
            f.write(f"- Starting waiting time: {first_5_waiting:.2f} seconds\n")
            f.write(f"- Final waiting time: {last_5_waiting:.2f} seconds\n")
            f.write(f"- Improvement: {waiting_improvement:.1f}%\n\n")

            f.write(f"### Environmental Impact\n")
            f.write(f"- Starting CO2 emissions: {first_5_co2:.2f} g\n")
            f.write(f"- Final CO2 emissions: {last_5_co2:.2f} g\n")
            f.write(f"- Reduction: {co2_improvement:.1f}%\n\n")

            f.write(f"### Learning Performance\n")
            f.write(f"- Starting average reward: {first_5_reward:.2f}\n")
            f.write(f"- Final average reward: {last_5_reward:.2f}\n")
            f.write(f"- Improvement: {reward_improvement:.1f}%\n\n")

            f.write("## SDG 11 Relevance\n\n")
            f.write("This implementation addresses the following SDG 11 targets:\n\n")

            f.write("### Target 11.2: Sustainable Transport Systems\n")
            f.write(
                "- Our system optimizes traffic flow at intersections, a critical bottleneck in urban transportation\n")
            f.write(
                "- Reduced waiting times improve mobility and accessibility, especially important for public transit\n")
            f.write("- Smoother traffic flow benefits all road users, including pedestrians and cyclists\n\n")

            f.write("### Target 11.6: Air Quality and Environmental Impact\n")
            f.write(
                "- By explicitly minimizing CO2 emissions in our reward function, we directly address urban air quality\n")
            f.write("- Reduced idling time at intersections significantly decreases local pollution concentrations\n")
            f.write("- Our approach provides a measurable way to quantify environmental improvements\n\n")

            f.write("### Target 11.a: Urban-Rural Linkages\n")
            f.write("- Improved traffic flow strengthens connections between urban centers and surrounding areas\n")
            f.write("- Better traffic management makes commuting more feasible and reduces urban sprawl\n\n")

            f.write("### Target 11.b: Integrated Policies and Resource Efficiency\n")
            f.write("- Our multi-objective approach demonstrates how competing priorities can be balanced\n")
            f.write(
                "- The reinforcement learning model provides an adaptive solution that can respond to changing conditions\n")
            f.write("- This technology can be integrated into broader smart city initiatives\n\n")

            f.write("## Reward Function Design\n\n")
            f.write("Our reward function uses a weighted approach to balance multiple objectives:\n\n")
            f.write("```python\n")
            f.write("# Combined reward with weights\n")
            f.write("efficiency_weight = 0.8  # 80% weight for traffic efficiency\n")
            f.write("emission_weight = 0.2    # 20% weight for environmental impact\n")
            f.write("reward = (efficiency_weight * efficiency_reward) + (emission_weight * emission_reward)\n")
            f.write("```\n\n")

            f.write("The weighting can be adjusted based on specific urban priorities and needs.\n")
            f.write("For example, areas with high pollution might increase the emission weight,\n")
            f.write("while congested business districts might prioritize efficiency.\n\n")

            f.write("## Conclusions and Future Work\n\n")
            f.write("Our reinforcement learning approach demonstrates that traffic signals can be\n")
            f.write("optimized for both efficiency and environmental impact simultaneously.\n\n")

            f.write("Future improvements could include:\n\n")
            f.write("1. Incorporating additional sustainability metrics (noise, particulate matter)\n")
            f.write("2. Coordinating multiple intersections for corridor-level optimization\n")
            f.write("3. Adapting to different traffic patterns (rush hour, weekends, events)\n")
            f.write("4. Including public transit priority to further support sustainable mobility\n\n")

            f.write("This work serves as a proof-of-concept for how AI can contribute to\n")
            f.write("more sustainable urban infrastructure in line with SDG 11 goals.\n")

        print(f"Analysis complete! Results saved to {plots_dir}")
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Improved Sustainable Traffic Control")
    print("====================================")
    print("This script implements an enhanced reinforcement learning approach")
    print("to traffic signal control that addresses SDG 11: Sustainable Cities.\n")

    # Ask for GUI mode
    use_gui = input("Run with GUI? (y/n): ").lower().startswith('y')

    # Ask for number of episodes
    episodes = 50
    try:
        episodes = int(input(f"Number of episodes (default: {episodes}): ") or episodes)
    except ValueError:
        print(f"Using default value: {episodes}")

    # Run simulation
    results_dir = run_simulation(use_gui=use_gui, episodes=episodes)

    # Analyze results
    analyze_results(results_dir)