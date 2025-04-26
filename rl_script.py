#!/usr/bin/env python3
"""
Simple Sustainable Traffic Control with SUMO
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
NET_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\single-intersection\single-intersection.net.xml'
ROUTE_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\single-intersection\single-intersection.rou.xml'

# Create output directory
OUT_DIR = 'results/sustainable'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Global dict to store data
emission_data = {}
waiting_data = {}


def sustainable_reward(traffic_signal):
    """Custom reward function for sustainable traffic control"""
    # Get SUMO connection
    sumo = traffic_signal.sumo

    # Traffic Efficiency Component - Queue length and waiting time
    waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_times)
    total_queue = traffic_signal.get_total_queued()

    # Track waiting time data
    global waiting_data
    waiting_data[traffic_signal.id] = waiting_data.get(traffic_signal.id, []) + [total_waiting_time]

    # Efficiency reward (negative values to minimize)
    efficiency_reward = -(total_waiting_time + total_queue)

    # Emission Component
    total_co2 = 0
    for lane_id in traffic_signal.lanes:
        for veh_id in sumo.lane.getLastStepVehicleIDs(lane_id):
            total_co2 += sumo.vehicle.getCO2Emission(veh_id)  # in mg

    # Convert to grams for better scale
    total_co2 = total_co2 / 1000.0

    # Track emission data
    global emission_data
    emission_data[traffic_signal.id] = emission_data.get(traffic_signal.id, []) + [total_co2]

    # Emission reward (negative values to minimize)
    emission_reward = -total_co2

    # Combined reward with weights
    # 80% traffic efficiency, 20% emissions
    efficiency_weight = 0.8
    emission_weight = 0.2

    # Normalize to prevent one component from dominating
    norm_efficiency = efficiency_reward / 100.0
    norm_emission = emission_reward / 10.0

    # Combined reward
    reward = (efficiency_weight * norm_efficiency) + (emission_weight * norm_emission)

    return reward


# Simple Q-learning implementation
class SimpleQLearning:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}  # State-action values

    def get_state_key(self, state):
        """Convert state to a hashable representation"""
        if isinstance(state, np.ndarray):
            # Round and convert to tuple for hashing
            return tuple((state * 10).astype(int))
        return str(state)  # Fallback for other types

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)

        # If state not in Q-table, add it
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_space.n

        # Explore or exploit
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()  # Random action
        else:
            return np.argmax(self.q_table[state_key])  # Best action

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values based on experience"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Ensure states exist in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_space.n

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.action_space.n

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Max Q-value for next state
        max_next_q = max(self.q_table[next_state_key])

        # Q-learning formula
        new_q = current_q + self.learning_rate * (
                reward + (self.discount_factor * max_next_q * (1 - done)) - current_q
        )

        # Update Q-value
        self.q_table[state_key][action] = new_q

    def decay_exploration_rate(self, decay_factor=0.995, min_rate=0.01):
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

    # Set up environment
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=f'{results_dir}/metrics',
        use_gui=use_gui,
        num_seconds=3600,
        delta_time=5,
        min_green=5,
        max_green=50,
        reward_fn=sustainable_reward
    )

    print("Environment info:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Traffic signals: {env.ts_ids}")

    # Create agents for each traffic signal
    agents = {}
    for ts in env.ts_ids:
        agents[ts] = SimpleQLearning(
            action_space=env.action_space,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.1  # Initial exploration rate
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
    """Analyze and visualize the simulation results"""
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

        # Plot waiting time, emissions, and rewards
        plt.figure(figsize=(12, 12))

        # Waiting time by episode
        plt.subplot(3, 1, 1)
        episode_waiting = metrics.groupby('episode')['waiting_time'].mean().reset_index()
        plt.plot(episode_waiting['episode'], episode_waiting['waiting_time'])
        plt.xlabel('Episode')
        plt.ylabel('Average Waiting Time')
        plt.title('Traffic Efficiency Improvement')
        plt.grid(True)

        # CO2 emissions by episode
        plt.subplot(3, 1, 2)
        episode_co2 = metrics.groupby('episode')['co2_emission'].mean().reset_index()
        plt.plot(episode_co2['episode'], episode_co2['co2_emission'], color='green')
        plt.xlabel('Episode')
        plt.ylabel('Average CO2 Emissions (g)')
        plt.title('Environmental Impact Reduction')
        plt.grid(True)

        # Rewards by episode
        plt.subplot(3, 1, 3)
        episode_rewards = pd.read_csv(Path(results_dir) / "episode_rewards.csv")
        plt.plot(episode_rewards['episode'], episode_rewards['reward'], color='red')
        plt.xlabel('Episode')
        plt.ylabel('Total Episode Reward')
        plt.title('Learning Progress')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / "sustainability_metrics.png")

        # Generate summary report
        with open(plots_dir / "summary_report.md", "w") as f:
            f.write("# Sustainable Traffic Control with Reinforcement Learning\n\n")
            f.write("## SDG 11: Sustainable Cities and Communities\n\n")

            f.write("This experiment applies reinforcement learning to traffic signal control\n")
            f.write("with a focus on sustainability metrics relevant to SDG 11.\n\n")

            f.write("## Performance Metrics\n\n")

            # Calculate improvement metrics
            first_5_waiting = metrics[metrics['episode'] < 5]['waiting_time'].mean()
            last_5_waiting = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['waiting_time'].mean()
            waiting_improvement = ((first_5_waiting - last_5_waiting) / first_5_waiting) * 100

            first_5_co2 = metrics[metrics['episode'] < 5]['co2_emission'].mean()
            last_5_co2 = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['co2_emission'].mean()
            co2_improvement = ((first_5_co2 - last_5_co2) / first_5_co2) * 100

            first_5_reward = episode_rewards[episode_rewards['episode'] <= 5]['reward'].mean()
            last_5_reward = episode_rewards[episode_rewards['episode'] > episode_rewards['episode'].max() - 5][
                'reward'].mean()
            reward_improvement = ((last_5_reward - first_5_reward) / abs(first_5_reward)) * 100

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
            f.write("1. **Sustainable Transportation**: By optimizing traffic flow and reducing waiting times\n")
            f.write("2. **Environmental Sustainability**: By explicitly considering and reducing vehicle emissions\n")
            f.write("3. **Resource Efficiency**: By making better use of existing infrastructure\n\n")

            f.write("The multi-objective reward function balances these priorities with weights of:\n")
            f.write("- 80% for traffic efficiency (waiting times and queue length)\n")
            f.write("- 20% for environmental impact (CO2 emissions)\n\n")

            f.write("This weighting can be adjusted based on specific urban priorities and needs.\n")

        print(f"Analysis complete! Results saved to {plots_dir}")
    except Exception as e:
        print(f"Error analyzing results: {e}")


if __name__ == "__main__":
    print("Sustainable Traffic Control with SUMO")
    print("====================================")
    print("This script implements a reinforcement learning approach to traffic")
    print("signal control that addresses SDG 11: Sustainable Cities.\n")

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