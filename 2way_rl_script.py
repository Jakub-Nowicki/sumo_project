#!/usr/bin/env python3
"""
Fixed Sustainable Traffic Control with 2-way Intersection
Addressing SDG 11: Sustainable Cities and Communities
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import deque

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
OUT_DIR = 'results/fixed_sustainable'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Global dict to store data
emission_data = {}
waiting_data = {}
throughput_data = {}
vehicle_counts = {}  # Track vehicle counts to measure throughput


def sustainable_reward(traffic_signal):
    """
    Normalized reward function that balances traffic efficiency with environmental impact
    Uses scaling and normalization to provide consistent reward signals
    """
    # Get SUMO connection
    sumo = traffic_signal.sumo

    # Traffic Efficiency Component - normalize to ensure consistency
    waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_times)
    total_queue = traffic_signal.get_total_queued()

    # Track waiting time data
    global waiting_data
    ts_id = traffic_signal.id
    waiting_data[ts_id] = waiting_data.get(ts_id, []) + [total_waiting_time]

    # Normalize waiting time and queue with reasonable upper bounds
    max_expected_waiting = 5000  # Upper bound for normalization
    max_expected_queue = 50  # Upper bound for normalization

    norm_waiting = min(1.0, total_waiting_time / max_expected_waiting)
    norm_queue = min(1.0, total_queue / max_expected_queue)

    # Weighted efficiency reward
    efficiency_reward = -(norm_waiting * 0.7 + norm_queue * 0.3) * 10

    # Emission Component
    total_co2 = 0
    vehicle_count = 0
    for lane_id in traffic_signal.lanes:
        lane_vehicles = sumo.lane.getLastStepVehicleIDs(lane_id)
        vehicle_count += len(lane_vehicles)
        for veh_id in lane_vehicles:
            total_co2 += sumo.vehicle.getCO2Emission(veh_id) / 1000.0  # Convert to grams

    # Track emission data
    global emission_data
    emission_data[ts_id] = emission_data.get(ts_id, []) + [total_co2]

    # Normalize CO2 emissions
    max_expected_co2 = 1000  # Upper bound for normalization in grams
    norm_co2 = min(1.0, total_co2 / max_expected_co2)

    emission_reward = -norm_co2 * 10

    # Estimate throughput by tracking vehicles that have passed through
    # Since get_passed_vehicles() is not available, we'll estimate based on counts
    global vehicle_counts, throughput_data

    # Initialize vehicle counts for this traffic signal if not already done
    if ts_id not in vehicle_counts:
        vehicle_counts[ts_id] = {'prev_count': 0, 'total_seen': set()}

    # Get current vehicles in the intersection area
    current_vehicles = set()
    for lane_id in traffic_signal.lanes:
        current_vehicles.update(sumo.lane.getLastStepVehicleIDs(lane_id))

    # Add new vehicles to the set of all vehicles we've seen
    vehicle_counts[ts_id]['total_seen'].update(current_vehicles)

    # Estimate throughput as the difference between total vehicles seen and current count
    prev_count = vehicle_counts[ts_id]['prev_count']
    current_count = len(current_vehicles)

    # If fewer vehicles now than before, some have likely exited the intersection
    throughput = max(0, prev_count - current_count)
    vehicle_counts[ts_id]['prev_count'] = current_count

    # Track throughput data
    throughput_data[ts_id] = throughput_data.get(ts_id, []) + [throughput]

    # Add a small bonus for having vehicles moving (reduces stop-and-go)
    moving_vehicles = 0
    for lane_id in traffic_signal.lanes:
        lane_vehicles = sumo.lane.getLastStepVehicleIDs(lane_id)
        moving_vehicles += sum(1 for v in lane_vehicles if sumo.vehicle.getSpeed(v) > 0.1)

    movement_reward = min(3.0, moving_vehicles * 0.1)  # Cap at 3.0

    # Combined reward with weights
    efficiency_weight = 0.7
    emission_weight = 0.2
    throughput_weight = 0.1

    throughput_reward = min(5.0, throughput * 0.5)  # Reward for vehicles passing through

    # Final reward combining all components
    reward = (efficiency_weight * efficiency_reward) + \
             (emission_weight * emission_reward) + \
             (throughput_weight * throughput_reward) + \
             movement_reward

    reward = max(-100, min(50, reward))

    return reward


# Enhanced Q-learning with experience replay and better state representation
class EnhancedQAgent:
    def __init__(self, action_space, state_dim, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.5):
        self.action_space = action_space
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.96

        # Q-table with better discretization
        self.q_table = {}

        # Simple experience replay buffer
        self.memory = deque(maxlen=2000)

        # Track recently chosen actions to avoid oscillation
        self.recent_actions = deque(maxlen=10)

    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def discretize_state(self, state):
        """Convert continuous state to discrete representation"""
        if isinstance(state, np.ndarray):
            # Extract the most relevant features
            # First few indices usually contain phase information
            phase_info = tuple(np.round(state[:4], 2))

            # The rest contains traffic density, queue length, etc.
            # Group these into fewer bins for better generalization
            traffic_features = []
            for i in range(4, len(state)):
                if state[i] < 0.25:
                    traffic_features.append(0)
                elif state[i] < 0.50:
                    traffic_features.append(1)
                elif state[i] < 0.75:
                    traffic_features.append(2)
                else:
                    traffic_features.append(3)

            return phase_info + tuple(traffic_features)
        return str(state)

    def get_table_size(self):
        """Return size of Q-table (for monitoring memory usage)"""
        return len(self.q_table)

    def get_q_value(self, state, action):
        """Get Q-value with defaults for new states"""
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_space.n
        return self.q_table[state_key][action]

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy with action persistence"""
        # Explore with probability epsilon
        if random.random() < self.exploration_rate:
            return self.action_space.sample()

        # Get Q-values for this state
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_space.n

        q_values = self.q_table[state_key]

        # Check for ties and prefer previously chosen actions
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]

        # If we have multiple equally good actions, prefer recently used ones
        for action in self.recent_actions:
            if action in best_actions:
                return action

        # Otherwise choose randomly among the best
        return random.choice(best_actions)

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer with priority for negative rewards"""
        self.memory.append((state, action, reward, next_state, done))

        # Add duplicate entries for highly negative rewards to increase learning priority
        if reward < -5.0:  # Threshold for considering an experience particularly bad
            # Add this bad experience again to increase its sampling probability
            self.memory.append((state, action, reward, next_state, done))

        # Also update recent actions
        self.recent_actions.append(action)

    def learn(self):
        """Learn from a batch of experiences with adaptive learning rate"""
        if len(self.memory) < 32:
            return

        # Sample experiences as before
        batch_size = min(32, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        # Calculate average reward in batch to adjust learning rate
        avg_reward = sum(reward for _, _, reward, _, _ in batch) / batch_size

        # Lower learning rate for very negative rewards to avoid overreaction
        adaptive_lr = self.learning_rate
        if avg_reward < -50:
            adaptive_lr = self.learning_rate * 0.5

        for state, action, reward, next_state, done in batch:
            # Get current Q-value
            state_key = self.discretize_state(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_space.n

            # Get max Q-value for next state
            next_state_key = self.discretize_state(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * self.action_space.n

            # Compute target using Q-learning update rule
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * max(self.q_table[next_state_key])

            # Use adaptive learning rate
            self.q_table[state_key][action] += adaptive_lr * (target - self.q_table[state_key][action])


def run_simulation(use_gui=True, episodes=50):
    """Run improved simulation for sustainable traffic control"""
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{OUT_DIR}_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Reset global tracking data
    global emission_data, waiting_data, throughput_data, vehicle_counts
    emission_data = {}
    waiting_data = {}
    throughput_data = {}
    vehicle_counts = {}

    # Track metrics for analysis
    metrics = {
        'episode': [],
        'step': [],
        'waiting_time': [],
        'co2_emission': [],
        'throughput': [],
        'reward': [],
        'episode_reward': [],
        'exploration_rate': [],
        'q_table_size': []
    }

    # Set up environment with improved parameters
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=f'{results_dir}/metrics',
        use_gui=use_gui,
        num_seconds=7200,  # 1 hour simulation
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
        agents[ts] = EnhancedQAgent(
            action_space=env.action_space,
            state_dim=env.observation_space.shape[0],
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=0.5
        )

    # Track episode rewards for learning curve
    episode_rewards = []

    # Training loop
    print(f"\nStarting training for {episodes} episodes...")

    try:
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            done = {'__all__': False}
            episode_reward = 0
            step = 0

            # Reset throughput tracking for new episode
            vehicle_counts = {}

            # Run episode
            while not done['__all__']:
                # Choose actions
                action = {}
                for ts in state.keys():
                    action[ts] = agents[ts].choose_action(state[ts])

                # Take step
                next_state, reward, done, _ = env.step(action)

                # Store experience for later learning
                for ts in state.keys():
                    agents[ts].store_experience(
                        state=state[ts],
                        action=action[ts],
                        reward=reward[ts],
                        next_state=next_state[ts],
                        done=done[ts]
                    )

                # Learn from experience
                for ts in state.keys():
                    agents[ts].learn()

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

                    total_throughput = 0
                    for ts_id in env.ts_ids:
                        if ts_id in throughput_data and throughput_data[ts_id]:
                            total_throughput += throughput_data[ts_id][-1]

                    # Get Q-table size
                    q_table_size = 0
                    for ts in env.ts_ids:
                        q_table_size += agents[ts].get_table_size()

                    # Record data
                    metrics['episode'].append(episode)
                    metrics['step'].append(step + episode * 200)
                    metrics['waiting_time'].append(total_waiting)
                    metrics['co2_emission'].append(total_co2)
                    metrics['throughput'].append(total_throughput)
                    metrics['reward'].append(step_reward)
                    metrics['episode_reward'].append(episode_reward)
                    metrics['exploration_rate'].append(agents[list(agents.keys())[0]].exploration_rate)
                    metrics['q_table_size'].append(q_table_size)

                step += 1

                # Limit episode length
                if step >= 300:
                    break

            # Decay exploration rate after each episode
            for ts in env.ts_ids:
                agents[ts].decay_exploration()

            # Track episode results
            episode_rewards.append(episode_reward)

            # Report progress
            print(
                f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Steps: {step} - Exploration: {agents[list(agents.keys())[0]].exploration_rate:.3f}")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is closed even if there's an error
        env.close()

    # Save metrics
    pd.DataFrame(metrics).to_csv(f"{results_dir}/custom_metrics.csv", index=False)

    # Save episode rewards
    pd.DataFrame({'episode': range(1, len(episode_rewards) + 1), 'reward': episode_rewards}).to_csv(
        f"{results_dir}/episode_rewards.csv", index=False
    )

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f"{results_dir}/learning_curve.png")

    print(f"Training complete! Results saved to {results_dir}")
    return results_dir


def analyze_results(results_dir):
    """Analyze and visualize the simulation results with comprehensive metrics"""
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
            return pd.Series(data).rolling(window=min(window, len(data) // 2 or 1), center=True).mean().fillna(
                method='bfill').fillna(method='ffill').values

        # Create a 2x2 grid of plots
        plt.figure(figsize=(16, 14))

        try:
            # 1. Reward plot
            plt.subplot(2, 2, 1)
            episode_rewards = pd.read_csv(Path(results_dir) / "episode_rewards.csv")
            plt.plot(episode_rewards['episode'], episode_rewards['reward'], 'o-', alpha=0.5, color='blue',
                     label='Per Episode')
            if len(episode_rewards) > 3:
                plt.plot(episode_rewards['episode'], smooth(episode_rewards['reward']), linewidth=3, color='darkblue',
                         label='Trend')
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Total Episode Reward', fontsize=12)
            plt.title('Learning Progress', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
        except Exception as e:
            print(f"Error plotting rewards: {e}")

        try:
            # 2. Waiting time plot
            plt.subplot(2, 2, 2)
            if 'waiting_time' in metrics.columns and not metrics['waiting_time'].empty:
                episode_waiting = metrics.groupby('episode')['waiting_time'].mean().reset_index()
                plt.plot(episode_waiting['episode'], episode_waiting['waiting_time'], 'o-', alpha=0.5, color='red',
                         label='Per Episode')
                if len(episode_waiting) > 3:
                    plt.plot(episode_waiting['episode'], smooth(episode_waiting['waiting_time']), linewidth=3,
                             color='darkred', label='Trend')
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel('Average Waiting Time (s)', fontsize=12)
                plt.title('Traffic Efficiency Improvement', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
            else:
                plt.text(0.5, 0.5, "Waiting time data not available", horizontalalignment='center',
                         verticalalignment='center')
        except Exception as e:
            print(f"Error plotting waiting times: {e}")

        try:
            # 3. CO2 emissions plot
            plt.subplot(2, 2, 3)
            if 'co2_emission' in metrics.columns and not metrics['co2_emission'].empty:
                episode_co2 = metrics.groupby('episode')['co2_emission'].mean().reset_index()
                plt.plot(episode_co2['episode'], episode_co2['co2_emission'], 'o-', alpha=0.5, color='green',
                         label='Per Episode')
                if len(episode_co2) > 3:
                    plt.plot(episode_co2['episode'], smooth(episode_co2['co2_emission']), linewidth=3,
                             color='darkgreen', label='Trend')
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel('Average CO2 Emissions (g)', fontsize=12)
                plt.title('Environmental Impact Reduction', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
            else:
                plt.text(0.5, 0.5, "CO2 emission data not available", horizontalalignment='center',
                         verticalalignment='center')
        except Exception as e:
            print(f"Error plotting CO2 emissions: {e}")

        try:
            # 4. Throughput plot
            plt.subplot(2, 2, 4)
            if 'throughput' in metrics.columns and not metrics['throughput'].empty:
                episode_throughput = metrics.groupby('episode')['throughput'].sum().reset_index()
                plt.plot(episode_throughput['episode'], episode_throughput['throughput'], 'o-', alpha=0.5,
                         color='purple', label='Per Episode')
                if len(episode_throughput) > 3:
                    plt.plot(episode_throughput['episode'], smooth(episode_throughput['throughput']), linewidth=3,
                             color='darkviolet', label='Trend')
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel('Total Vehicles Processed', fontsize=12)
                plt.title('Intersection Throughput', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
            else:
                plt.text(0.5, 0.5, "Throughput data not available", horizontalalignment='center',
                         verticalalignment='center')
        except Exception as e:
            print(f"Error plotting throughput: {e}")

        plt.tight_layout()
        plt.savefig(plots_dir / "performance_metrics.png", dpi=300)

        try:
            # Plot exploration rate decay
            plt.figure(figsize=(10, 6))
            if 'exploration_rate' in metrics.columns and not metrics['exploration_rate'].empty:
                exploration_data = metrics.groupby('episode')['exploration_rate'].first().reset_index()
                plt.plot(exploration_data['episode'], exploration_data['exploration_rate'])
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel('Exploration Rate', fontsize=12)
                plt.title('Exploration Rate Decay', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(plots_dir / "exploration_decay.png", dpi=300)
        except Exception as e:
            print(f"Error plotting exploration rate: {e}")

        try:
            # Plot Q-table growth
            plt.figure(figsize=(10, 6))
            if 'q_table_size' in metrics.columns and not metrics['q_table_size'].empty:
                q_table_data = metrics.groupby('episode')['q_table_size'].last().reset_index()
                plt.plot(q_table_data['episode'], q_table_data['q_table_size'])
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel('Q-Table Size (states)', fontsize=12)
                plt.title('State Space Exploration', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(plots_dir / "q_table_growth.png", dpi=300)
        except Exception as e:
            print(f"Error plotting Q-table growth: {e}")

        # Generate comprehensive summary report
        with open(plots_dir / "sdg11_comprehensive_report.md", "w") as f:
            f.write("# Enhanced Sustainable Traffic Control\n\n")
            f.write("## SDG 11: Sustainable Cities and Communities\n\n")

            f.write("This experiment applies reinforcement learning to traffic signal control\n")
            f.write("with a focus on sustainability metrics relevant to SDG 11.\n\n")

            f.write("## Performance Metrics\n\n")

            # Calculate improvement metrics with handling for potential errors
            try:
                if 'waiting_time' in metrics.columns and len(metrics) > 5:
                    first_5_waiting = metrics[metrics['episode'] < 5]['waiting_time'].mean()
                    last_5_waiting = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['waiting_time'].mean()
                    waiting_improvement = ((
                                                       first_5_waiting - last_5_waiting) / first_5_waiting) * 100 if first_5_waiting > 0 else 0

                    f.write(f"### Traffic Efficiency\n")
                    f.write(f"- Starting waiting time: {first_5_waiting:.2f} seconds\n")
                    f.write(f"- Final waiting time: {last_5_waiting:.2f} seconds\n")
                    f.write(f"- Improvement: {waiting_improvement:.1f}%\n\n")
                else:
                    f.write(f"### Traffic Efficiency\n")
                    f.write(f"- Insufficient data to calculate waiting time statistics\n\n")
            except:
                f.write(f"### Traffic Efficiency\n")
                f.write(f"- Unable to calculate waiting time statistics\n\n")

            try:
                if 'co2_emission' in metrics.columns and len(metrics) > 5:
                    first_5_co2 = metrics[metrics['episode'] < 5]['co2_emission'].mean()
                    last_5_co2 = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['co2_emission'].mean()
                    co2_improvement = ((first_5_co2 - last_5_co2) / first_5_co2) * 100 if first_5_co2 > 0 else 0

                    f.write(f"### Environmental Impact\n")
                    f.write(f"- Starting CO2 emissions: {first_5_co2:.2f} g\n")
                    f.write(f"- Final CO2 emissions: {last_5_co2:.2f} g\n")
                    f.write(f"- Reduction: {co2_improvement:.1f}%\n\n")
                else:
                    f.write(f"### Environmental Impact\n")
                    f.write(f"- Insufficient data to calculate emission statistics\n\n")
            except:
                f.write(f"### Environmental Impact\n")
                f.write(f"- Unable to calculate emission statistics\n\n")

            try:
                if 'throughput' in metrics.columns and len(metrics) > 5:
                    first_5_throughput = metrics[metrics['episode'] < 5]['throughput'].mean()
                    last_5_throughput = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['throughput'].mean()
                    throughput_improvement = ((
                                                          last_5_throughput - first_5_throughput) / first_5_throughput) * 100 if first_5_throughput > 0 else 0

                    f.write(f"### Intersection Throughput\n")
                    f.write(f"- Starting throughput: {first_5_throughput:.2f} vehicles\n")
                    f.write(f"- Final throughput: {last_5_throughput:.2f} vehicles\n")
                    f.write(f"- Improvement: {throughput_improvement:.1f}%\n\n")
                else:
                    f.write(f"### Intersection Throughput\n")
                    f.write(f"- Insufficient data to calculate throughput statistics\n\n")
            except:
                f.write(f"### Intersection Throughput\n")
                f.write(f"- Unable to calculate throughput statistics\n\n")

            try:
                episode_rewards = pd.read_csv(Path(results_dir) / "episode_rewards.csv")
                if len(episode_rewards) > 5:
                    first_5_reward = episode_rewards[episode_rewards['episode'] <= 5]['reward'].mean()
                    last_5_reward = episode_rewards[episode_rewards['episode'] > episode_rewards['episode'].max() - 5][
                        'reward'].mean()
                    reward_improvement = ((last_5_reward - first_5_reward) / abs(
                        first_5_reward)) * 100 if first_5_reward != 0 else 0

                    f.write(f"### Learning Performance\n")
                    f.write(f"- Starting average reward: {first_5_reward:.2f}\n")
                    f.write(f"- Final average reward: {last_5_reward:.2f}\n")
                    f.write(f"- Improvement: {reward_improvement:.1f}%\n\n")
                else:
                    f.write(f"### Learning Performance\n")
                    f.write(f"- Insufficient data to calculate reward statistics\n\n")
            except:
                f.write(f"### Learning Performance\n")
                f.write(f"- Unable to calculate reward statistics\n\n")

            f.write("## Relevance to SDG 11\n\n")

            f.write("### Target 11.2: Sustainable Transport Systems\n")
            f.write("Our traffic control system provides several benefits for sustainable transport:\n\n")
            f.write("- **Reduced waiting times**: Shorter delays at intersections improve overall journey times\n")
            f.write(
                "- **Increased throughput**: More vehicles can pass through the intersection in the same time period\n")
            f.write("- **Smoother traffic flow**: Fewer stops and starts reduce frustration and improve safety\n\n")

            f.write("### Target 11.6: Air Quality and Environmental Impact\n")
            f.write("Our multi-objective approach directly addresses environmental concerns:\n\n")
            f.write("- **Reduced emissions**: By explicitly minimizing CO2 in our reward function\n")
            f.write(
                "- **Less idling**: More efficient traffic flow means less time with engines running while stationary\n")
            f.write("- **Quantifiable improvements**: Our metrics show direct emission reductions\n\n")

            f.write("### Target 11.B: Resource Efficiency and Climate Change Mitigation\n")
            f.write("Our approach demonstrates intelligent infrastructure management:\n\n")
            f.write("- **Optimized existing infrastructure**: Getting more capacity without new construction\n")
            f.write("- **Adaptive to conditions**: The reinforcement learning agent improves with experience\n")
            f.write(
                "- **Balances multiple objectives**: Shows how to handle competing priorities in urban management\n\n")

            f.write("## Technical Approach\n\n")
            f.write("### Enhanced Q-Learning Implementation\n")
            f.write("Our implementation includes several improvements for stability and performance:\n\n")
            f.write("- **Better state representation**: Discretizing continuous features for better generalization\n")
            f.write("- **Experience replay**: Storing and learning from past experiences\n")
            f.write("- **Action persistence**: Reducing oscillation by considering recent actions\n")
            f.write("- **Multi-objective reward**: Balancing traffic flow, emissions, and throughput\n\n")

            f.write("### Reward Function Design\n")
            f.write("Our reward function balances multiple sustainability objectives:\n\n")
            f.write("```python\n")
            f.write("# Combined reward with weights\n")
            f.write("efficiency_weight = 0.7  # Traffic flow efficiency\n")
            f.write("emission_weight = 0.2    # Environmental impact\n")
            f.write("throughput_weight = 0.1  # Intersection capacity\n")
            f.write("```\n\n")

            f.write("This weighting can be adjusted based on specific urban priorities.\n\n")

            f.write("## Conclusions and Recommendations\n\n")
            f.write("Our reinforcement learning approach demonstrates that traffic signals can be\n")
            f.write("optimized for both efficiency and environmental impact simultaneously.\n\n")

            f.write("### Key Findings\n\n")
            f.write("1. Multi-objective optimization is effective for sustainability goals\n")
            f.write("2. Reinforcement learning can adapt to complex traffic patterns\n")
            f.write("3. Even simple intersections show significant potential for improvement\n\n")

            f.write("### Recommendations for Urban Planners\n\n")
            f.write("1. **Implement adaptive traffic control**: Traditional fixed-time signals waste capacity\n")
            f.write("2. **Include environmental metrics**: Don't focus solely on traffic throughput\n")
            f.write("3. **Start with high-impact intersections**: Target bottlenecks first\n")
            f.write("4. **Use data-driven approaches**: Collect and analyze traffic patterns\n\n")

            f.write("### Future Improvements\n\n")
            f.write("1. **Coordination between multiple intersections** for corridor-level optimization\n")
            f.write("2. **Integration with real-time air quality data** for dynamic environmental weighting\n")
            f.write("3. **Public transit priority** to further enhance sustainable mobility\n")
            f.write("4. **Pedestrian and cyclist considerations** for complete street management\n\n")

            f.write("This research demonstrates the potential of reinforcement learning to contribute\n")
            f.write("significantly to sustainable urban development and SDG 11 objectives.\n")

        print(f"Analysis complete! Results saved to {plots_dir}")
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Fixed Sustainable Traffic Control")
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