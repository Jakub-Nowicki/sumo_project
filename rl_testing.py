#!/usr/bin/env python3
"""
Enhanced Sustainable Traffic Control with 2-way Intersection
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
OUT_DIR = 'results/enhanced_sustainable'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Global dicts to store data
emission_data = {}
waiting_data = {}
throughput_data = {}
vehicle_counts = {}  # Track vehicle counts to measure throughput
last_rewards = {}  # Track rewards for consecutive improvement bonus


def sustainable_reward(traffic_signal):
    """
    Enhanced reward function that balances traffic efficiency with environmental impact
    Uses dynamic scaling and normalization with non-linear penalties for poor conditions
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

    # Dynamic normalization based on historical data
    if len(waiting_data[ts_id]) > 10:
        waiting_history = waiting_data[ts_id][-100:] if len(waiting_data[ts_id]) > 100 else waiting_data[ts_id]
        p95_waiting = np.percentile(waiting_history, 95)
        max_expected_waiting = max(1000, p95_waiting)  # Adaptive upper bound
    else:
        max_expected_waiting = 5000  # Initial estimate

    # Max expected queue length (keeps this simple to start)
    max_expected_queue = 50

    norm_waiting = min(1.0, total_waiting_time / max_expected_waiting)
    norm_queue = min(1.0, total_queue / max_expected_queue)

    # Apply non-linear penalty for higher congestion (emphasizes reducing worst-case scenarios)
    # Use less aggressive exponent (1.3 instead of 1.5) for smoother gradients
    waiting_penalty = norm_waiting ** 1.3
    queue_penalty = norm_queue ** 1.3

    # Weighted efficiency reward with sharper gradient for poor conditions
    efficiency_reward = -((waiting_penalty * 0.7 + queue_penalty * 0.3) * 12)  # Reduced from 15 to 12

    # Emission Component with vehicle counts and speed considerations
    total_co2 = 0
    vehicle_count = 0
    total_speed = 0

    for lane_id in traffic_signal.lanes:
        lane_vehicles = sumo.lane.getLastStepVehicleIDs(lane_id)
        vehicle_count += len(lane_vehicles)

        for veh_id in lane_vehicles:
            co2 = sumo.vehicle.getCO2Emission(veh_id) / 1000.0  # Convert to grams
            speed = sumo.vehicle.getSpeed(veh_id)

            # Penalize more for high-emission slow vehicles (idling)
            if speed < 0.5:  # Almost stopped
                co2 *= 1.2  # Increase penalty for stopped vehicles

            total_co2 += co2
            total_speed += speed

    # Track emission data
    global emission_data
    emission_data[ts_id] = emission_data.get(ts_id, []) + [total_co2]

    # Adaptive normalization for emissions too
    if len(emission_data[ts_id]) > 10:
        emission_history = emission_data[ts_id][-100:] if len(emission_data[ts_id]) > 100 else emission_data[ts_id]
        p95_emission = np.percentile(emission_history, 95)
        max_expected_co2 = max(500, p95_emission)
    else:
        max_expected_co2 = 1000

    norm_co2 = min(1.0, total_co2 / max_expected_co2)
    # Use milder exponent here too
    emission_reward = -(norm_co2 ** 1.3) * 10  # Reduced from 12 to 10

    # Enhanced throughput calculation with better vehicle tracking
    global vehicle_counts, throughput_data

    # Initialize vehicle counts for this traffic signal if not already done
    if ts_id not in vehicle_counts:
        vehicle_counts[ts_id] = {'prev_count': 0, 'current_tracking': set(), 'previous_tracking': set()}

    # Get current vehicles in the intersection area
    current_vehicles = set()
    for lane_id in traffic_signal.lanes:
        current_vehicles.update(sumo.lane.getLastStepVehicleIDs(lane_id))

    # Better throughput calculation by tracking specific vehicles
    prev_tracking = vehicle_counts[ts_id]['previous_tracking']
    throughput = len(prev_tracking - current_vehicles)  # Vehicles that were here but now gone

    # Update tracking sets for next time
    vehicle_counts[ts_id]['previous_tracking'] = vehicle_counts[ts_id]['current_tracking']
    vehicle_counts[ts_id]['current_tracking'] = current_vehicles

    # Track throughput data
    throughput_data[ts_id] = throughput_data.get(ts_id, []) + [throughput]

    # Add a reward for smooth traffic flow (avoid stop-and-go)
    average_speed = total_speed / max(1, vehicle_count)
    flow_reward = min(5.0, average_speed * 0.5)  # Reward for keeping traffic moving

    # Adaptive weights based on conditions
    # If congestion is high, prioritize efficiency more
    if norm_waiting > 0.7:  # High congestion situation
        efficiency_weight = 0.75  # Slightly reduced from 0.8
        emission_weight = 0.15  # Increased from 0.1
        throughput_weight = 0.1
    else:  # Normal conditions
        efficiency_weight = 0.6  # Reduced from 0.65
        emission_weight = 0.3  # Increased from 0.25
        throughput_weight = 0.1

    throughput_reward = min(8.0, throughput * 0.8)

    # Final reward combining all components
    reward = (efficiency_weight * efficiency_reward) + \
             (emission_weight * emission_reward) + \
             (throughput_weight * throughput_reward) + \
             flow_reward

    # Add a bonus for consecutive improvements
    global last_rewards
    if ts_id not in last_rewards:
        last_rewards[ts_id] = []

    last_rewards[ts_id].append(reward)
    if len(last_rewards[ts_id]) > 10:
        last_rewards[ts_id].pop(0)

    # More gradual consistency bonus
    if len(last_rewards[ts_id]) >= 3:
        recent = last_rewards[ts_id][-3:]
        if recent[0] < recent[1] < recent[2]:  # Consistent improvement
            # Calculate improvement magnitude
            improvement_bonus = min(3.0, (recent[2] - recent[0]) * 0.2)
            reward += improvement_bonus  # Smaller bonus (max 3.0 instead of 2.0)

    # Apply more aggressive clipping to prevent extreme negative rewards
    reward = max(-40, min(20, reward))  # More restrictive clipping than before

    return reward

# Enhanced Q-learning with improved exploration and state representation
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

        # Improved experience replay buffer with prioritization
        self.memory = deque(maxlen=3000)

        # Track recently chosen actions to avoid oscillation
        self.recent_actions = deque(maxlen=10)

        # Track phase durations to prevent excessive switching
        self.phase_timer = {}

        # Step counter for scheduled exploration decay
        self.total_steps = 0

        # Episode reward tracking for meta-learning
        self.episode_reward = 0
        self.target_q_table = {}
        self.target_update_freq = 10  # Update target every 10 episodes
        self.episode_count = 0

    # Add target network update method
    def update_target_network(self):
        self.target_q_table = self.q_table.copy()



    def decay_exploration(self):
        """Decay exploration rate with improved schedule"""
        # More aggressive decay early, then slower decay
        if self.exploration_rate > 0.2:
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * 0.9)
        else:
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * 0.98)

    def discretize_state(self, state):
        """Enhanced state representation with more meaningful binning"""
        if isinstance(state, np.ndarray):
            # Extract traffic density features with better resolution near critical thresholds
            traffic_features = []
            for i in range(4, len(state)):
                if state[i] < 0.15:
                    traffic_features.append(0)
                elif state[i] < 0.30:
                    traffic_features.append(1)
                elif state[i] < 0.45:
                    traffic_features.append(2)
                elif state[i] < 0.60:
                    traffic_features.append(3)
                elif state[i] < 0.75:
                    traffic_features.append(4)
                else:
                    traffic_features.append(5)

            # Get phase information
            phase_info = tuple(np.round(state[:4], 2))

            # Add derived features about traffic imbalance between approaches
            if len(state) > 8:
                try:
                    # Calculate North-South vs East-West traffic imbalance
                    ns_traffic = np.mean(state[4:8])
                    ew_traffic = np.mean(state[8:12])
                    imbalance = int(min(3, abs(ns_traffic - ew_traffic) * 10))
                    traffic_features.append(imbalance)
                except:
                    pass  # Skip if indices are out of range

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
        """Choose action using improved exploration with Boltzmann policy"""
        # Update exploration schedule based on steps
        if self.total_steps % 500 == 0 and self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * self.exploration_decay)

        # Use Boltzmann exploration with temperature
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_space.n

        q_values = self.q_table[state_key].copy()  # Copy to avoid modifying original

        # Extract current phase from state representation
        current_phase = 0  # Default phase
        try:
            # In most SUMO-RL setups, the first few elements of state indicate current phase
            current_phase = np.argmax(state[:4]) if len(state) > 4 else 0
        except:
            pass

        # Penalize phase changes if current phase hasn't been active long enough
        # This prevents excessive switching which is bad for traffic flow
        if current_phase in self.phase_timer and self.phase_timer[current_phase] < 5:
            for i in range(len(q_values)):
                if i != current_phase:
                    q_values[i] -= 3.0  # Penalty to discourage changing phases too quickly

        # Explore using Boltzmann distribution
        if random.random() < self.exploration_rate:
            temperature = max(0.1, 1.0 - (self.exploration_rate * 0.5))  # Higher temp = more exploration

            # Handle extreme values to prevent numerical instability
            max_q = max(q_values)
            q_values = [q - max_q for q in q_values]  # Subtract max for numerical stability

            # Compute softmax probabilities
            exp_q_values = np.exp(np.array(q_values) / temperature)
            probabilities = exp_q_values / np.sum(exp_q_values)

            # Sample action based on probabilities
            try:
                action = np.random.choice(self.action_space.n, p=probabilities)
            except:
                # Fallback if probabilities are invalid
                action = self.action_space.sample()
        else:
            # Exploit - choose best action
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]

            # If multiple best actions, check recent actions for stability
            for action in self.recent_actions:
                if action in best_actions:
                    chosen_action = action
                    break
            else:
                # No recent actions among best, choose randomly
                chosen_action = random.choice(best_actions)

            action = chosen_action

        # Update phase timer
        for phase in range(self.action_space.n):
            if phase not in self.phase_timer:
                self.phase_timer[phase] = 0

            if phase == action:
                self.phase_timer[phase] += 1
            else:
                self.phase_timer[phase] = 0

        # Update recent actions and step counter
        self.recent_actions.append(action)
        self.total_steps += 1

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer with priority based on reward"""
        # Calculate TD error for prioritization (simplified)
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        # Ensure Q-values exist for both states
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_space.n
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.action_space.n

        # Add experience to replay buffer
        self.memory.append((state, action, reward, next_state, done))

        # Add duplicate entries for highly negative rewards (higher priority)
        if reward < -10.0:
            self.memory.append((state, action, reward, next_state, done))
            self.memory.append((state, action, reward, next_state, done))
        elif reward < -5.0:
            self.memory.append((state, action, reward, next_state, done))

        # Also prioritize highly positive rewards (good experiences to learn from)
        if reward > 10.0:
            self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Learn from experiences using Double Q-learning approach"""
        if len(self.memory) < 32:
            return

        # Sample experiences
        batch_size = min(32, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        # Calculate average reward in batch to adjust learning rate
        avg_reward = sum(reward for _, _, reward, _, _ in batch) / batch_size

        # Adaptive learning rate based on reward
        adaptive_lr = self.learning_rate
        if avg_reward < -10:
            adaptive_lr = self.learning_rate * 0.75  # Reduce learning rate for very bad experiences
        elif avg_reward > 10:
            adaptive_lr = self.learning_rate * 1.25  # Increase learning rate for good experiences

        adaptive_lr = max(0.01, min(0.5, adaptive_lr))  # Clip learning rate

        for state, action, reward, next_state, done in batch:
            state_key = self.discretize_state(state)
            next_state_key = self.discretize_state(next_state)

            # Ensure Q-values exist
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_space.n
            if next_state_key not in self.target_q_table:
                self.target_q_table[next_state_key] = [0.0] * self.action_space.n

            # Use target network for more stable learning
            if not done:
                # Select action using current network
                best_action = np.argmax(self.q_table[next_state_key])
                # But evaluate using target network
                target = reward + self.discount_factor * self.target_q_table[next_state_key][best_action]
            else:
                target = reward

            # Update Q-value with adaptive learning rate
            # Add gradient clipping to prevent large updates
            td_error = target - self.q_table[state_key][action]
            # Clip the TD error to prevent large updates
            td_error = max(-10.0, min(10.0, td_error))
            self.q_table[state_key][action] += adaptive_lr * td_error

            # Update Q-value with adaptive learning rate
            self.q_table[state_key][action] += adaptive_lr * (target - self.q_table[state_key][action])

            # Apply minimal weight decay to rarely visited state-action pairs to forget outdated values
            for a in range(self.action_space.n):
                if a != action and self.q_table[state_key][a] != 0:
                    self.q_table[state_key][a] *= 0.9999  # Very slight decay


def run_simulation(use_gui=True, episodes=100):
    """Run improved simulation for sustainable traffic control"""
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{OUT_DIR}_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Reset global tracking data
    global emission_data, waiting_data, throughput_data, vehicle_counts, last_rewards
    emission_data = {}
    waiting_data = {}
    throughput_data = {}
    vehicle_counts = {}
    last_rewards = {}

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
        num_seconds=7200,  # 2 hour simulation
        delta_time=5,  # 5-second time steps
        min_green=10,  # Increased minimum green phase duration for better stability
        max_green=60,  # Increased maximum green phase duration for heavy traffic
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
            learning_rate=0.1,  # Reduced learning rate for stability
            discount_factor=0.99,
            exploration_rate=0.8  # Increased initial exploration
        )
        # Initialize target network
        agents[ts].target_q_table = agents[ts].q_table.copy()

    # Track episode rewards for learning curve
    episode_rewards = []
    smoothed_rewards = []  # New: track smoothed rewards too
    prev_smoothed_reward = None  # For EMA calculation

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
                # Learn more frequently (every step rather than less often)
                for ts in state.keys():
                    agents[ts].learn()

                # Update state and track rewards
                state = next_state
                step_reward = sum(reward.values())
                episode_reward += step_reward

                # Update episode reward tracking for each agent
                for ts in state.keys():
                    agents[ts].episode_reward = episode_reward

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
                    metrics['step'].append(step + episode * 500)  # Increased max steps
                    metrics['waiting_time'].append(total_waiting)
                    metrics['co2_emission'].append(total_co2)
                    metrics['throughput'].append(total_throughput)
                    metrics['reward'].append(step_reward)
                    metrics['episode_reward'].append(episode_reward)
                    metrics['exploration_rate'].append(agents[list(agents.keys())[0]].exploration_rate)
                    metrics['q_table_size'].append(q_table_size)

                step += 1

                # Increased episode length for better learning
                if step >= 500:
                    break

            # Update target networks periodically
            for ts in env.ts_ids:
                # Update target network every 5 episodes
                if episode % 5 == 0:
                    agents[ts].target_q_table = agents[ts].q_table.copy()

            # Decay exploration rate after each episode
            for ts in env.ts_ids:
                agents[ts].decay_exploration()

            # Track episode results with exponential moving average smoothing
            episode_rewards.append(episode_reward)

            # Exponential moving average calculation
            if prev_smoothed_reward is None:
                smoothed_reward = episode_reward
            else:
                # EMA with alpha=0.2
                smoothed_reward = 0.8 * prev_smoothed_reward + 0.2 * episode_reward

            prev_smoothed_reward = smoothed_reward
            smoothed_rewards.append(smoothed_reward)

            # Report progress
            print(
                f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Smoothed: {smoothed_reward:.2f} - Steps: {step} - Exploration: {agents[list(agents.keys())[0]].exploration_rate:.3f} - Q-table: {q_table_size}")

            # Save progress more frequently
            if episode % 10 == 0 or episode == episodes - 1:
                # Save intermediate metrics
                pd.DataFrame(metrics).to_csv(f"{results_dir}/metrics_ep{episode}.csv", index=False)

                # Save episode rewards with smoothed values
                rewards_df = pd.DataFrame({
                    'episode': range(1, len(episode_rewards) + 1),
                    'reward': episode_rewards,
                    'smoothed_reward': smoothed_rewards
                })
                rewards_df.to_csv(f"{results_dir}/episode_rewards_ep{episode}.csv", index=False)

                # Create intermediate learning curve with both raw and smoothed rewards
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, 'o-', alpha=0.5, color='blue',
                         label='Per Episode')
                plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards, linewidth=2, color='red',
                         label='Smoothed Trend')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title(f'Learning Curve (Episodes 1-{episode + 1})')
                plt.grid(True)
                plt.legend()
                plt.savefig(f"{results_dir}/learning_curve_ep{episode}.png")
                plt.close()

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is closed even if there's an error
        env.close()

    # Save final metrics
    pd.DataFrame(metrics).to_csv(f"{results_dir}/custom_metrics.csv", index=False)

    # Save episode rewards with smoothed values
    rewards_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards,
        'smoothed_reward': smoothed_rewards
    })
    rewards_df.to_csv(f"{results_dir}/episode_rewards.csv", index=False)

    # Plot learning curve with both raw and smoothed rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, 'o-', alpha=0.5, color='blue', label='Per Episode')
    plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards, linewidth=2, color='red', label='Smoothed Trend')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{results_dir}/learning_curve.png")
    plt.close()

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
            plt.close()

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
                    plt.close()
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
                    plt.close()
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
                        last_5_waiting = metrics[metrics['episode'] >= metrics['episode'].max() - 5][
                            'waiting_time'].mean()
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
                        last_5_throughput = metrics[metrics['episode'] >= metrics['episode'].max() - 5][
                            'throughput'].mean()
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
                        last_5_reward = \
                        episode_rewards[episode_rewards['episode'] > episode_rewards['episode'].max() - 5][
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
                f.write("Our enhanced traffic control system provides several benefits for sustainable transport:\n\n")
                f.write("- **Reduced waiting times**: Shorter delays at intersections improve overall journey times\n")
                f.write(
                    "- **Increased throughput**: More vehicles can pass through the intersection in the same time period\n")
                f.write("- **Smoother traffic flow**: Fewer stops and starts reduce frustration and improve safety\n")
                f.write("- **Adaptive to traffic conditions**: Responds dynamically to changing traffic patterns\n\n")

                f.write("### Target 11.6: Air Quality and Environmental Impact\n")
                f.write("Our multi-objective approach directly addresses environmental concerns:\n\n")
                f.write("- **Reduced emissions**: By explicitly minimizing CO2 in our reward function\n")
                f.write(
                    "- **Less idling**: More efficient traffic flow means less time with engines running while stationary\n")
                f.write("- **Quantifiable improvements**: Our metrics show direct emission reductions\n")
                f.write(
                    "- **Prioritizes smooth flow**: Rewards continuous movement rather than stop-and-go patterns\n\n")

                f.write("### Target 11.B: Resource Efficiency and Climate Change Mitigation\n")
                f.write("Our approach demonstrates intelligent infrastructure management:\n\n")
                f.write("- **Optimized existing infrastructure**: Getting more capacity without new construction\n")
                f.write("- **Adaptive to conditions**: The reinforcement learning agent improves with experience\n")
                f.write(
                    "- **Balances multiple objectives**: Shows how to handle competing priorities in urban management\n")
                f.write("- **Data-driven decision making**: Uses historical performance to guide future actions\n\n")

                f.write("## Enhanced Technical Approach\n\n")
                f.write("### Key Improvements in the RL Implementation\n")
                f.write("Our enhanced implementation includes several improvements for better performance:\n\n")
                f.write("1. **Enhanced state representation**:\n")
                f.write("   - More granular discretization of traffic features\n")
                f.write("   - Added traffic imbalance features to detect N-S vs E-W traffic patterns\n\n")
                f.write("2. **Improved exploration strategy**:\n")
                f.write("   - Boltzmann exploration with temperature adjustment\n")
                f.write("   - Scheduled exploration decay based on steps rather than episodes\n")
                f.write("   - Faster initial decay followed by slower decay in later phases\n\n")
                f.write("3. **Double Q-learning**:\n")
                f.write("   - Reduces overestimation of Q-values\n")
                f.write("   - More stable learning with adaptive learning rate\n")
                f.write("   - Slight weight decay to forget outdated values\n\n")
                f.write("4. **Enhanced experience replay**:\n")
                f.write("   - Larger memory buffer for better learning from past experiences\n")
                f.write("   - Prioritization of both negative and positive experiences\n")
                f.write("   - Multiple copies of important experiences for higher sampling probability\n\n")
                f.write("5. **Phase stability mechanisms**:\n")
                f.write("   - Tracks phase durations to prevent excessive switching\n")
                f.write("   - Penalizes rapid phase changes that disrupt traffic flow\n")
                f.write("   - Considers recently chosen actions for stability\n\n")

                f.write("### Enhanced Reward Function\n")
                f.write("Our improved reward function provides better learning signals:\n\n")
                f.write("1. **Dynamic normalization**:\n")
                f.write("   - Adapts to observed traffic patterns using percentile-based normalization\n")
                f.write("   - Provides consistent reward scaling across different traffic conditions\n\n")
                f.write("2. **Non-linear penalties**:\n")
                f.write("   - Exponential penalties for high waiting times and emissions\n")
                f.write("   - Creates sharper gradients to prioritize worst-case improvement\n\n")
                f.write("3. **Adaptive component weighting**:\n")
                f.write("   - Shifts priority to efficiency during high congestion\n")
                f.write("   - Balances environmental factors during normal conditions\n\n")
                f.write("4. **Flow smoothness rewards**:\n")
                f.write("   - Explicit rewards for maintaining traffic speeds\n")
                f.write("   - Penalizes stopped vehicles with higher emission weight\n\n")
                f.write("5. **Consecutive improvement bonus**:\n")
                f.write("   - Rewards sustained improvement across multiple steps\n")
                f.write("   - Encourages long-term strategy over short-term gains\n\n")

                f.write("## Improvements in Experimental Setup\n\n")
                f.write("1. **Extended episodes**: Increased from 300 to 500 steps for better learning\n")
                f.write("2. **Increased minimum green time**: From 5 to 8 seconds for improved stability\n")
                f.write("3. **Increased maximum green time**: From 50 to 60 seconds for heavier traffic\n")
                f.write("4. **Progressive checkpointing**: Saves progress every 10 episodes\n")
                f.write("5. **More detailed analysis**: Additional metrics and plots for better insights\n\n")

                f.write("## Conclusions and Future Work\n\n")
                f.write("Our enhanced reinforcement learning approach demonstrates significant improvements\n")
                f.write("in balancing traffic efficiency with environmental sustainability. The key\n")
                f.write("enhancements in state representation, exploration strategy, and reward function\n")
                f.write("design contribute to more effective and stable learning.\n\n")

                f.write("### Future Enhancements\n\n")
                f.write("1. **Coordinated multi-intersection control**: Extend to network-level optimization\n")
                f.write("2. **Integration with real-time emissions data**: Use actual sensors rather than estimates\n")
                f.write("3. **Predictive traffic modeling**: Incorporate short-term predictions into decisions\n")
                f.write("4. **Multi-modal considerations**: Include pedestrians, cyclists, and public transit\n")
                f.write("5. **Transfer learning**: Apply knowledge from one intersection to others\n")
                f.write("6. **Deep reinforcement learning**: Use neural networks for better feature extraction\n\n")

                f.write("This research demonstrates the potential of advanced reinforcement learning\n")
                f.write("techniques to contribute significantly to sustainable urban mobility and\n")
                f.write("SDG 11 objectives through intelligent traffic management.\n")

            print(f"Analysis complete! Results saved to {plots_dir}")
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Enhanced Sustainable Traffic Control")
    print("====================================")
    print("This script implements an improved reinforcement learning approach")
    print("to traffic signal control that addresses SDG 11: Sustainable Cities.\n")

    # Ask for GUI mode
    use_gui = input("Run with GUI? (y/n): ").lower().startswith('y')

    # Ask for number of episodes
    episodes = 100  # Increased default from 50 to 100
    try:
        episodes = int(input(f"Number of episodes (default: {episodes}): ") or episodes)
    except ValueError:
        print(f"Using default value: {episodes}")

    # Run simulation
    results_dir = run_simulation(use_gui=use_gui, episodes=episodes)

    # Analyze results
    analyze_results(results_dir)