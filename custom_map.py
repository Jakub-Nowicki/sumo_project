#!/usr/bin/env python3
"""
Basic Custom Map Test with RL
A simplified script to test reinforcement learning on a custom map
"""

import os
import sys
import random
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

NET_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\custom_map\mynetwork.net.xml'
ROUTE_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\custom_map\routes.rou.xml'
# Create output directory
OUT_DIR = 'results/custom_map_test'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# Simple Q-learning agent
class SimpleQLearningAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.5
        self.exploration_decay = 0.99
        self.min_exploration_rate = 0.01

    def get_state_key(self, state):
        """Convert state array to hashable key"""
        # Simplify state representation by rounding values
        return tuple(np.round(state, 1))

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)

        # Explore: choose a random action
        if random.random() < self.exploration_rate:
            return self.action_space.sample()

        # Exploit: choose best known action
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * self.action_space.n

        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * self.action_space.n
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0] * self.action_space.n

        # Q-learning update rule
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * max(self.q_table[next_state_key])

        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])

    def decay_exploration(self):
        """Reduce exploration rate over time"""
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def get_table_size(self):
        """Return number of states in Q-table"""
        return len(self.q_table)


def run_simulation(use_gui=True, episodes=20):
    """Run a simple RL simulation on the custom map"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{OUT_DIR}_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Track metrics
    metrics = {
        'episode': [],
        'reward': []
    }

    # Set up environment
    try:
        print(f"Creating environment with net file: {NET_FILE}")
        print(f"Route file: {ROUTE_FILE}")

        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            out_csv_name=f'{results_dir}/metrics',
            use_gui=use_gui,
            num_seconds=1800,  # 30 minutes
            delta_time=5,  # 5-second time steps
            min_green=5,  # Minimum green time
            max_green=50,  # Maximum green time
            yellow_time=3
        )

        print("Environment created successfully")
        print(f"Traffic signals found: {env.ts_ids}")

        if not env.ts_ids or len(env.ts_ids) == 0:
            print("ERROR: No traffic signals found in the network.")
            print("Checking if network file exists...")
            if os.path.exists(NET_FILE):
                print(f"Network file exists. Size: {os.path.getsize(NET_FILE)} bytes")
            else:
                print("Network file does not exist!")
            print("Checking if route file exists...")
            if os.path.exists(ROUTE_FILE):
                print(f"Route file exists. Size: {os.path.getsize(ROUTE_FILE)} bytes")
            else:
                print("Route file does not exist!")
            return None

        # Create agents for each traffic signal
        agents = {}
        for ts in env.ts_ids:
            agents[ts] = SimpleQLearningAgent(env.action_space)

        # Track episode rewards
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
                episode_reward += sum(reward.values())
                step += 1

                # Limit episode length for testing
                if step >= 200:
                    break

            # Decay exploration rate after each episode
            for ts in env.ts_ids:
                agents[ts].decay_exploration()

            # Track episode results
            episode_rewards.append(episode_reward)

            # Record metrics
            metrics['episode'].append(episode + 1)
            metrics['reward'].append(episode_reward)

            # Report progress
            print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Steps: {step}")

        # Save metrics
        pd.DataFrame(metrics).to_csv(f"{results_dir}/episode_rewards.csv", index=False)

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

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Ensure environment is closed even if there's an error
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    print("Basic Custom Map RL Test")
    print("============================")
    print("Testing reinforcement learning on a custom map\n")

    # Ask for GUI mode
    use_gui = input("Run with GUI? (y/n): ").lower().startswith('y')

    # Ask for number of episodes
    episodes = 20
    try:
        episodes = int(input(f"Number of episodes (default: {episodes}): ") or episodes)
    except ValueError:
        print(f"Using default value: {episodes}")

    # Run simulation
    results_dir = run_simulation(use_gui=use_gui, episodes=episodes)