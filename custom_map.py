#!/usr/bin/env python3
"""
Sustainable Traffic Control using a custom OSM map
Addressing SDG 11: Sustainable Cities and Communities
With improved OSM handling and error management
"""

import os
import sys
import subprocess
import argparse
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

# Paths for OSM and generated files
OSM_FILE = "map.osm"
MAP_DIR = "osm_map"
NET_FILE = os.path.join(MAP_DIR, "osm.net.xml")
TLS_FILE = os.path.join(MAP_DIR, "osm.tll.xml")  # Traffic light settings
POLY_FILE = os.path.join(MAP_DIR, "osm.poly.xml")
ADD_FILE = os.path.join(MAP_DIR, "osm.add.xml")  # Additional configurations
ROUTE_FILE = os.path.join(MAP_DIR, "osm.rou.xml")
VEHICLE_TYPES_FILE = os.path.join(MAP_DIR, "vehicle_types.add.xml")

# Create output directory
OUT_DIR = 'results/custom_osm'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(MAP_DIR).mkdir(parents=True, exist_ok=True)

# Global dict to store data
emission_data = {}
waiting_data = {}


def create_vehicle_types_file():
    """Create a file with custom vehicle types with emission properties"""
    with open(VEHICLE_TYPES_FILE, 'w') as f:
        f.write("""<additional>
    <vType id="passenger" vClass="passenger" emissionClass="HBEFA3/PC_G_EU4" color="0,0.5,1"/>
    <vType id="bus" vClass="bus" emissionClass="HBEFA3/Bus" color="1,0,0"/>
    <vType id="truck" vClass="truck" emissionClass="HBEFA3/HDV" color="0.5,0.5,0.5"/>
    <vType id="motorcycle" vClass="motorcycle" emissionClass="HBEFA3/PC_G_EU4" color="1,0.5,0"/>
    <vType id="bicycle" vClass="bicycle" emissionClass="zero" color="0,1,0"/>
</additional>""")
    return VEHICLE_TYPES_FILE


def convert_osm_to_sumo():
    """Convert OSM file to SUMO network files with improved error handling"""
    print("Converting OSM file to SUMO network...")

    # Check if OSM file exists
    if not os.path.exists(OSM_FILE):
        print(f"Error: OSM file '{OSM_FILE}' not found.")
        return False

    try:
        # Create vehicle types file for emissions
        create_vehicle_types_file()

        # Step 1: Use netconvert with better options for reliable network
        netconvert_cmd = [
            "netconvert",
            "--osm", OSM_FILE,
            "--output-file", NET_FILE,
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--ramps.guess", "true",
            "--junctions.join", "true",
            "--tls.guess", "true",
            "--tls.discard-simple", "true",
            "--tls.join", "true",
            "--tls.default-type", "static",
            "--tls.guess-signals", "true",
            "--tls.red.time", "30",
            "--tls.green.time", "30",
            "--output.street-names", "true",
            "--output.original-names", "true",
            "--ignore-errors.connections", "true",
            "--ignore-errors", "true"
        ]
        print("Running netconvert...")
        subprocess.run(netconvert_cmd, check=True)

        # Step 2: Generate polygons for visualization
        polyconvert_cmd = [
            "polyconvert",
            "--net-file", NET_FILE,
            "--osm-files", OSM_FILE,
            "--output-file", POLY_FILE
        ]
        print("Running polyconvert...")
        subprocess.run(polyconvert_cmd, check=True)

        # Step 3: Use SUMO's built-in randomTrips to generate demand
        randomTrips_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        trips_file = os.path.join(MAP_DIR, "osm.trips.xml")

        # Generate a lower number of trips with higher period (less dense traffic)
        # This helps avoid route connectivity issues
        print("Generating random trips...")
        randomTrips_cmd = [
            sys.executable, randomTrips_path,
            "-n", NET_FILE,
            "-o", trips_file,
            "--random",
            "-p", "3",  # Higher period = fewer vehicles
            "-e", "1000",  # End time
            "--fringe-factor", "5",  # More trips from/to fringe
            "--min-distance", "300",  # Longer trips
            "--route-file", ROUTE_FILE,
            "--validate",
            "--ignore-errors",
            "--vehicle-class", "passenger",  # Only passenger vehicles for simplicity
            "--vclass", "passenger",
            "--prefix", "veh",
            "--verbose"
        ]
        subprocess.run(randomTrips_cmd, check=True)

        # Create an additional file that includes vehicle types
        with open(ADD_FILE, 'w') as f:
            f.write(f"""<additional>
    <include href="{VEHICLE_TYPES_FILE}"/>
</additional>""")

        print("SUMO network and routes generated successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def sustainable_reward(traffic_signal):
    """
    Custom reward function for sustainable traffic control
    Balances waiting time reduction with emission reduction
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

    # Efficiency reward (negative values to minimize)
    efficiency_reward = -(total_waiting_time + total_queue)

    # Emission Component
    total_co2 = 0
    vehicle_ids = []
    for lane_id in traffic_signal.lanes:
        vehicle_ids.extend(sumo.lane.getLastStepVehicleIDs(lane_id))

    # Calculate total CO2 emissions
    for veh_id in vehicle_ids:
        total_co2 += sumo.vehicle.getCO2Emission(veh_id)  # in mg

    # Convert to grams for better scale
    total_co2 = total_co2 / 1000.0

    # Track emission data
    global emission_data
    emission_data[ts_id] = emission_data.get(ts_id, []) + [total_co2]

    # Emission reward (negative values to minimize)
    emission_reward = -total_co2

    # Combined reward with weights (80% efficiency, 20% emissions)
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
        self.q_table = {}

    def get_state_key(self, state):
        """Convert state to a hashable representation"""
        if isinstance(state, np.ndarray):
            # Round and convert to tuple for hashing
            return tuple((state * 10).astype(int))
        return str(state)

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


def run_simulation(use_gui=True, episodes=30):
    """Run RL simulation for sustainable traffic control using OSM map"""
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

    try:
        # Set up environment with additional parameters to handle OSM issues
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            out_csv_name=f'{results_dir}/metrics',
            use_gui=use_gui,
            num_seconds=1000,
            delta_time=5,
            min_green=5,
            max_green=50,
            reward_fn=sustainable_reward,
            additional_sumo_cmd=[
                "--ignore-route-errors", "true",
                "--time-to-teleport", "300",  # Allow teleporting after long waits
                "--collision.mingap-factor", "0",  # Be more permissive with collisions
                "--collision.action", "warn",  # Only warn on collisions
                "--scale", "0.5",  # Scale down demand to reduce congestion
                "--max-depart-delay", "300",  # Allow delayed departures
                "--time-to-impatience", "60"  # Quicker rerouting on congestion
            ]
        )

        print("Environment info:")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Traffic signals: {env.ts_ids}")

        # If no traffic signals found, we cannot proceed
        if not env.ts_ids:
            print("Warning: No traffic signals found in the network!")
            print("The simulation will run but no learning will occur.")

        # Create agents for each traffic signal
        agents = {}
        for ts in env.ts_ids:
            agents[ts] = SimpleQLearning(
                action_space=env.action_space,
                learning_rate=0.1,
                discount_factor=0.95,
                exploration_rate=0.1
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
            if len(agents) > 0:
                sample_ts = list(agents.keys())[0]
                exploration_rate = agents[sample_ts].exploration_rate
                print(
                    f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Steps: {step} - Exploration: {exploration_rate:.3f}")
            else:
                print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Steps: {step}")

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

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
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
            f.write("on a real-world map, with a focus on sustainability metrics relevant to SDG 11.\n\n")

            # Calculate improvement metrics if we have enough data
            if len(metrics['episode'].unique()) >= 5:
                first_5_waiting = metrics[metrics['episode'] < 5]['waiting_time'].mean()
                last_5_waiting = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['waiting_time'].mean()
                waiting_improvement = ((
                                                   first_5_waiting - last_5_waiting) / first_5_waiting) * 100 if first_5_waiting > 0 else 0

                first_5_co2 = metrics[metrics['episode'] < 5]['co2_emission'].mean()
                last_5_co2 = metrics[metrics['episode'] >= metrics['episode'].max() - 5]['co2_emission'].mean()
                co2_improvement = ((first_5_co2 - last_5_co2) / first_5_co2) * 100 if first_5_co2 > 0 else 0

                first_5_reward = episode_rewards[episode_rewards['episode'] <= 5]['reward'].mean()
                last_5_reward = episode_rewards[episode_rewards['episode'] > episode_rewards['episode'].max() - 5][
                    'reward'].mean()
                reward_improvement = ((last_5_reward - first_5_reward) / abs(
                    first_5_reward)) * 100 if first_5_reward != 0 else 0

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
            else:
                f.write("Not enough episodes to calculate improvement metrics.\n\n")

            f.write("## SDG 11 Relevance\n\n")
            f.write("This implementation addresses the following SDG 11 targets:\n\n")
            f.write("1. **Sustainable Transportation**: By optimizing traffic flow and reducing waiting times\n")
            f.write("2. **Environmental Sustainability**: By explicitly considering and reducing vehicle emissions\n")
            f.write("3. **Resource Efficiency**: By making better use of existing infrastructure\n\n")

            f.write("The multi-objective reward function balances these priorities with weights of:\n")
            f.write("- 80% for traffic efficiency (waiting times and queue length)\n")
            f.write("- 20% for environmental impact (CO2 emissions)\n\n")

            f.write("This implementation demonstrates how smart traffic systems can contribute to\n")
            f.write("more sustainable cities, directly addressing SDG 11 objectives.\n")

        print(f"Analysis complete! Results saved to {plots_dir}")
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sustainable Traffic Control with Custom OSM Map')
    parser.add_argument('--osm', type=str, default="map.osm", help='Path to OSM file')
    parser.add_argument('--gui', action='store_true', help='Run with GUI')
    parser.add_argument('--episodes', type=int, default=30, help='Number of episodes')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of SUMO files')
    return parser.parse_args()


if __name__ == "__main__":
    print("Sustainable Traffic Control with Custom OSM Map")
    print("==============================================")
    print("This script implements a reinforcement learning approach to traffic")
    print("signal control that addresses SDG 11: Sustainable Cities.\n")

    # Parse command line arguments
    args = parse_args()

    # Update OSM file path if provided
    if args.osm != "map.osm":
        OSM_FILE = args.osm
        print(f"Using OSM file: {OSM_FILE}")

    # Convert OSM to SUMO network if needed or forced
    if not os.path.exists(NET_FILE) or not os.path.exists(ROUTE_FILE) or args.rebuild:
        success = convert_osm_to_sumo()
        if not success:
            print("Could not convert OSM file to SUMO network. Exiting.")
            sys.exit(1)

    # Run simulation
    results_dir = run_simulation(use_gui=args.gui, episodes=args.episodes)

    # Analyze results
    analyze_results(results_dir)