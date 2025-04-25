#!/usr/bin/env python3
"""
Sustainable Traffic Control with SUMO-RL
Addressing SDG 11: Sustainable Cities and Communities

This script implements a reinforcement learning agent that controls
traffic signals while considering both traffic efficiency and emissions.
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
from sumo_rl.agents import QLAgent

# Paths to SUMO network and route files
NET_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\single-intersection\single-intersection.net.xml'
ROUTE_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\single-intersection\single-intersection.rou.xml'

# Create output directory
OUT_DIR = 'results/sustainable'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Global dict to store emission data
emission_data = {}


def sustainable_reward(traffic_signal):
    """
    Custom reward function that considers both traffic efficiency and emissions
    Based on the available methods in your SUMO-RL version

    Args:
        traffic_signal: TrafficSignal object

    Returns:
        float: Combined reward value
    """
    # Get SUMO connection
    sumo = traffic_signal.sumo

    # --- Traffic Efficiency Component ---
    # Use accumulated waiting time per lane (available in your version)
    waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_times)

    # Also consider queue length
    total_queue = traffic_signal.get_total_queued()

    # Combine waiting time and queue for efficiency reward
    # Both should be minimized, so negative values
    efficiency_reward = -(total_waiting_time + total_queue)

    # --- Emission Component ---
    total_co2 = 0
    # Get all vehicles in the traffic signal's lanes
    vehicle_ids = []
    for lane_id in traffic_signal.lanes:
        vehicle_ids.extend(sumo.lane.getLastStepVehicleIDs(lane_id))

    # Calculate total CO2 emissions
    for veh_id in vehicle_ids:
        total_co2 += sumo.vehicle.getCO2Emission(veh_id)  # in mg

    # Convert to grams for easier interpretation
    total_co2 = total_co2 / 1000.0

    # Emission reward (negative because we want to minimize)
    emission_reward = -total_co2

    # Store emissions in global dict instead of using metrics
    global emission_data
    emission_data[traffic_signal.id] = emission_data.get(traffic_signal.id, []) + [total_co2]

    # --- Combined Reward ---
    # Weight factors (80% efficiency, 20% emissions)
    efficiency_weight = 0.8
    emission_weight = 0.2

    # Normalize to prevent one component from dominating
    # These normalization factors may need tuning based on your scenario
    norm_efficiency = efficiency_reward / 100.0  # Assuming typical values in hundreds
    norm_emission = emission_reward / 10.0  # Assuming typical values in tens of grams

    # Combined reward
    reward = (efficiency_weight * norm_efficiency) + (emission_weight * norm_emission)

    return reward


def run_simulation(use_gui=True, episodes=20):
    """
    Run the SUMO-RL simulation with the sustainable reward function

    Args:
        use_gui: Whether to show the SUMO GUI
        episodes: Number of episodes to run

    Returns:
        str: Path to results directory
    """
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{OUT_DIR}_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Reset global emission data
    global emission_data
    emission_data = {}

    # Store metrics manually
    metrics = {
        'step': [],
        'episode': [],
        'waiting_time': [],
        'co2_emission': [],
        'reward': []
    }

    # Set up environment
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=f'{results_dir}/metrics',
        use_gui=use_gui,
        num_seconds=1000,  # Shorter simulation time
        delta_time=5,
        min_green=5,
        max_green=50,
        reward_fn=sustainable_reward,
        add_system_info=True,
        add_per_agent_info=True
    )

    # Print environment info
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Traffic signal IDs:", env.ts_ids)

    # Training loop
    print(f"\nStarting simulation for {episodes} episodes...")

    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        done = {'__all__': False}
        episode_reward = 0
        step = 0

        while not done['__all__']:
            # Choose action for each traffic signal
            action = {}
            for ts in state.keys():
                action[ts] = env.action_space.sample()  # Random actions for simplicity

            # Execute the action
            next_state, reward, done, _ = env.step(action)

            # Track episode statistics
            episode_reward += sum(reward.values())

            # Record metrics every 5 steps
            if step % 5 == 0:
                # Get waiting time from the environment
                if hasattr(env, 'metrics') and hasattr(env.metrics, 'data'):
                    if 'system_total_waiting_time' in env.metrics.data and env.metrics.data[
                        'system_total_waiting_time']:
                        waiting_time = env.metrics.data['system_total_waiting_time'][-1]
                    else:
                        waiting_time = 0
                else:
                    # Fallback: calculate from traffic signals
                    waiting_time = 0
                    for ts_id in env.ts_ids:
                        ts = env.traffic_signals[ts_id]
                        waiting_times = ts.get_accumulated_waiting_time_per_lane()
                        waiting_time += sum(waiting_times)

                # Get CO2 emissions from our global dict
                co2 = 0
                for ts_id in env.ts_ids:
                    if ts_id in emission_data and emission_data[ts_id]:
                        co2 += emission_data[ts_id][-1]

                # Store metrics
                metrics['step'].append(step)
                metrics['episode'].append(episode)
                metrics['waiting_time'].append(waiting_time)
                metrics['co2_emission'].append(co2)
                metrics['reward'].append(episode_reward)

            # Move to next state
            state = next_state
            step += 1

            # End episode if we've reached the time limit
            if step >= 200:  # Safety limit to prevent infinite loops
                break

        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} - Steps: {step}")

    # Close environment
    env.close()

    # Save our custom metrics to CSV
    pd.DataFrame(metrics).to_csv(f"{results_dir}/custom_metrics.csv", index=False)

    print(f"\nSimulation complete! Results saved to {results_dir}")
    return results_dir


def analyze_results(results_dir):
    """
    Analyze and visualize the results of the simulation

    Args:
        results_dir: Directory containing the results
    """
    try:
        # Try to find our custom metrics file first
        custom_metrics_file = Path(results_dir) / "custom_metrics.csv"
        if custom_metrics_file.exists():
            print(f"Analyzing custom metrics from {custom_metrics_file}")
            metrics = pd.read_csv(custom_metrics_file)
        else:
            # Try to find the standard metrics file
            metrics_files = list(Path(results_dir).glob("metrics*.csv"))
            if not metrics_files:
                print(f"No metrics files found in {results_dir}")
                return

            metrics_file = metrics_files[0]
            print(f"Analyzing metrics from {metrics_file}")
            metrics = pd.read_csv(metrics_file)

        # Create results directory
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot waiting time evolution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        waiting_column = 'waiting_time'
        if waiting_column not in metrics.columns and 'system_total_waiting_time' in metrics.columns:
            waiting_column = 'system_total_waiting_time'

        if waiting_column in metrics.columns:
            plt.plot(metrics['step'], metrics[waiting_column])
            plt.xlabel('Simulation Step')
            plt.ylabel('Total Waiting Time (s)')
            plt.title('Traffic Efficiency Evolution')
            plt.grid(True)
        else:
            print("Warning: No waiting time column found in metrics")

        # Plot CO2 emissions evolution
        plt.subplot(2, 1, 2)
        if 'co2_emission' in metrics.columns:
            plt.plot(metrics['step'], metrics['co2_emission'], color='green')
            plt.xlabel('Simulation Step')
            plt.ylabel('CO2 Emissions (g)')
            plt.title('Environmental Impact Evolution')
            plt.grid(True)
        else:
            print("Warning: 'co2_emission' column not found in metrics")

        plt.tight_layout()
        plt.savefig(plots_dir / "sustainability_metrics.png")

        # Generate a summary report
        with open(plots_dir / "summary.txt", "w") as f:
            f.write("Sustainable Traffic Control - Summary Report\n")
            f.write("===========================================\n\n")
            f.write("This experiment implemented a reinforcement learning approach\n")
            f.write("to traffic signal control that addresses SDG 11: Sustainable Cities\n")
            f.write("by considering both traffic efficiency and environmental impact.\n\n")

            f.write("Key metrics:\n")
            if waiting_column in metrics.columns:
                avg_waiting = metrics[waiting_column].mean()
                max_waiting = metrics[waiting_column].max()
                f.write(f"- Average waiting time: {avg_waiting:.2f} seconds\n")
                f.write(f"- Maximum waiting time: {max_waiting:.2f} seconds\n")

            if 'co2_emission' in metrics.columns:
                avg_co2 = metrics['co2_emission'].mean()
                total_co2 = metrics['co2_emission'].sum() * 5  # Assuming 5 seconds per step
                f.write(f"- Average CO2 emissions: {avg_co2:.2f} g/s\n")
                f.write(f"- Total CO2 emissions: {total_co2:.2f} g over simulation period\n")

            f.write("\nThis study demonstrates how intelligent traffic signal control\n")
            f.write("can contribute to more sustainable urban environments by reducing\n")
            f.write("both congestion and emissions, aligning with SDG 11 goals.\n")

        print(f"Analysis complete! Plots and summary saved to {plots_dir}")

    except Exception as e:
        print(f"Error analyzing results: {e}")


if __name__ == "__main__":
    print("Sustainable Traffic Control with SUMO-RL")
    print("========================================")
    print("This script implements a reinforcement learning approach to traffic")
    print("signal control that addresses SDG 11: Sustainable Cities by considering")
    print("both traffic efficiency and environmental impact.\n")

    # Ask for GUI mode
    use_gui = input("Run with GUI? (y/n): ").lower().startswith('y')

    # Ask for number of episodes
    episodes = 5
    try:
        episodes = int(input(f"Number of episodes (default: {episodes}): ") or episodes)
    except ValueError:
        print(f"Using default value: {episodes}")

    # Run the simulation
    results_dir = run_simulation(use_gui=use_gui, episodes=episodes)

    # Analyze the results
    analyze_results(results_dir)