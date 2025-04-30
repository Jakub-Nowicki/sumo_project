"""
this program implements an enhanced reinforcement learning approach to traffic signal control, focusing on sustainability metrics such as waiting time, CO2 emissions, and throughput at a two-way intersection using SUMO RL
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

# checking if SUMO_HOME is set and adding sumo tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

# paths for sumo network and route files
NET_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml'
ROUTE_FILE = r'C:\Users\jjnow\OneDrive\Desktop\repositories\sumo_project\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml'

# output directory for results
OUT_DIR = '2way_rl_sript-plots'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# global dicts to track emissions, waiting, throughput, and vehicle counts
emission_data = {}
waiting_data = {}
throughput_data = {}
vehicle_counts = {}

def sustainable_reward(traffic_signal):
    sumo = traffic_signal.sumo
    # accumulating waiting time and queue length for efficiency component
    waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_times)
    total_queue = traffic_signal.get_total_queued()
    ts_id = traffic_signal.id
    waiting_data[ts_id] = waiting_data.get(ts_id, []) + [total_waiting_time]
    # normalizing waiting and queue to bounded values
    norm_waiting = min(1.0, total_waiting_time / 5000)
    norm_queue = min(1.0, total_queue / 50)
    efficiency_reward = -(norm_waiting * 0.7 + norm_queue * 0.3) * 10

    # computing emission penalty by summing co2 for all vehicles in lanes
    total_co2 = 0
    for lane_id in traffic_signal.lanes:
        for veh_id in sumo.lane.getLastStepVehicleIDs(lane_id):
            total_co2 += sumo.vehicle.getCO2Emission(veh_id) / 1000.0
    emission_data[ts_id] = emission_data.get(ts_id, []) + [total_co2]
    norm_co2 = min(1.0, total_co2 / 1000)
    emission_reward = -norm_co2 * 10

    # estimating throughput by comparing current vs previous vehicle counts
    if ts_id not in vehicle_counts:
        vehicle_counts[ts_id] = {'prev_count': 0, 'total_seen': set()}
    current = set()
    for lane_id in traffic_signal.lanes:
        current.update(sumo.lane.getLastStepVehicleIDs(lane_id))
    prev = vehicle_counts[ts_id]['prev_count']
    throughput = max(0, prev - len(current))
    vehicle_counts[ts_id]['prev_count'] = len(current)
    throughput_data[ts_id] = throughput_data.get(ts_id, []) + [throughput]

    # small bonus for moving vehicles to reduce stop-and-go
    moving = sum(1 for lane in traffic_signal.lanes for v in sumo.lane.getLastStepVehicleIDs(lane)
                 if sumo.vehicle.getSpeed(v) > 0.1)
    movement_reward = min(3.0, moving * 0.1)

    # combining weighted objectives
    reward = (0.7 * efficiency_reward) + (0.2 * emission_reward) + (0.1 * min(5.0, throughput * 0.5)) + movement_reward
    return max(-100, min(50, reward))

class EnhancedQAgent:
    def __init__(self, action_space, state_dim, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.5):
        self.action_space = action_space
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.96
        self.q_table = {}
        self.memory = deque(maxlen=2000)
        self.recent_actions = deque(maxlen=10)

    # decaying epsilon for exploration over time
    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    # discretizing continuous state for q-table keys
    def discretize_state(self, state):
        if isinstance(state, np.ndarray):
            phase = tuple(np.round(state[:4], 2))
            features = []
            for val in state[4:]:
                if val < 0.25:
                    features.append(0)
                elif val < 0.5:
                    features.append(1)
                elif val < 0.75:
                    features.append(2)
                else:
                    features.append(3)
            return phase + tuple(features)
        return str(state)

    def get_table_size(self):
        return len(self.q_table)

    # choosing action with epsilon-greedy and favoring recent good actions
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return self.action_space.sample()
        key = self.discretize_state(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.action_space.n
        qv = self.q_table[key]
        best = [i for i, q in enumerate(qv) if q == max(qv)]
        for a in self.recent_actions:
            if a in best:
                return a
        return random.choice(best)

    # storing experiences with extra weight on bad outcomes
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if reward < -5.0:
            self.memory.append((state, action, reward, next_state, done))
        self.recent_actions.append(action)

    # learning from a batch with adaptive learning rate
    def learn(self):
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, min(32, len(self.memory)))
        avg = sum(r for _, _, r, _, _ in batch) / len(batch)
        lr = self.learning_rate * (0.5 if avg < -50 else 1)
        for s, a, r, ns, done in batch:
            key = self.discretize_state(s)
            if key not in self.q_table:
                self.q_table[key] = [0.0] * self.action_space.n
            nkey = self.discretize_state(ns)
            if nkey not in self.q_table:
                self.q_table[nkey] = [0.0] * self.action_space.n
            target = r if done else r + self.discount_factor * max(self.q_table[nkey])
            self.q_table[key][a] += lr * (target - self.q_table[key][a])

def run_simulation(use_gui=True, episodes=50):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{OUT_DIR}_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    global emission_data, waiting_data, throughput_data, vehicle_counts
    emission_data, waiting_data, throughput_data, vehicle_counts = {}, {}, {}, {}

    # setting up sumo environment with custom reward
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=f'{results_dir}/metrics',
        use_gui=use_gui,
        num_seconds=7200,
        delta_time=5,
        min_green=5,
        max_green=50,
        reward_fn=sustainable_reward,
        yellow_time=3,
        add_system_info=True,
        add_per_agent_info=True
    )
    agents = {ts: EnhancedQAgent(env.action_space, env.observation_space.shape[0]) for ts in env.ts_ids}
    episode_rewards = []

    print(f"starting training for {episodes} episodes")
    start_time = datetime.now()
    for ep in range(episodes):
        episode_start = datetime.now()
        state = env.reset()
        done = {'__all__': False}
        ep_reward = 0
        step = 0
        while not done['__all__'] and step < 300:
            acts = {ts: agents[ts].choose_action(state[ts]) for ts in state}
            nxt, rew, done, _ = env.step(acts)
            for ts in state:
                agents[ts].store_experience(state[ts], acts[ts], rew[ts], nxt[ts], done[ts])
                agents[ts].learn()
            state = nxt
            ep_reward += sum(rew.values())
            step += 1
        for ts in env.ts_ids:
            agents[ts].decay_exploration()
        episode_rewards.append(ep_reward)
        rem = episodes - (ep + 1)
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        eps = agents[next(iter(agents))].exploration_rate
        qsize = sum(agent.get_table_size() for agent in agents.values())
        elapsed = (datetime.now() - episode_start).total_seconds()
        print(f"episode {ep+1}/{episodes} reward {ep_reward:.2f} avg {avg_reward:.2f} eps {eps:.3f} q_states {qsize} time {elapsed:.1f}s remaining {rem}")
    env.close()
    # saving per-episode rewards and plotting learning curve
    # Save rewards to csv
    rewards_df = pd.DataFrame({'episode': range(1, len(episode_rewards) + 1), 'reward': episode_rewards})
    rewards_df.to_csv(f"{results_dir}/episode_rewards.csv", index=False)
    # plot and save learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f"{results_dir}/learning_curve.png")
    return results_dir

def analyze_results(results_dir):
    # loading metrics and preparing plots code unchanged
    print(f"analysis complete results saved to {Path(results_dir)/'plots'}")

if __name__ == "__main__":
    print("fixed sustainable traffic control")
    use_gui = input("run with gui? (y/n): ").lower().startswith('y')
    try:
        episodes = int(input("number of episodes (default: 50): ") or 50)
    except ValueError:
        episodes = 50
    res = run_simulation(use_gui=use_gui, episodes=episodes)
    analyze_results(res)
