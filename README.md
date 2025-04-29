# Enhanced Sustainable Traffic Control

## Overview
This project implements an enhanced reinforcement learning approach to traffic signal control using the SUMO-RL framework. It balances traffic efficiency, environmental impact (CO₂ emissions), and throughput at a two-way intersection to address Sustainable Development Goal 11.

## Prerequisites
- Clone the SUMO-RL repository:
  ```bash
git clone https://github.com/LucasAlegre/sumo-rl.git
  ```
- install SUMO and set the `SUMO_HOME` environment variable to your SUMO installation path
- Python 3.7 or higher

## Installation
1. clone this repository:
   ```bash
git clone <this-repo-url>
cd <this-repo-folder>
   ```
2. install Python dependencies:
   ```bash
pip install -r requirements.txt
   ```

## Usage
run the main script to start training:
```bash
python sustainable_traffic_control.py
```
- you'll be prompted to choose GUI mode and number of episodes
- results (metrics CSV, episode_rewards.csv, plots) will be saved under `results/YYYYMMDD_HHMMSS`

## Project Structure
```
├── sustainable_traffic_control.py    # main training and analysis script
├── requirements.txt                  # Python dependencies
└── nets/                             # SUMO network and route definitions (from sumo-rl)
```

## Notes
- ensure that the `nets/` folder from the cloned `sumo-rl` repository is accessible at the paths defined in the script
- adjust `NET_FILE` and `ROUTE_FILE` constants if your directory structure differs

## License
MIT

