# Enhanced Sustainable Traffic Control

## Overview
This repository provides an enhanced reinforcement learning solution for traffic signal control at a two-way intersection using the [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) framework
it optimizes traffic efficiency, CO₂ emissions, and vehicle throughput to support Sustainable Development Goal 11 (sustainable cities)

## Prerequisites
- **SUMO** installed (version ≥ 1.6.0)
- **Python** 3.7 or higher
- **git** for cloning repositories

## Setup
1. **clone SUMO-RL repository**
   ```bash
   git clone https://github.com/LucasAlegre/sumo-rl.git
   ```
2. **clone this repository**
   ```bash
   git clone https://github.com/Jakub-Nowicki/sumo_project.git
   cd sumo_project
   ```
3. **install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **set SUMO_HOME** environment variable to your SUMO installation path
   - on linux/macOS:
     ```bash
     export SUMO_HOME="/path/to/sumo"
     ```
   - on Windows (PowerShell):
     ```powershell
     setx SUMO_HOME C:\path\to\sumo
     ```

## Configuration
- open `2way_rl_script.py`
- update `NET_FILE` and `ROUTE_FILE` constants if your `sumo-rl/sumo_rl/nets/` folder is in a different location
- ensure the `sumo-rl/sumo_rl/nets/` directory from the cloned SUMO-RL repository is accessible at the specified paths

## Usage
run the main script to start training:
```bash
python 2way_rl_script.py
```
- you will be prompted:
  - whether to use SUMO GUI (y/n)
  - number of training episodes
- a timestamped folder under `results/` will store:
  - `metrics.csv` (SUMO-RL output)
  - `episode_rewards.csv` (per-episode reward log)
  - `learning_curve.png` (reward plot)
  - `plots/` subfolder with additional performance charts

## Project Structure
```
├── 2way_rl_script.py    # main training & analysis script
├── requirements.txt      # Python dependencies
├── README.md             # this file
├── results/              # timestamped output directories
└── sumo-rl/              # cloned SUMO-RL repository with nets definitions
```

