
# 🚦 Traffic Zero

Adaptive multi-agent traffic lights system

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![SUMO](https://img.shields.io/badge/SUMO-simulation-orange)](https://sumo.dlr.de/docs/index.html)

This repository implements a **Multi-Agent System (MAS) for adaptive traffic light control** using SUMO simulations and reinforcement learning. Agents represent intersections and optimize traffic flow using local and neighbor observations.

## Features

- **Traffic Simulation:** Configurable networks (toy grids, arterial corridors, real maps from OpenStreetMap).
- **Controller Types:**
  - Fixed-time and actuated baseline controllers.
  - RL-based adaptive controllers (decentralized, centralized, or hybrid).
- **Multi-Agent Coordination:** Neighbor communication protocols and CTDE setups.
- **Data Integration:** Supports NGSIM, METR-LA, NYC Taxi, and synthetic traffic generators.
- **Metrics & Evaluation:** Average waiting time, travel time, throughput, emissions, and fairness.

## Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:traffic-zero/traffic-zero.git
cd traffic-zero
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.\.venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install uv
uv sync
pre-commit install # sets up pre-commit hooks for linting and formatting
```

### 4. Install SUMO

Follow the instructions on the [SUMO website](https://sumo.dlr.de/docs/Installing/index.html) to install SUMO on your system.

### 5. Run a Simulation

```bash
python run_simulation.py --config configs/toy_grid.yaml
```

### 6. Generate Traffic Data

```bash
python data/generate_routes.py --config configs/traffic_data.yaml
```

### 7. Train an RL Agent

```bash
python agents/train_agent.py --config configs/rl_agent.yaml
```

## Repository Structure

```plaintext
adaptive-traffic/
├── sim/                # SUMO networks, configs, route generators
├── agents/             # Baseline & RL controllers
├── data/               # Raw & processed datasets, OD generation scripts
├── experiments/        # Experiment configs, sweep scripts
├── analysis/           # Jupyter notebooks, plots, metrics
├── configs/            # YAML configuration files
├── models/             # Saved RL models, training logs
├── scripts/            # Utility scripts (visualization, evaluation)
├── dashboard/          # Visualization / dashboards
├── docs/               # Reports, slides, tutorials
├── run_simulation.py   # Simulation script
└── pyproject.toml      # Project metadata and dependencies
```

## Metrics & Evaluation

- Primary Metrics: Average waiting time, travel time, throughput, maximum queue length.

- Secondary Metrics: Emissions, fuel consumption, fairness, robustness under sensor/communication noise.

- Experiments: Evaluate RL vs baseline, V2X penetration sweep, traffic volume sweep, incident simulation.

## Contributing

1. Fork the repo

2. Create a feature branch: `git checkout -b feature/your-feature`

3. Commit changes: `git commit -m "Add your feature"`

4. Push to branch: `git push origin feature/your-feature`

5. Open a Pull Request

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Resources & References

- [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
- [TraCI API](https://sumo.dlr.de/docs/TraCI.html)
- [NGSIM Dataset](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
- [METR-LA Dataset](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset)
- [OpenAI Gym](https://gym.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- [RLlib](https://docs.ray.io/en/latest/rllib.html)
- [OpenStreetMap](https://www.openstreetmap.org/)
- Multi-Agent RL Papers (traffic control):
  - [MA-PPO Traffic Signal Control](https://arxiv.org/abs/2503.02189)
  - [Cooperative MARL for Traffic Lights](https://www.mdpi.com/2071-1050/15/4/3479)
