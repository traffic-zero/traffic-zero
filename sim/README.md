# Traffic Simulations

Welcome to the traffic simulation module! This folder contains everything you need to run traffic simulations using SUMO and optionally visualize them in CARLA's 3D environment.

## 📚 Documentation Guide

Read these guides in order to get started:

### 1. [SUMO.md](./SUMO.md) - **Start Here!** ⭐

Your first stop for running traffic simulations. This guide covers:

- **Interactive Mode** - Manual control using SUMO-GUI (perfect for learning)
- **Automated Mode** - Programmatic control for experiments and AI agents
- Available simulation scenarios
- How to choose and run simulations

**Start with this guide** to run your first simulation!

### 2. [CARLA.md](./CARLA.md) - Optional 3D Visualization

Want to see your simulations in beautiful 3D? This guide shows you how to:

- Install and setup CARLA simulator
- Run SUMO simulations in CARLA's 3D environment
- Get realistic visualization of traffic scenarios

**Note**: CARLA is completely optional. Your SUMO simulations work perfectly fine without it!

### 3. [DATA_COLLECTION.md](./DATA_COLLECTION.md) - Data Collection & ML Training

For ML training and data analysis, this guide covers:

- Comprehensive TraCI data collection
- Evaluation metrics computation
- Dynamic traffic light control with action logging
- CSV export and in-memory data access
- How to prepare data for reinforcement learning

**Use this guide** when you need to collect simulation data for ML training!

## 🚀 Quick Start

```python
from sim import run_interactive, run_automated, run_carla

# Interactive mode - manual control (recommended for beginners)
run_interactive("simple4")

# Automated mode - programmatic control for experiments
run_automated("simple4")

# CARLA mode - 3D visualization (requires CARLA setup)
run_carla("simple4", duration=120)
```

## 📂 Folder Structure

```plaintext
sim/
├── README.md           # This file - documentation overview
├── SUMO.md            # SUMO simulation guide (start here!)
├── CARLA.md           # CARLA 3D visualization guide (optional)
├── sumo/              # SUMO-specific code and runners
├── carla/             # CARLA integration code
└── intersections/     # Simulation scenarios
    └── simple4/       # Example: 4-way intersection
        ├── nodes.nod.xml
        ├── edges.edg.xml
        ├── routes.rou.xml
        ├── network.net.xml
        └── simple4.sumocfg (auto-generated)
```

## 🎯 What's Available

### Simulation Modes

| Mode | Function | Purpose | GUI |
|------|----------|---------|-----|
| **Interactive** | `run_interactive()` | Manual control, learning | ✅ SUMO-GUI |
| **Automated** | `run_automated()` | Experiments, AI control | ✅ SUMO-GUI |
| **CARLA** | `run_carla()` | 3D visualization | ✅ CARLA 3D |

### Available Scenarios

1. **simple4** - A 4-way intersection with traffic lights
   - Custom traffic light programs
   - TraCI-based control
   - Basic vehicle flow

More scenarios coming soon! 🚧

## 🔧 Command Line Usage

```bash
# Interactive SUMO simulation
python -c "from sim import run_interactive; run_interactive('simple4')"

# Automated SUMO simulation
python -c "from sim import run_automated; run_automated('simple4')"

# CARLA 3D visualization (requires CARLA running)
python -m sim.carla simple4

# Generate SUMO config manually (usually auto-generated)
python -m sim.sumo.generate_config simple4
```

## 💡 Tips

- **New to SUMO?** Start with `run_interactive()` - it gives you full control and helps you learn
- **Building an AI agent?** Use `run_automated()` - it provides TraCI access for programmatic control
- **Want eye candy?** Set up CARLA using [CARLA.md](./CARLA.md) for beautiful 3D visualization
- **Configuration files** are auto-generated - you don't need to create `.sumocfg` files manually

## 🆘 Need Help?

1. **First simulation**: Read [SUMO.md](./SUMO.md)
2. **3D visualization**: Read [CARLA.md](./CARLA.md)
3. **Data collection for ML**: Read [DATA_COLLECTION.md](./DATA_COLLECTION.md)
4. **Project setup**: See main [README.md](../README.md)
5. **Code details**:
   - SUMO runner: `sim/sumo/runner.py`
   - CARLA bridge: `sim/carla/bridge.py`
   - Data collection: `sim/sumo/data_collector.py`

## 📖 External Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [CARLA Documentation](https://carla.readthedocs.io/)
- [TraCI Documentation](https://sumo.dlr.de/docs/TraCI.html) (for programmatic control)

---

**Ready to start?** Head over to [SUMO.md](./SUMO.md) and run your first simulation! 🚗💨

**Need to collect data for ML?** Check out [DATA_COLLECTION.md](./DATA_COLLECTION.md) for comprehensive data collection and metrics! 📊
