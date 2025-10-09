# CARLA 3D Visualization

Want to see your SUMO traffic simulations in beautiful 3D? Let's get CARLA running!

## What You'll Get

- 🎮 **3D visualization** of your traffic scenarios
- 🗺️ **Custom maps** generated from your SUMO networks - no complex cities, just YOUR roads!
- 🚗 **Real-time vehicle tracking** - watch your SUMO vehicles move in 3D
- 🚦 **Traffic lights** that match your SUMO configuration
- 🎥 **Free camera** to explore and watch from any angle

## Quick Setup (First Time Only)

### Step 1: Download CARLA

Grab CARLA 0.9.15 from [GitHub releases](https://github.com/carla-simulator/carla/releases/tag/0.9.15)

- **Windows**: Download `CARLA_0.9.15.zip` (~3.5GB)
- **Linux**: Download `CARLA_0.9.15.tar.gz` (~3.5GB)

Extract it somewhere nice, like `C:\CARLA\CARLA_0.9.15`

### Step 2: Set Environment Variable

Tell your system where CARLA lives:

**Windows** (PowerShell as Admin):
```powershell
[System.Environment]::SetEnvironmentVariable('CARLA_ROOT', 'C:\CARLA\CARLA_0.9.15', 'User')
[System.Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\CARLA\CARLA_0.9.15\PythonAPI\carla', 'User')
```

**Linux/macOS**:
```bash
echo 'export CARLA_ROOT="/opt/carla/CARLA_0.9.15"' >> ~/.bashrc
echo 'export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"' >> ~/.bashrc
source ~/.bashrc
```

⚠️ **Important**: Restart your terminal after this!

### Step 3: Generate SUMO Config

Your scenarios need a `.sumocfg` file for CARLA:

```bash
python -m sim.sumo.generate_config simple4
```

You should see:
```
✓ Configuration file created: sim/intersections/simple4/simple4.sumocfg
✓ Ready for CARLA co-simulation!
```

## Running Your First Simulation

### Start CARLA Server

Open a terminal and start CARLA:

```bash
cd C:\CARLA\CARLA_0.9.15
.\CarlaUE4.exe -windowed -ResX=1280 -ResY=720
```

Wait for the CARLA window to appear. You'll see a 3D city.

### Run Your Simulation

In another terminal:

```bash
python -m sim.carla simple4 --duration 120
```

## What You'll See

The terminal will show something like this:

```
Starting co-simulation for 'simple4'...
Duration: 120s
Use SUMO network: True
------------------------------------------------------------

============================================================
Starting CARLA-SUMO Co-Simulation
============================================================
Connecting to CARLA server at localhost:2000...
Loading SUMO network as OpenDRIVE: ...simple4.xodr...
✓ SUMO network loaded as CARLA map!
  Generated procedural 3D mesh from OpenDRIVE
✓ Connected to CARLA (map: Carla/Maps/OpenDriveMap)
Starting SUMO with ...simple4.sumocfg...
✓ SUMO started successfully
✓ Camera positioned to view simulation area
  🎮 Camera Controls:
     • Mouse: Look around
     • W/A/S/D: Move forward/left/back/right
     • Q/E: Move down/up
     • Scroll UP: Increase movement speed ⚡
     • Scroll DOWN: Decrease movement speed

  💡 Tip: Scroll up several times for FAST camera movement!

Simulation parameters:
  Step length: 0.05s
  Duration: 120
  TLS manager: sumo

▶ Simulation running... (Press Ctrl+C to stop)

✓ Spawned vehicle veh0 at CARLA (98.4, -194.9)
✓ Spawned vehicle EW_0 at CARLA (194.9, -101.6)
✓ Spawned vehicle SN_0 at CARLA (101.6, -5.1)
...

Step  1200 | Elapsed:   60.2s | Vehicles:   5
✓ SUMO simulation finished (no more vehicles)

Cleaning up...
✓ CARLA settings restored
✓ SUMO closed
Cleanup complete.
```

**In the CARLA window**, you'll see:
- Your intersection rendered as a plain 3D map (no city buildings!)
- Colorful vehicles moving around
- Traffic lights at your intersection
- Clean environment - just roads and ground

## Using It

### Python API

```python
from sim import run_in_carla

# Run for 2 minutes
run_in_carla('simple4', duration=120)
```

### Command Line

```bash
# Run simulation
python -m sim.carla simple4 --duration 120

# List available scenarios
python -m sim.carla --list

# Use a CARLA town map instead of custom network
python -m sim.carla simple4 --no-sumo-network
```

## Camera Controls (in CARLA window)

- **Mouse** - Look around
- **W/A/S/D** - Move camera
- **Q/E** - Move up/down
- **Scroll UP** - Faster movement (do this a few times!)
- **Scroll DOWN** - Slower movement

💡 **Tip**: Scroll up multiple times when simulation starts for much faster camera movement!

## Cool Features

### Your SUMO Network Becomes the Map

The simulation automatically:
1. Converts your SUMO network to OpenDRIVE format
2. Loads it as a custom CARLA map
3. Generates a clean 3D environment with just YOUR roads

No city buildings, no distractions - just your traffic scenario in 3D!

### Real-Time Synchronization

- SUMO handles all the traffic logic
- CARLA shows everything in beautiful 3D
- They stay perfectly in sync
- Your RL agent can control traffic via SUMO while you watch in CARLA

## Troubleshooting

### "CARLA_ROOT environment variable not set"

Set it and **restart your terminal**:

```powershell
# Windows - check if it's set
echo $env:CARLA_ROOT

# Linux/macOS - check if it's set
echo $CARLA_ROOT
```

### "Cannot connect to CARLA server"

Make sure CARLA is running first! Look for the CARLA window.

### "CARLA Python API not found"

Add CARLA to your Python path:

```powershell
# Windows
$env:PYTHONPATH = "C:\CARLA\CARLA_0.9.15\PythonAPI\carla;$env:PYTHONPATH"

# Linux/macOS
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"
```

### Simulation Running But Can't See Vehicles

The camera might be looking at the wrong place. When simulation starts:
1. Look around with your mouse
2. Move camera with WASD
3. Scroll UP several times for faster movement
4. Navigate to where you see vehicles spawning (check the coordinates in terminal)

### Performance Issues

CARLA can be demanding. Try:

```bash
# Lower graphics quality
./CarlaUE4.exe -quality-level=Low

# Smaller window
./CarlaUE4.exe -windowed -ResX=800 -ResY=600

# No graphics (for training)
./CarlaUE4.exe -RenderOffScreen
```

## Common Commands

```bash
# Generate config for a scenario
python -m sim.sumo.generate_config <scenario_name>

# Run in CARLA with custom map
python -m sim.carla <scenario_name> --duration <seconds>

# Start CARLA with better performance
cd $CARLA_ROOT && ./CarlaUE4.exe -quality-level=Low
```

## FAQ

**Q: Do I need CARLA to run SUMO simulations?**  
A: Nope! SUMO works fine on its own. CARLA is just for fancy 3D visualization.

**Q: Can I use this with my RL agent?**  
A: Absolutely! Your agent controls traffic via SUMO's TraCI, and CARLA just shows you what's happening in 3D.

**Q: Do I need to restart CARLA between simulations?**  
A: No, keep it running and run multiple simulations!

**Q: Why is the map plain/empty?**  
A: That's intentional! We convert your SUMO network to a custom map with only YOUR roads. No city buildings to block your view.

**Q: Can I record the simulation?**  
A: Yes! CARLA has built-in recording. Check the [CARLA docs](https://carla.readthedocs.io/) for details.

## Need Help?

- Check `sim/carla/bridge.py` for implementation details
- Check `sim/carla/runner.py` for the main interface
- Visit [CARLA docs](https://carla.readthedocs.io/) for CARLA-specific questions
- See [CARLA-SUMO tutorial](https://carla.readthedocs.io/en/latest/adv_sumo/) for advanced topics

---

Happy simulating! 🚗🚦🎮
