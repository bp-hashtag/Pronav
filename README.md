# ProNav 3D Simulation

Multi-Target Missile Guidance Simulator with Live Data Feed Support

A comprehensive 3D proportional navigation simulation featuring multi-mode guidance, Kalman filter sensor fusion, and flexible live data integration.

## Features

| Capability | Description |
|------------|-------------|
| Multi-Mode Guidance | World-frame ProNav → Hard Turn → Body-frame recovery |
| Live Data Support | File replay, WebSocket/TCP streaming, or UDP sensor input |
| Sensor Fusion | 4-node radar network with Kalman Filter processing |
| Threat Assessment | Weighted scoring: distance (40%), speed (40%), altitude (25%), direction (20%) |
| Overshoot Recovery | Automatic hard turn + body-frame correction (up to 3 attempts) |
| Video Export | MP4 animation output with telemetry overlay |

## Project Structure

Pronav/
├── main.py                  # Entry point & orchestration
├── config.py                # Global simulation parameters
├── models.py                # TargetConfig & MissileConfig dataclasses
├── target_system.py         # MultiTargetSystem & threat assessment
├── radar.py                 # Radar node simulation with range-dependent noise
├── kalman_filter.py         # 3D constant-acceleration Kalman Filter
├── live_feed.py             # LiveDataFeed (WebSocket/File/UDP)
└── README.md                # This file

## Installation

Clone repository:
git clone https://github.com/bp-hashtag/Pronav.git
cd Pronav

Install dependencies:
pip install numpy matplotlib

### Optional: Video Export
For MP4 video generation:
sudo apt install ffmpeg      # Ubuntu/Debian
brew install ffmpeg          # macOS

## Usage

### Standard Simulation Mode

python main.py

Runs with pre-configured targets from main.py. Edit target_configs in main.py to change scenario.

### Live Data Mode

Enable external data source via config.py:

LIVE_MODE = True           # Enable live data input
LIVE_FEED_TYPE = 'file'    # Options: 'file', 'network', 'udp'
LIVE_LOG_FILE = 'data.csv' # For file mode
LIVE_DATA_PORT = 5000      # For network mode
LIVE_UDP_PORT = 6000       # For UDP mode

#### Option A: File Replay (CSV/JSON)

LIVE_FEED_TYPE = 'file'
LIVE_LOG_FILE = 'target_trajectories_custom_5.csv'

CSV format columns: t, 0_x, 0_y, 0_z, 0_vx, 0_vy, 0_vz, 1_x, ...

#### Option B: WebSocket Network Stream

LIVE_FEED_TYPE = 'network'
LIVE_DATA_PORT = 5000

Expects JSON: {tid: int, x, y, z, vx, vy, vz, timestamp}

#### Option C: UDP Sensor

LIVE_FEED_TYPE = 'udp'
LIVE_UDP_PORT = 6000

Receives JSON packets over UDP socket.

## Configuration Parameters

Key settings in config.py:

| Parameter | Default | Description |
|-----------|---------|-------------|
| DT | 0.1s | Simulation timestep |
| SIM_TIME | 45s | Total simulation duration |
| N | 4 | ProNav navigation constant |
| INTERCEPT_DISTANCE | 10m | Hit confirmation threshold |
| MAX_MISSILES_PER_TARGET | 2 | Max missiles per threat |
| RADAR_MAX_RANGE | 3000m | Maximum radar detection range |
| SAVE_VIDEO | False | Enable MP4 export |
| VIDEO_FPS | 30 | Output video frame rate |

## Algorithms

### 1. Proportional Navigation (World-Frame)
a = N × Vc × λ̇
Where N=4, Vc = closing velocity, λ̇ = line-of-sight rate

### 2. Constant-Acceleration Kalman Filter
State vector: [x, y, z, vx, vy, vz, ax, ay, az]

Predict:
x_new = x + vx·dt + 0.5·ax·dt²
vx_new = vx + ax·dt
ax_new = ax

### 3. Overshoot Detection & Recovery
Monitors closing velocity sign change → triggers:
1. Hard Turn (1.5s): Maximum lateral acceleration
2. Body-Frame ProNav: LOS-rate-only guidance
3. Repeat up to 3 times before deactivation

### 4. Range-Dependent Radar Noise
effective_noise = base_noise × (1 + distance / max_range)

## Example Output

============================================================
MISSILE TO TARGET ASSIGNMENTS (Initial)
============================================================
  Missile 1  -> T1         (TID: 0)
  Missile 2  -> T2         (TID: 1)
  ...
============================================================

[Missile 1] 🔥 LAUNCH at t=2s (Target TID: 0)
[Missile 1] CLOSING t=2.0s | d=2400m | Vc=+89m/s
...
[Missile 1] ✓ HIT T1 at t=15.3s within 8.4m
[Missile 1] ↻ OVERSHOOT t=12.1s (Vc: +35→-12, d=650m)
[Missile 1] 🔄 BODY-PRONAV t=13.6s | Recovered: 1×

## Key Files Explained

| File | Purpose |
|------|---------|
| main.py | Orchestrates simulation loop, target configs, live feed initialization |
| config.py | All global constants (timing, thresholds, ranges) |
| models.py | Dataclass definitions for Target and Missile configurations |
| target_system.py | Manages multiple targets, threat scoring, ground impact detection |
| radar.py | Simulates radar sweep with range-dependent noise modeling |
| kalman_filter.py | 3D constant-acceleration state estimation |
| live_feed.py | Abstract base class + WebSocket/File/UDP implementations |

## License

MIT License

## Author

bp-hashtag

---

Created: August 2026
