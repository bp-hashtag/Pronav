"""Global configuration for ProNav Simulation"""
import numpy as np

# === Simulation ===
DT = 0.1
SIM_TIME = 90
N = 4                                           # ProNav constant (3-5)
ALPHA = 0.3                                     # Noise reduction factor (0.2 - 0.5)
MAX_ACC = 100                                   # Max allowed missile acceleration (m/s²)

# === Intercept & Radar ===
INTERCEPT_DISTANCE = 12.0                       # meters for hit confirmation
INTERCEPT_CONFIRMATION_FRAMES = 3               # consecutive frames below threshold
GROUND_IMPACT_ALTITUDE = 30                     # consider crashed if below this
APPROACH_ANGLE_THRESHOLD = np.radians(120)      # threat assessment angle
DEFENDED_RADIUS = 2000.0
RADAR_MAX_RANGE = 3000.0
MAX_MISSILES_PER_TARGET = 2
MISSILE_SELECTION_MODE = 2                      # 1 - Top threats take all, 2 - ring assignment
RADAR_FUSION = 3                                # 1 - linear, 2 - weighted, 3 - KF filter
KFP_NOISE = 0.2                                 # Kalman Filter process noise
KFM_NOISE = 3                                   # Kalman Filter measurement noise

# === Multi-mode guidance ===
MAX_RECOVERY_ATTEMPTS = 3                       # Prevent infinite oscillation
OVERSHOOT_DURATION = 1.0                        # seconds of hard turn before recovery
VC_CLOSING_THRESHOLD = 5.0                      # minimum Vc to be considered closing
VC_OPENING_THRESHOLD = -5.0                     # maximum Vc to be considered opening
NEAR_THRESHOLD = 500.0                          # distance to consider "near target" for miss detection

# === Output ===
SAVE_VIDEO = False
OUTPUT_FILE = 'pronav_multi_target.mp4'
VIDEO_FPS = 30

# === LIVE DATA MODE ===
LIVE_MODE = False                               # Set True to use external data
LIVE_FEED_TYPE = 'network'                      # Options: 'network', 'file', 'udp'
LIVE_DATA_PORT = 5000                           # WebSocket/TCP port
LIVE_UDP_PORT = 6000                            # UDP port
LIVE_LOG_FILE = None                            # Path to CSV/log file for replay
