"""Configuration dataclasses"""
from dataclasses import dataclass

@dataclass
class TargetConfig:
    name: str
    x0: float
    y0: float
    z0: float
    speed: float
    heading_deg: float = None
    pitch_deg: float = None
    target_x: float = None
    target_y: float = None
    target_z: float = None
    priority_altitude_weight: float = 0.4
    priority_speed_weight: float = 0.3
    priority_distance_weight: float = 0.2

@dataclass
class MissileConfig:
    name: str
    x0: float
    y0: float
    cruise_speed: float
    z0: float = 0.0
    N: float = 4.0
    filter_alpha: float = 0.4
    color: str = 'red'
    launch_delay: float = 0.0
