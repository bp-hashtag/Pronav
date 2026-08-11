"""Multi-target management and threat assessment with live data support"""
import numpy as np
from config import APPROACH_ANGLE_THRESHOLD, DEFENDED_RADIUS, RADAR_MAX_RANGE

class MultiTargetSystem:
    def __init__(self, targets_config, dt=0.1):
        self.targets = []
        self.dt = dt
        self.configs = targets_config
        self.next_tid = 0
        
        for cfg in targets_config:
            target = self._create_target_instance(cfg)
            self.targets.append(target)
        self.reset()

    def _create_target_instance(self, config):
        return {
            'tid': self.next_tid,
            'config': config,
            'x': [], 'y': [], 'z': [],
            'vx': [], 'vy': [], 'vz': [],
            'ax': [], 'ay': [], 'az': [],
            'active': True,
            'hit': False,
            'hit_by': None,
            'hit_time': None,
            'ground_impact': False,
            'ground_impact_time': None,
            'target_kf': None,
            'target_kf_initialized': False,
        }

    def reset(self):
        for target in self.targets:
            cfg = target['config']
            target['x'][:] = []
            target['y'][:] = []
            target['z'][:] = []
            target['vx'][:] = []
            target['vy'][:] = []
            target['vz'][:] = []
            target['ax'][:] = []
            target['ay'][:] = []
            target['az'][:] = []
            target['active'] = True
            target['hit'] = False
            target['hit_by'] = None
            target['hit_time'] = None
            target['ground_impact'] = False
            target['ground_impact_time'] = None

            target['x'].append(cfg.x0)
            target['y'].append(cfg.y0)
            target['z'].append(cfg.z0)

            # Calculate initial velocity
            if cfg.target_x is not None:
                dx = cfg.target_x - cfg.x0
                dy = cfg.target_y - cfg.y0
                dz = cfg.target_z - cfg.z0
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist > 0.1:
                    target['vx'].append((dx / dist) * cfg.speed)
                    target['vy'].append((dy / dist) * cfg.speed)
                    target['vz'].append((dz / dist) * cfg.speed)
                else:
                    target['vx'].append(0)
                    target['vy'].append(0)
                    target['vz'].append(0)

            elif cfg.heading_deg is not None or cfg.pitch_deg is not None:
                heading = np.radians(cfg.heading_deg if cfg.heading_deg else 0)
                pitch = np.radians(cfg.pitch_deg if cfg.pitch_deg else 0)
                target['vx'].append(cfg.speed * np.cos(pitch) * np.cos(heading))
                target['vy'].append(cfg.speed * np.cos(pitch) * np.sin(heading))
                target['vz'].append(cfg.speed * np.sin(pitch))

            else:
                dx = -cfg.x0
                dy = -cfg.y0
                dz = -cfg.z0
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist > 0.1:
                    target['vx'].append((dx / dist) * cfg.speed)
                    target['vy'].append((dy / dist) * cfg.speed)
                    target['vz'].append((dz / dist) * cfg.speed)
                else:
                    target['vx'].append(0)
                    target['vy'].append(0)
                    target['vz'].append(0)

            target['ax'].append(0)
            target['ay'].append(0)
            target['az'].append(0)

    def step_all(self, t):
        """Standard motion simulation (disabled in live mode)"""
        for target in self.targets:
            if not target['active'] or target['hit']:
                continue

            prev_x = target['x'][-1]
            prev_y = target['y'][-1]
            prev_z = target['z'][-1]
            prev_vx = target['vx'][-1]
            prev_vy = target['vy'][-1]
            prev_vz = target['vz'][-1]

            new_x = prev_x + prev_vx * self.dt
            new_y = prev_y + prev_vy * self.dt
            new_z = prev_z + prev_vz * self.dt

            target['x'].append(new_x)
            target['y'].append(new_y)
            target['z'].append(new_z)
            target['vx'].append(prev_vx)
            target['vy'].append(prev_vy)
            target['vz'].append(prev_vz)
            target['ax'].append(0)
            target['ay'].append(0)
            target['az'].append(0)

            if new_z <= 0:
                target['active'] = False
                target['ground_impact'] = True
                target['ground_impact_time'] = t

    def inject_target_state(self, tid, x, y, z, vx, vy, vz, t=None, ax=0, ay=0, az=0):
        """Inject live position/velocity (for live data mode)"""
        if tid >= len(self.targets):
            print(f"[TARGET_SYSTEM] Invalid TID={tid}")
            return
        
        target = self.targets[tid]
        target['x'].append(x)
        target['y'].append(y)
        target['z'].append(z)
        target['vx'].append(vx)
        target['vy'].append(vy)
        target['vz'].append(vz)
        target['ax'].append(ax)
        target['ay'].append(ay)
        target['az'].append(az)
        target['active'] = True
        target['hit'] = False

    def get_primary_target(self, t):
        for target in self.targets:
            if target['active'] and not target['hit']:
                idx = int(t / self.dt)
                if idx >= len(target['x']):
                    idx = len(target['x']) - 1
                return {
                    'x': target['x'][idx],
                    'y': target['y'][idx],
                    'z': target['z'][idx],
                    'vx': target['vx'][idx],
                    'vy': target['vy'][idx],
                    'vz': target['vz'][idx],
                    'ax': 0, 'ay': 0, 'az': 0,
                    'config': target['config']
                }
        return None

    def assess_threat(self, missile_x, missile_y, missile_z, t):
        """Single source of threat assessment."""
        threats = []

        for tid, target in enumerate(self.targets):
            if not target['active'] or target.get('hit', False):
                continue

            idx = int(t / self.dt)
            if idx >= len(target['x']):
                idx = len(target['x']) - 1

            tx, ty, tz = target['x'][idx], target['y'][idx], target['z'][idx]
            tvx, tvy, tvz = target['vx'][idx], target['vy'][idx], target['vz'][idx]
            cfg = target['config']

            dx = -tx
            dy = -ty
            dz = -tz
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            velocity_mag = np.sqrt(tvx**2 + tvy**2 + tvz**2)

            if distance > 0.1 and velocity_mag > 0.1:
                pos_unit = np.array([dx / distance, dy / distance, dz / distance])
                vel_unit = np.array([tvx / velocity_mag, tvy / velocity_mag, tvz / velocity_mag])
                cosine_similarity = np.dot(pos_unit, vel_unit)
                approach_angle = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))

                if approach_angle > APPROACH_ANGLE_THRESHOLD:
                    direction_threat = 0
                else:
                    direction_threat = 1.0 - (approach_angle / np.pi)

                distance_threat = max(0, min(1, 1.0 - (distance / 2000.0)))
                speed_threat = min(1, velocity_mag / 150.0)
                altitude_threat = max(0, min(1, 1.0 - (tz / 2000.0)))
            else:
                direction_threat = 0
                distance_threat = max(0, min(1, 1.0 - (distance / 2000.0)))
                speed_threat = min(1, velocity_mag / 150.0)
                altitude_threat = max(0, min(1, 1.0 - (tz / 2000.0)))

            is_within_coverage = distance <= RADAR_MAX_RANGE
            is_in_defended_zone = distance <= DEFENDED_RADIUS

            if not is_within_coverage and not is_in_defended_zone:
                distance_threat = 0
                direction_threat = 0

            total = (distance_threat * 0.40 +
                     speed_threat * 0.30 +
                     altitude_threat * 0.30 +
                     direction_threat * 0.25)

            if total >= 0.1:
                threats.append((tid, total))

        threats.sort(key=lambda x: x[1], reverse=True)
        return threats

    def get_active_targets(self, t):
        """Return dict of only active targets keyed by tid."""
        active = {}
        for tid, target in enumerate(self.targets):
            if target['active'] and not target.get('hit', False):
                idx = min(int(t / self.dt), len(target['x']) - 1)
                active[tid] = {
                    'id': tid,
                    'x': target['x'][idx], 'y': target['y'][idx], 'z': target['z'][idx],
                    'vx': target['vx'][idx], 'vy': target['vy'][idx], 'vz': target['vz'][idx],
                    'config': target['config']
                }
        return active

    def destroy_target(self, tid, killer_name, hit_time):
        """Mark target as destroyed."""
        if tid < len(self.targets):
            target = self.targets[tid]
            target['hit'] = True
            target['active'] = False
            target['hit_by'] = killer_name
            target['hit_time'] = hit_time
