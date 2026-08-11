"""Multi-mode ProNav Missile with state machine"""
import numpy as np
from models import MissileConfig
from config import INTERCEPT_DISTANCE, INTERCEPT_CONFIRMATION_FRAMES, GROUND_IMPACT_ALTITUDE, MAX_RECOVERY_ATTEMPTS

class Missile:
    def __init__(self, config: MissileConfig, dt: float, n_steps: int, sim_time: float,
                 multi_target_system=None):
        self.config = config
        self.dt = dt
        self.sim_time = sim_time
        self.n_steps = n_steps
        self.multi_target = multi_target_system
        self.locked_target_id = None

        self.time = np.zeros(n_steps)
        self.x = np.full(n_steps, config.x0)
        self.y = np.full(n_steps, config.y0)
        self.z = np.full(n_steps, config.z0)
        self.vx = np.zeros(n_steps)
        self.vy = np.zeros(n_steps)
        self.vz = np.zeros(n_steps)
        self.speed = np.zeros(n_steps)
        self.los_angle = np.zeros(n_steps)
        self.los_rate = np.zeros(n_steps)
        self.los_angle_body = np.zeros(n_steps)
        self.vc = np.zeros(n_steps)
        self.distance = np.zeros(n_steps)
        self.acc_mag = np.zeros(n_steps)

        self.x[0] = config.x0
        self.y[0] = config.y0
        self.z[0] = config.z0
        self.x0 = config.x0
        self.y0 = config.y0
        self.z0 = config.z0

        self.launch_x0 = config.x0
        self.launch_y0 = config.y0
        self.launch_z0 = config.z0

        if multi_target_system:
            primary = multi_target_system.get_primary_target(0)
            if primary:
                dx = primary['x'] - config.x0
                dy = primary['y'] - config.y0
                dz = primary['z'] - config.z0
            else:
                dx, dy, dz = 1, 0, 0
        else:
            dx, dy, dz = 1, 0, 0

        init_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if init_dist > 0:
            self.vx[0] = config.cruise_speed * dx / init_dist
            self.vy[0] = config.cruise_speed * dy / init_dist
            self.vz[0] = config.cruise_speed * dz / init_dist
        else:
            self.vx[0] = config.cruise_speed
            self.vy[0] = 0.0
            self.vz[0] = 0.0

        self.speed[0] = np.sqrt(self.vx[0]**2 + self.vy[0]**2 + self.vz[0]**2)

        self.previous_filtered_los_rate = None
        self.prev_los_az = None
        self.prev_los_angle_body = None
        self.prev_los_el = None

        self.intercept_detected = False
        self.intercept_step = None
        self.intercept_mx = None
        self.intercept_my = None
        self.intercept_mz = None
        self.intercept_tx = None
        self.intercept_ty = None
        self.intercept_tz = None
        self.consecutive_close_frames = 0

        self.launch_time = None
        self.launch_triggered = False
        self.hit_time = None

        self.state = 'tracking'
        self.closest_dist_seen = 1e9
        self.active = True
        self.final_len = n_steps
        self.prev_dist = None

        self.overshoot_detected = False
        self.overshoot_time = None
        self.recovery_start_time = None
        self.recovery_count = 0
        self.max_recovery_attempts = MAX_RECOVERY_ATTEMPTS

        self.engagement_duration = 0.0
        self.turn_duration = 0.0
        self.recovery_duration = 0.0

        self.last_target_pos = {'x': None, 'y': None, 'z': None}
        self.last_target_vel = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
        self.prev_vc = None

    @staticmethod
    def _unwrap_angle(prev, curr):
        diff = curr - prev
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return prev + diff

    def _compute_guidance(self, i, target_x, target_y, target_z):
        dx = target_x - self.x[i]
        dy = target_y - self.y[i]
        dz = target_z - self.z[i]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 0.1:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        alpha = 0.3
        if self.last_target_pos['x'] is not None:
            t_vx_raw = (target_x - self.last_target_pos['x']) / self.dt
            t_vy_raw = (target_y - self.last_target_pos['y']) / self.dt
            t_vz_raw = (target_z - self.last_target_pos['z']) / self.dt
            t_vx = alpha * t_vx_raw + (1-alpha) * self.last_target_vel['vx']
            t_vy = alpha * t_vy_raw + (1-alpha) * self.last_target_vel['vy']
            t_vz = alpha * t_vz_raw + (1-alpha) * self.last_target_vel['vz']
            self.last_target_vel = {'vx': t_vx, 'vy': t_vy, 'vz': t_vz}
        else:
            t_vx, t_vy, t_vz = 0.0, 0.0, 0.0

        self.last_target_pos = {'x': target_x, 'y': target_y, 'z': target_z}

        m_speed = np.sqrt(self.vx[i]**2 + self.vy[i]**2 + self.vz[i]**2)
        los_x = dx / dist if dist > 0.1 else 0.0
        los_y = dy / dist if dist > 0.1 else 0.0
        los_z = dz / dist if dist > 0.1 else 0.0

        rel_vx = t_vx - self.vx[i]
        rel_vy = t_vy - self.vy[i]
        rel_vz = t_vz - self.vz[i]
        vc = -(rel_vx * los_x + rel_vy * los_y + rel_vz * los_z)

        if m_speed > 0.1 and vc > 1.0:
            time_to_go = dist / vc
        else:
            time_to_go = dist / max(m_speed, 1.0)

        time_to_go = min(max(time_to_go, 0.01), 3.0)

        pred_target_x = target_x + t_vx * time_to_go
        pred_target_y = target_y + t_vy * time_to_go
        pred_target_z = target_z + t_vz * time_to_go

        pdx = pred_target_x - self.x[i]
        pdy = pred_target_y - self.y[i]
        pdz = pred_target_z - self.z[i]
        pred_dist = np.sqrt(pdx**2 + pdy**2 + pdz**2)

        if pred_dist > 0.1:
            desired_x = pdx / pred_dist
            desired_y = pdy / pred_dist
            desired_z = pdz / pred_dist
        else:
            desired_x, desired_y, desired_z = los_x, los_y, los_z

        if m_speed > 0.1:
            current_x = self.vx[i] / m_speed
            current_y = self.vy[i] / m_speed
            current_z = self.vz[i] / m_speed
        else:
            current_x, current_y, current_z = desired_x, desired_y, desired_z

        error_x = desired_x - current_x
        error_y = desired_y - current_y
        error_z = desired_z - current_z

        N = self.config.N
        max_acc = 100.0

        base_gain = N * (100.0 / (dist + 50.0))
        accel_scale = min(base_gain * max_acc, max_acc)

        miss_dx = error_x * accel_scale
        miss_dy = error_y * accel_scale
        miss_dz = error_z * accel_scale

        cos_angle = current_x * desired_x + current_y * desired_y + current_z * desired_z
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        los_angle_body = np.arccos(cos_angle)
        self.los_angle_body[i] = los_angle_body

        los_azimuth = np.arctan2(dy, dx)

        return miss_dx, miss_dy, miss_dz, los_azimuth, 0.0, vc, los_angle_body

    def _compute_bodyframe_pronav(self, i, target_x, target_y, target_z, dt):
        """Body-frame ProNav for recovery - uses LOS rate only."""
        dx = target_x - self.x[i]
        dy = target_y - self.y[i]
        dz = target_z - self.z[i]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 0.1:
            return 0.0, 0.0, 0.0, 0.0

        los_az = np.arctan2(dy, dx)
        los_el = np.arctan2(dz, np.sqrt(dx**2 + dy**2))

        if self.prev_los_az is not None:
            d_los_az = self._unwrap_angle(self.prev_los_az, los_az)
            d_los_el = self._unwrap_angle(self.prev_los_el, los_el)
            los_rate_az = d_los_az / dt
            los_rate_el = d_los_el / dt
        else:
            los_rate_az = 0.0
            los_rate_el = 0.0

        self.prev_los_az = los_az
        self.prev_los_el = los_el

        m_speed = np.sqrt(self.vx[i]**2 + self.vy[i]**2 + self.vz[i]**2)
        los_x, los_y, los_z = dx/dist, dy/dist, dz/dist
        vc = -(self.vx[i]*los_x + self.vy[i]*los_y + self.vz[i]*los_z)

        cmd_az = self.config.N * abs(vc) * los_rate_az
        cmd_el = self.config.N * abs(vc) * los_rate_el

        perp_az_x, perp_az_y, perp_az_z = -np.sin(los_az), np.cos(los_az), 0.0
        perp_el_x = -np.sin(los_el) * np.cos(los_az)
        perp_el_y = -np.sin(los_el) * np.sin(los_az)
        perp_el_z = np.cos(los_el)

        acc_x = cmd_az * perp_az_x + cmd_el * perp_el_x
        acc_y = cmd_az * perp_az_y + cmd_el * perp_el_y
        acc_z = cmd_az * perp_az_z + cmd_el * perp_el_z

        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        self.acc_mag[i] = acc_mag
        if acc_mag > 100.0:
            scale = 100.0 / acc_mag
            acc_x *= scale
            acc_y *= scale
            acc_z *= scale

        return acc_x, acc_y, acc_z, vc

    def _compute_hardturn(self, i, target_x, target_y, target_z):
        """Max lateral acceleration turn toward target."""
        dx = target_x - self.x[i]
        dy = target_y - self.y[i]
        dz = target_z - self.z[i]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 0.1:
            return 0.0, 0.0, 0.0

        los_x, los_y, los_z = dx/dist, dy/dist, dz/dist
        m_speed = np.sqrt(self.vx[i]**2 + self.vy[i]**2 + self.vz[i]**2)

        if m_speed > 0.1:
            body_x = self.vx[i] / m_speed
            body_y = self.vy[i] / m_speed
            body_z = self.vz[i] / m_speed
        else:
            body_x, body_y, body_z = los_x, los_y, los_z

        error_x = los_x - body_x
        error_y = los_y - body_y
        error_z = los_z - body_z

        error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        if error_mag > 0.1:
            error_x /= error_mag
            error_y /= error_mag
            error_z /= error_mag

        return error_x * 100.0, error_y * 100.0, error_z * 100.0

    def _record_intercept(self, step, t, tx, ty, tz):
        self.intercept_detected = True
        self.intercept_step = step
        self.intercept_mx = self.x[step]
        self.intercept_my = self.y[step]
        self.intercept_mz = self.z[step]
        self.intercept_tx = tx
        self.intercept_ty = ty
        self.intercept_tz = tz
        self.hit_time = t
        self.active = False
        self.state = 'hit'

    def _recalculate_velocity_at_launch(self, i, target_x, target_y, target_z,
                                       target_vx=0, target_vy=0, target_vz=0):
        dx = target_x - self.x[i]
        dy = target_y - self.y[i]
        dz = target_z - self.z[i]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 0.1:
            return

        m_speed = self.config.cruise_speed
        los_x, los_y, los_z = dx/dist, dy/dist, dz/dist

        if m_speed > 0.1 and dist > 0.1:
            time_to_go = dist / m_speed
        else:
            time_to_go = dist / max(m_speed, 1.0)

        pred_x = target_x + target_vx * time_to_go
        pred_y = target_y + target_vy * time_to_go
        pred_z = target_z + target_vz * time_to_go

        pdx = pred_x - self.x[i]
        pdy = pred_y - self.y[i]
        pdz = pred_z - self.z[i]
        pred_dist = np.sqrt(pdx**2 + pdy**2 + pdz**2)

        if pred_dist > 0.1:
            self.vx[i] = self.config.cruise_speed * pdx / pred_dist
            self.vy[i] = self.config.cruise_speed * pdy / pred_dist
            self.vz[i] = self.config.cruise_speed * pdz / pred_dist
            self.speed[i] = self.config.cruise_speed
        else:
            self.vx[i] = self.config.cruise_speed * los_x
            self.vy[i] = self.config.cruise_speed * los_y
            self.vz[i] = self.config.cruise_speed * los_z

    def _check_intercept(self, dist):
        if dist < INTERCEPT_DISTANCE:
            self.consecutive_close_frames += 1
            if self.consecutive_close_frames >= INTERCEPT_CONFIRMATION_FRAMES:
                return True
        else:
            self.consecutive_close_frames = max(0, self.consecutive_close_frames - 1)
        return False

    def step(self, i, t, radar_tx, radar_ty, radar_tz, radar_tx_prev, radar_ty_prev, radar_tz_prev):
        if not self.active:
            return False

        dt = self.dt
        cruise = self.config.cruise_speed

        dx = radar_tx - self.x[i]
        dy = radar_ty - self.y[i]
        dz = radar_tz - self.z[i]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        self.distance[i] = dist
        self.time[i] = t

        if dist < self.closest_dist_seen:
            self.closest_dist_seen = dist

        if not self.intercept_detected and self._check_intercept(dist):
            print(f"  [{self.config.name}] HIT CONFIRMED at t={t:.1f}s, dist={dist:.1f}m, state={self.state}")
            self._record_intercept(i, t, radar_tx, radar_ty, radar_tz)
            return True

        if radar_tz < GROUND_IMPACT_ALTITUDE and not self.intercept_detected:
            print(f"  [{self.config.name}] TARGET HIT GROUND t={t:.1f}s, missed by {dist:.1f}m")
            self.active = False
            return False

        self.los_angle[i] = np.arctan2(dy, dx)
        self.speed[i] = np.sqrt(self.vx[i]**2 + self.vy[i]**2 + self.vz[i]**2)

        if self.state == 'tracking':
            self.x[i] = self.x0
            self.y[i] = self.y0
            self.z[i] = self.z0
            self.vx[i+1] = self.vx[i]
            self.vy[i+1] = self.vy[i]
            self.vz[i+1] = self.vz[i]
            self.x[i+1] = self.x0
            self.y[i+1] = self.y0
            self.z[i+1] = self.z0
            self.prev_dist = dist

        elif self.state == 'engaged':
            self.engagement_duration += dt
            acc_x, acc_y, acc_z, los, los_rate, vc, los_body = self._compute_guidance(i, radar_tx, radar_ty, radar_tz)
            self.los_angle[i] = los
            self.los_rate[i] = los_rate
            self.vc[i] = vc
            self.acc_mag[i] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            if abs(vc) > 300:
                self.vc[i] = max(-400, min(400, vc))

            if (self.prev_vc is not None and
                self.prev_vc > 5.0 and
                vc < -5.0 and
                dist < 700.0 and
                not self.overshoot_detected):

                self.overshoot_detected = True
                self.overshoot_time = t
                self.state = 'overshoot'
                self.recovery_count += 1
                print(f"  [{self.config.name}] ⚠ OVERSHOOT t={t:.1f}s (Vc: {self.prev_vc:+.1f}→{vc:+.1f}, d={dist:.1f}m)")

                if self.recovery_count > self.max_recovery_attempts:
                    print(f"  [{self.config.name}] ✗ MAX RECOVERY ATTEMPTS - deactivating")
                    self.active = False
                    return False

            self.vx[i+1] = self.vx[i] + acc_x * dt
            self.vy[i+1] = self.vy[i] + acc_y * dt
            self.vz[i+1] = self.vz[i] + acc_z * dt

            new_speed = np.sqrt(self.vx[i+1]**2 + self.vy[i+1]**2 + self.vz[i+1]**2)
            if new_speed > 0:
                ratio = cruise / new_speed
                self.vx[i+1] *= ratio
                self.vy[i+1] *= ratio
                self.vz[i+1] *= ratio

            self.x[i+1] = self.x[i] + self.vx[i] * dt
            self.y[i+1] = self.y[i] + self.vy[i] * dt
            self.z[i+1] = self.z[i] + self.vz[i] * dt

            if i % 20 == 0 and not self.intercept_detected:
                status = "CLOSING" if vc > 0 else "OPENING"
                print(f"  [{self.config.name}] {status} t={t:.1f}s | d={dist:.1f}m | Vc={vc:+.1f}")

            self.prev_dist = dist
            self.prev_vc = vc

        elif self.state == 'overshoot':
            self.turn_duration += dt
            acc_x, acc_y, acc_z = self._compute_hardturn(i, radar_tx, radar_ty, radar_tz)

            self.vx[i+1] = self.vx[i] + acc_x * dt
            self.vy[i+1] = self.vy[i] + acc_y * dt
            self.vz[i+1] = self.vz[i] + acc_z * dt

            new_speed = np.sqrt(self.vx[i+1]**2 + self.vy[i+1]**2 + self.vz[i+1]**2)
            if new_speed > 0:
                ratio = cruise / new_speed
                self.vx[i+1] *= ratio
                self.vy[i+1] *= ratio
                self.vz[i+1] *= ratio

            self.x[i+1] = self.x[i] + self.vx[i] * dt
            self.y[i+1] = self.y[i] + self.vy[i] * dt
            self.z[i+1] = self.z[i] + self.vz[i] * dt

            if t - self.overshoot_time >= 1.0:
                self.state = 'recovery'
                self.recovery_start_time = t
                print(f"  [{self.config.name}] ↻ BODY-PRONAV at t={t:.1f}s")

        elif self.state == 'recovery':
            self.recovery_duration += dt
            if i % 30 == 0:
                print(f"  [{self.config.name}] 🔄 BODY-PRONAV t={t:.1f}s | d={dist:.1f}m | Try #{self.recovery_count}")

            acc_x, acc_y, acc_z, vc = self._compute_bodyframe_pronav(i, radar_tx, radar_ty, radar_tz, dt)
            self.vc[i] = vc
            self.acc_mag[i] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            self.vx[i+1] = self.vx[i] + acc_x * dt
            self.vy[i+1] = self.vy[i] + acc_y * dt
            self.vz[i+1] = self.vz[i] + acc_z * dt

            new_speed = np.sqrt(self.vx[i+1]**2 + self.vy[i+1]**2 + self.vz[i+1]**2)
            if new_speed > 0:
                ratio = cruise / new_speed
                self.vx[i+1] *= ratio
                self.vy[i+1] *= ratio
                self.vz[i+1] *= ratio

            self.x[i+1] = self.x[i] + self.vx[i] * dt
            self.y[i+1] = self.y[i] + self.vy[i] * dt
            self.z[i+1] = self.z[i] + self.vz[i] * dt

            self.prev_dist = dist
            self.prev_vc = vc

        else:
            print(f"  [{self.config.name}] WARNING: Unknown state '{self.state}'")
            self.state = 'tracking'
            self.prev_dist = dist

        return False

    def trim_arrays(self, final_len):
        self.final_len = final_len
        for attr in ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'speed', 'los_angle', 'los_rate', 'vc', 'acc_mag']:
            arr = getattr(self, attr)
            if final_len < len(arr):
                setattr(self, attr, arr[:final_len])
        if len(self.distance) > final_len:
            self.distance = self.distance[:final_len]
