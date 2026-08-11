"""Main simulation orchestrator with animation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D, art3d
from missile import Missile
from target_system import MultiTargetSystem
from radar import Radar
from config import SIM_TIME, DT

class ProNavSimulator:
    def __init__(self, dt=0.1, N=4, sim_time=90, multi_target_system=None,
                 missile_configs=None, launch_delay=0.0):
        self.dt = dt
        self.N = N
        self.sim_time = sim_time
        self.launch_delay = launch_delay
        self.multi_target = multi_target_system

        n_steps = int(sim_time / dt) + 1
        self.time = np.zeros(n_steps)

        self.target_x = np.zeros(n_steps)
        self.target_y = np.zeros(n_steps)
        self.target_z = np.zeros(n_steps)

        self.missiles = []
        self.radars = []
        self.missile_configs = missile_configs or []

        for cfg in missile_configs:
            missile = Missile(cfg, dt, n_steps, sim_time, multi_target_system)
            missile.launch_delay = self.launch_delay
            self.missiles.append(missile)

    def run_simulation(self, live_feed=None):
        """Run the main simulation loop."""
        n_steps = len(self.time)
        num_targets = len(self.multi_target.targets) if self.multi_target else 0
        target_radar = {
            tid: {'x': np.zeros(n_steps), 'y': np.zeros(n_steps), 'z': np.zeros(n_steps)}
            for tid in range(num_targets)
        }

        # ===== PHASE 1: Initial Assignment =====
        threats = self.multi_target.assess_threat(0, 0, 0, 0)
        threats_list = list(threats)
        num_missiles = len(self.missiles)

        missile_assignments = {}
        target_counts = {tid: 0 for tid, _ in threats}
        unassigned_missiles = list(self.missiles)

        # Selection mode
        from config import MAX_MISSILES_PER_TARGET, MISSILE_SELECTION_MODE
        if MISSILE_SELECTION_MODE == 1:
            for tid, _ in threats_list:
                if len(unassigned_missiles) <= 0:
                    break
                assigned_this_target = 0
                while assigned_this_target < MAX_MISSILES_PER_TARGET and len(unassigned_missiles) > 0:
                    m = unassigned_missiles.pop(0)
                    missile_assignments[m.config.name] = tid
                    target_counts[tid] += 1
                    assigned_this_target += 1
        elif MISSILE_SELECTION_MODE == 2:
            first_sweep_complete = False
            while len(unassigned_missiles) > 0 and not first_sweep_complete:
                first_sweep_complete = True
                for tid, _ in threats_list:
                    if len(unassigned_missiles) <= 0:
                        break
                    if target_counts[tid] < 1:
                        m = unassigned_missiles.pop(0)
                        missile_assignments[m.config.name] = tid
                        target_counts[tid] += 1
                        first_sweep_complete = False
            for tid, _ in threats_list:
                if len(unassigned_missiles) <= 0:
                    break
                while target_counts[tid] < MAX_MISSILES_PER_TARGET and len(unassigned_missiles) > 0:
                    m = unassigned_missiles.pop(0)
                    missile_assignments[m.config.name] = tid
                    target_counts[tid] += 1
        else:
            for tid, _ in threats_list:
                if len(unassigned_missiles) <= 0:
                    break
                assigned_this_target = 0
                while assigned_this_target < MAX_MISSILES_PER_TARGET and len(unassigned_missiles) > 0:
                    m = unassigned_missiles.pop(0)
                    missile_assignments[m.config.name] = tid
                    target_counts[tid] += 1
                    assigned_this_target += 1

        if len(unassigned_missiles) > 0 and threats_list:
            highest_threat_tid = threats_list[0][0]
            while (len(unassigned_missiles) > 0
                   and target_counts[highest_threat_tid] < MAX_MISSILES_PER_TARGET):
                m = unassigned_missiles.pop(0)
                missile_assignments[m.config.name] = highest_threat_tid
                target_counts[highest_threat_tid] += 1

        self.initial_missile_assignments = missile_assignments.copy()

        self.radar_x = np.zeros(n_steps)
        self.radar_y = np.zeros(n_steps)
        self.radar_z = np.zeros(n_steps)

        # ===== MAIN SIMULATION LOOP =====
        from config import KFP_NOISE, KFM_NOISE, RADAR_FUSION
        from kalman_filter import KalmanFilter3D

        for i in range(n_steps - 1):
            t = i * self.dt
            self.time[i] = t
            dt = self.dt

            # 1. Advance targets
            if live_feed:
                live_states = live_feed.get_data(t)
                for state in live_states:
                    self.multi_target.inject_target_state(
                        tid=state['tid'], x=state['x'], y=state['y'], z=state['z'],
                        vx=state.get('vx', 0), vy=state.get('vy', 0), vz=state.get('vz', 0),
                        t=t
                    )
            else:
                self.multi_target.step_all(t)

            # 2. Build active targets snapshot
            active_targets = self.multi_target.get_active_targets(t)
            threats_now = self.multi_target.assess_threat(0, 0, 0, t)

            # 3. Radar detection per target
            per_target_detections = {}
            for tid, tgt in active_targets.items():
                positions = []
                for r in self.radars:
                    r.last_detection_time = -np.inf
                    r.detect(tgt['x'], tgt['y'], tgt['z'],
                             tgt['vx'], tgt['vy'], tgt['vz'],
                             0, 0, 0, t)
                    det = r.get_latest_estimate()
                    if det and det[0] is not None:
                        positions.append(det[0])

                if not tgt.get('target_kf_initialized', False):
                    initial_pos = np.array([0.0, 0.0, 0.0])
                    tgt['target_kf'] = KalmanFilter3D(initial_pos, process_noise=KFP_NOISE, measurement_noise=KFM_NOISE)
                    tgt['target_kf_initialized'] = True

                if len(positions) >= 1:
                    dx = sum(p[0] for p in positions) / len(positions)
                    dy = sum(p[1] for p in positions) / len(positions)
                    dz = sum(p[2] for p in positions) / len(positions)

                    tgt['target_kf'].predict(dt)
                    tgt['target_kf'].update(np.array([dx, dy, dz]))
                    filtered_pos = tgt['target_kf'].x[:3]
                    dx, dy, dz = filtered_pos
                    tgt['kf_predict_count'] = tgt['target_kf'].predict_count
                    tgt['kf_update_count'] = tgt['target_kf'].update_count

                per_target_detections[tid] = (dx, dy, dz)
                target_radar[tid]['x'][i] = dx
                target_radar[tid]['y'][i] = dy
                target_radar[tid]['z'][i] = dz

            # 4. Legacy primary target arrays
            if active_targets:
                primary_tid = threats_now[0][0] if threats_now else list(active_targets.keys())[0]
                if primary_tid in active_targets:
                    primary = active_targets[primary_tid]
                    self.radar_x[i], self.radar_y[i], self.radar_z[i] = \
                        per_target_detections.get(primary_tid, (primary['x'], primary['y'], primary['z']))

            # 5. Ground impact check
            for tid, tgt in list(active_targets.items()):
                if tgt['z'] <= 0:
                    print(f"\n[{tgt['config'].name}] ✗ HIT GROUND at t={t:.1f}s")
                    self.multi_target.destroy_target(tid, 'GROUND', t)

            # 6. Missile guidance
            from config import INTERCEPT_DISTANCE, INTERCEPT_CONFIRMATION_FRAMES, GROUND_IMPACT_ALTITUDE, MAX_RECOVERY_ATTEMPTS

            target_has_active_missile = set()
            for m_check in self.missiles:
                if m_check.active and m_check.launch_triggered:
                    assigned_tid = missile_assignments.get(m_check.config.name, None)
                    if assigned_tid is not None:
                        target_obj = self.multi_target.targets[assigned_tid] if assigned_tid < len(self.multi_target.targets) else None
                        if target_obj and target_obj.get('active', False) and not target_obj.get('hit', False):
                            target_has_active_missile.add(assigned_tid)

            for m in self.missiles:
                if not m.active:
                    continue

                if m.config.name not in missile_assignments:
                    continue

                assigned_id = missile_assignments.get(m.config.name, 0)

                # Retargeting for launched missiles
                if not m.launch_triggered:
                    pass
                else:
                    target_obj = self.multi_target.targets[assigned_id] if assigned_id < len(self.multi_target.targets) else None

                    if target_obj is None or not target_obj.get('active', False) or target_obj.get('hit', False):
                        current_threats = self.multi_target.assess_threat(0, 0, 0, t)

                        remaining_threats = [th for th in current_threats
                                            if th[0] != assigned_id
                                            and th[0] < len(self.multi_target.targets)
                                            and self.multi_target.targets[th[0]].get('active', False)
                                            and th[0] not in target_has_active_missile]

                        if not remaining_threats:
                            remaining_threats = [th for th in current_threats
                                                if th[0] != assigned_id
                                                and th[0] < len(self.multi_target.targets)
                                                and self.multi_target.targets[th[0]].get('active', False)]

                        if remaining_threats:
                            new_tid = remaining_threats[0][0]

                            missile_assignments[m.config.name] = new_tid
                            m.locked_target_id = new_tid
                            assigned_id = new_tid

                            new_target = active_targets.get(new_tid)
                            if new_target:
                                m._recalculate_velocity_at_launch(i, new_target['x'], new_target['y'], new_target['z'],
                                                                 new_target['vx'], new_target['vy'], new_target['vz'])
                                m.consecutive_close_frames = 0
                                m.state = 'engaged'
                                m.overshoot_detected = False
                                m.recovery_count = 0
                                m.prev_vc = None
                                m.prev_los_az = None
                                m.prev_los_el = None
                                print(f"  [{m.config.name}] ↻ RETARGETING to T{new_tid}")
                            else:
                                print(f"  [{m.config.name}] WARNING: New target {new_tid} not in active targets!")
                                m.active = False
                                continue
                        else:
                            print(f"  [{m.config.name}] No valid targets available - missile deactivates")
                            m.active = False
                            continue

                # Radar data
                if assigned_id in target_radar:
                    m_radar_x = target_radar[assigned_id]['x'][i]
                    m_radar_y = target_radar[assigned_id]['y'][i]
                    m_radar_z = target_radar[assigned_id]['z'][i]
                    m_radar_x_prev = target_radar[assigned_id]['x'][max(0,i-1)]
                    m_radar_y_prev = target_radar[assigned_id]['y'][max(0,i-1)]
                    m_radar_z_prev = target_radar[assigned_id]['z'][max(0,i-1)]
                else:
                    m_radar_x = self.radar_x[i]
                    m_radar_y = self.radar_y[i]
                    m_radar_z = self.radar_z[i]
                    m_radar_x_prev = m_radar_x
                    m_radar_y_prev = m_radar_y
                    m_radar_z_prev = m_radar_z

                # Launch
                if t >= m.config.launch_delay and not m.launch_triggered:
                    if assigned_id in active_targets:
                        assigned_target = active_targets[assigned_id]
                        target_for_missile = (assigned_target['x'], assigned_target['y'], assigned_target['z'],
                                             assigned_target['vx'], assigned_target['vy'], assigned_target['vz'])
                    else:
                        target_for_missile = (m_radar_x, m_radar_y, m_radar_z, 0, 0, 0)

                    print(f"  [{m.config.name}] LAUNCH at t={t:.0f}s (Target TID: {assigned_id})")
                    m.launch_time = t
                    m._recalculate_velocity_at_launch(i, *target_for_missile[:3], *target_for_missile[3:])
                    m.state = 'engaged'
                    m.launch_triggered = True
                    m.consecutive_close_frames = 0
                    m.locked_target_id = assigned_id

                # Guide
                if m.launch_triggered:
                    m.step(i, t, m_radar_x, m_radar_y, m_radar_z,
                           m_radar_x_prev, m_radar_y_prev, m_radar_z_prev)

            # 7. Target destruction
            for m in self.missiles:
                if m.intercept_detected and m.intercept_step is not None:
                    assigned_id = missile_assignments.get(m.config.name, 0)
                    m.hit_target_id = assigned_id

                    target = self.multi_target.targets[assigned_id] if assigned_id < len(self.multi_target.targets) else None
                    if target and not target.get('hit', False):
                        print(f"  [{m.config.name}] ✓ IMPACT FINALIZED on Target TID={assigned_id}")
                        self.multi_target.destroy_target(assigned_id, m.config.name, m.hit_time)

        # Cleanup
        for m in self.missiles:
            fl = m.intercept_step + 1 if m.intercept_step else n_steps
            m.trim_arrays(fl)

        # Results summary
        print("\n" + "=" * 70)
        print("SIMULATION RESULTS SUMMARY")
        print("=" * 70)

        total_hits = 0
        total_standby = 0
        total_missed = 0
        total_recovered = 0
        total_failed_recoveries = 0

        for m in self.missiles:
            if m.intercept_detected:
                final_dist = m.distance[-1] if len(m.distance) > 0 else m.closest_dist_seen
                target_name = "?"
                if hasattr(m, 'hit_target_id') and m.hit_target_id is not None:
                    tid = m.hit_target_id
                    if tid < len(self.multi_target.targets):
                        target_name = self.multi_target.targets[tid]['config'].name
                print(f"  {m.config.name:6s}: ✓ HIT {target_name:3s} at {m.hit_time:.1f}s within {final_dist:.1f}m")
                total_hits += 1
            elif m.config.name not in missile_assignments:
                print(f"  {m.config.name:12s}: ⏸ STANDBY")
                total_standby += 1
            else:
                final_dist = m.distance[-1] if len(m.distance) > 0 else 0
                if m.recovery_count > 0:
                    recovered_status = f"| Recovered: {m.recovery_count}×"
                    total_recovered += 1
                else:
                    recovered_status = ""

                if m.recovery_count > m.max_recovery_attempts:
                    failed_status = "✗ FAILED (max recovery)"
                    total_failed_recoveries += 1
                else:
                    failed_status = "✗ Missed"

                print(f"  {m.config.name:12s}: {failed_status} (closest: {m.closest_dist_seen:.1f}m, final: {final_dist:.1f}m){recovered_status}")
                total_missed += 1

        print("-" * 70)
        print(f"Total {total_hits} hits out of {len(self.missiles)} missiles launched on total of {num_targets} targets")
        print(f"Standby Missiles: {total_standby}")
        print(f"Assigned, but missed (retargeting included): {total_missed}")
        print(f"Recovered Misses: {total_recovered}")
        print(f"Failed Recovery Attempts: {total_failed_recoveries}")
        print("=" * 70 + "\n")

        self.missile_assignments = missile_assignments
        self.initial_missile_assignments = missile_assignments.copy()

    def animate(self, interval=50, save_video=False, output_filename='pronav_sim.mp4', fps=30):
        """3D animation with 3 panels: 3D view, timeline, debug info."""
        fig = plt.figure(figsize=(16, 12))
        num_missiles = len(self.missiles)
        num_radars = len(self.radars)
        base_3d = 3.5
        time_height = 0.15
        debug_per_missile = 0.06
        debug_height = 0.2 + (num_missiles + num_radars - 1) * debug_per_missile
        gs = GridSpec(3, 1, figure=fig, height_ratios=[base_3d, time_height, debug_height])

        ax_main = fig.add_subplot(gs[0, :], projection='3d')
        ax_time = fig.add_subplot(gs[1, :])
        ax_debug = fig.add_subplot(gs[2, :])

        ax_main.patch.set_facecolor('black')
        title_text = (f"3D PRONAV SIMULATION (MULTI-MODE GUIDANCE) | "
                      f"Missiles: {num_missiles} | Targets: {len(self.multi_target.targets) if self.multi_target else 0} | "
                      f"N={self.N} | Modes: Engage→HardTurn→BodyProNav")
        fig.suptitle(title_text, color='black', fontsize=12, weight='bold', y=0.995)

        max_frames = max(max(len(m.time) for m in self.missiles), len(self.time))
        ax_main.view_init(elev=15, azim=-45)

        X_MIN, X_MAX = -700, 700
        Y_MIN, Y_MAX = -700, 700
        Z_MIN, Z_MAX = 0, 2300

        writer = None
        if save_video:
            try:
                writer = FFMpegWriter(fps=fps, extra_args=['-pix_fmt', 'yuv420p'])
            except Exception as e:
                print(f"[VIDEO] Warning: {e}")
                save_video = False

        radar_angles = [0.0 for _ in self.radars]

        def get_target_for_animation(missile, frame, ct):
            if hasattr(self, 'initial_missile_assignments'):
                assigned_id = self.initial_missile_assignments.get(missile.config.name, None)
                if assigned_id is not None:
                    return assigned_id
            if hasattr(self, 'missile_assignments'):
                assigned_id = self.missile_assignments.get(missile.config.name, None)
                if assigned_id is not None:
                    return assigned_id
            assigned_id = getattr(missile, 'locked_target_id', None)
            if assigned_id is not None:
                return assigned_id
            return None

        def update(frame):
            ct = self.time[frame] if frame < len(self.time) else self.time[-1]

            if hasattr(self, 'radars'):
                for idx, r in enumerate(self.radars):
                    delta_angle = r.sweep_freq * 2 * np.pi * (interval / 1000.0)
                    radar_angles[idx] = (radar_angles[idx] + delta_angle) % (2 * np.pi)

            ax_main.clear()
            ax_time.clear()
            ax_debug.clear()

            ax_main.view_init(elev=15, azim=-45)
            fig.set_facecolor('white')
            ax_main.set_facecolor('black')
            ax_time.set_facecolor('black')
            ax_debug.set_facecolor('black')

            ax_main.set_xlim(X_MIN, X_MAX)
            ax_main.set_ylim(Y_MIN, Y_MAX)
            ax_main.set_zlim(Z_MIN, Z_MAX)
            ax_main.tick_params(colors='white')
            ax_main.grid(True)
            ax_main.patch.set_facecolor('black')

            verts = [[X_MIN, Y_MIN, 0], [X_MAX, Y_MIN, 0], [X_MAX, Y_MAX, 0], [X_MIN, Y_MAX, 0]]
            ground = art3d.Poly3DCollection([verts], alpha=0.1, facecolor='black', edgecolor='none')
            ax_main.add_collection3d(ground)

            # Draw targets
            if self.multi_target:
                for target in self.multi_target.targets:
                    track_len = len(target['x'])
                    if track_len <= 1:
                        continue

                    idx = int(ct / self.multi_target.dt)
                    if idx >= len(target['x']):
                        idx = len(target['x']) - 1

                    track_end = min(idx + 1, len(target['x']))
                    ax_main.plot3D(target['x'][:track_end], target['y'][:track_end], target['z'][:track_end],
                                  color='green', linestyle='dotted', linewidth=2, alpha=0.7)

                    curr_x = target['x'][idx]
                    curr_y = target['y'][idx]
                    curr_z = target['z'][idx]
                    tvx = target['vx'][idx] if idx < len(target['vx']) else 0
                    tvy = target['vy'][idx] if idx < len(target['vy']) else 0
                    tvz = target['vz'][idx] if idx < len(target['vz']) else 0

                    vel_mag = np.sqrt(tvx**2 + tvy**2 + tvz**2)
                    if vel_mag > 0.1:
                        fixed_length = 20.0
                        fx = (tvx / vel_mag) * fixed_length
                        fy = (tvy / vel_mag) * fixed_length
                        fz = (tvz / vel_mag) * fixed_length
                    else:
                        fx, fy, fz = 0, 0, 0

                    ax_main.quiver(curr_x, curr_y, curr_z, fx, fy, fz,
                                  color='lime', linewidth=2, arrow_length_ratio=1.5, alpha=0.8)

            # Draw missiles
            for m in self.missiles:
                ml = min(frame, len(m.x) - 1)

                if ml < 0 or not hasattr(m, 'x') or len(m.x) == 0:
                    continue

                if m.launch_time is None or ct < m.launch_time:
                    x_track = np.full(min(ml+1, len(m.x)), m.x[0])
                    y_track = np.full(min(ml+1, len(m.y)), m.y[0])
                    z_track = np.full(min(ml+1, len(m.z)), m.z[0])
                    ax_main.plot3D(x_track, y_track, z_track,
                                  color='red', linestyle='dotted', linewidth=0.5, alpha=0.7)
                else:
                    actual_len = min(ml+1, len(m.x))
                    ax_main.plot3D(m.x[:actual_len], m.y[:actual_len], m.z[:actual_len],
                                  color='red', linestyle='dotted', linewidth=0.5, alpha=0.7)

            # Draw LOS lines to targets
            for m in self.missiles:
                ml = min(frame, len(m.x) - 1) if hasattr(m, 'x') and len(m.x) > 0 else -1

                if ml < 0:
                    continue

                mmx, mmy, mmz = m.x[ml], m.y[ml], m.z[ml]
                assigned_id = get_target_for_animation(m, frame, ct)

                if assigned_id is None:
                    continue

                tx, ty, tz = 0, 0, 1500

                if assigned_id < len(self.multi_target.targets):
                    target = self.multi_target.targets[assigned_id]
                    idx = int(ct / self.multi_target.dt)

                    if idx >= len(target['x']):
                        idx = len(target['x']) - 1

                    if idx >= 0 and idx < len(target['z']) and target['z'][idx] > 0:
                        tx = target['x'][idx]
                        ty = target['y'][idx]
                        tz = target['z'][idx]

                ax_main.plot3D([tx, mmx], [ty, mmy], [tz, mmz],
                              color='blue', linestyle='dotted', linewidth=0.5, alpha=0.8)

            # Draw radars
            if hasattr(self, 'radars'):
                for idx, r in enumerate(self.radars):
                    angle = radar_angles[idx]
                    dx = np.cos(angle) * np.cos(np.pi/4)
                    dy = np.sin(angle) * np.cos(np.pi/4)
                    dz = np.sin(np.pi/4)
                    ax_main.quiver(r.x, r.y, r.z, dx, dy, dz,
                                   color='green', linewidth=2, length=75, arrow_length_ratio=0.5)
                    ax_main.scatter(r.x, r.y, r.z, color='green', s=20, alpha=0.8, zorder=10, marker='o')

            # Draw velocity vectors for missiles
            for m in self.missiles:
                ml = min(frame, len(m.x) - 1)
                if ml < 0:
                    continue

                fixed_arrow_length = 35.0

                if m.launch_time is None or ct < m.launch_time:
                    mmx, mmy, mmz = m.config.x0, m.config.y0, m.config.z0
                    assigned_id = get_target_for_animation(m, frame, ct)

                    if assigned_id is not None:
                        tx_local, ty_local, tz_local = 0, 0, 1500
                        if assigned_id < len(self.multi_target.targets):
                            target = self.multi_target.targets[assigned_id]
                            idx = int(ct / self.multi_target.dt)
                            idx = min(idx, len(target['x']) - 1)
                            if idx >= 0 and idx < len(target['z']) and target['z'][idx] > 0:
                                tx_local = target['x'][idx]
                                ty_local = target['y'][idx]
                                tz_local = target['z'][idx]

                        aim_dx = tx_local - mmx
                        aim_dy = ty_local - mmy
                        aim_dz = tz_local - mmz
                        aim_dist = np.sqrt(aim_dx**2 + aim_dy**2 + aim_dz**2)

                        if aim_dist > 0.1:
                            mvx = (aim_dx / aim_dist) * fixed_arrow_length
                            mvy = (aim_dy / aim_dist) * fixed_arrow_length
                            mvz = (aim_dz / aim_dist) * fixed_arrow_length
                        else:
                            mvx, mvy, mvz = 0, 0, 0

                        ax_main.quiver(mmx, mmy, mmz, mvx, mvy, mvz,
                                     color='red', linewidth=1.5, arrow_length_ratio=0.5)

                else:
                    current_idx = min(frame, len(m.x) - 1)
                    mmx, mmy, mmz = m.x[current_idx], m.y[current_idx], m.z[current_idx]

                    if current_idx < len(m.vx):
                        mvx = m.vx[current_idx]
                        mvy = m.vy[current_idx]
                        mvz = m.vz[current_idx]

                        vel_mag = np.sqrt(mvx**2 + mvy**2 + mvz**2)
                        if vel_mag > 0.1:
                            mvx = (mvx / vel_mag) * fixed_arrow_length
                            mvy = (mvy / vel_mag) * fixed_arrow_length
                            mvz = (mvz / vel_mag) * fixed_arrow_length
                        else:
                            mvx, mvy, mvz = 0, 0, 0

                        if m.state == 'overshoot':
                            color = 'yellow'
                        elif m.state == 'recovery':
                            color = 'cyan'
                        else:
                            color = 'red'

                        ax_main.quiver(mmx, mmy, mmz, mvx, mvy, mvz,
                                     color=color, linewidth=1.5, arrow_length_ratio=0.5)

            # Draw intercept markers
            for m in self.missiles:
                if m.intercept_detected and m.intercept_step is not None and frame >= m.intercept_step:
                    ax_main.scatter([m.intercept_tx], [m.intercept_ty], [m.intercept_tz],
                                   c='cyan', s=70, alpha=0.6, marker='o',
                                   edgecolors='blue', linewidths=2, zorder=25)

            hits = sum(1 for m in self.missiles if m.intercept_detected and frame >= m.intercept_step)

            tracking_count = 0
            engaged_count = 0
            overshoot_count = 0
            recovery_count = 0

            for m in self.missiles:
                if m.intercept_detected and frame >= m.intercept_step:
                    continue
                if m.launch_time is None or ct < m.launch_time:
                    tracking_count += 1
                elif ct >= m.launch_time:
                    if m.state == 'engaged':
                        engaged_count += 1
                    elif m.state == 'overshoot':
                        overshoot_count += 1
                    elif m.state == 'recovery':
                        recovery_count += 1

            if hits > 0:
                st = f"[{hits} out of {len(self.missiles)} HIT]"
                sc = 'green'
            elif engaged_count > 0 or overshoot_count > 0 or recovery_count > 0:
                st = f"[ENG={engaged_count} | OT={overshoot_count} | REC={recovery_count} | TRK={tracking_count}]"
                sc = 'red'
            else:
                st = f"[{tracking_count} TRACKING]"
                sc = 'blue'

            ax_main.text2D(0.02, 0.02, st,
                transform=ax_main.transAxes,
                fontsize=12, weight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=sc, alpha=0.6, edgecolor='white'),
                verticalalignment='bottom',
                horizontalalignment='left')

            ax_main.set_title(f'3D ProNav (N={self.N}) - Red=Engaged/Yellow=HardTurn/Cyan=BodyProNav/Lime=Targets/Blue=LOS',
                            color='white', fontsize=14, pad=20)
            ax_main.set_xlabel('X (m)', color='white', fontsize=10)
            ax_main.set_ylabel('Y (m)', color='white', fontsize=10)
            ax_main.set_zlabel('Z (m)', color='white', fontsize=10)

            # Timeline panel
            ax_time.set_facecolor('black')
            ax_time.set_xlim(0, self.sim_time)
            ax_time.set_ylim(-1, 1)
            ax_time.tick_params(colors='white')
            ax_time.axhline(0, color='white', lw=1)
            ax_time.scatter([ct], [0], c='blue', s=30, zorder=10)

            for m in self.missiles:
                if m.launch_time is not None:
                    ax_time.axvline(m.launch_time, color='green', lw=1.5, alpha=0.8)
                if m.hit_time is not None:
                    ax_time.axvline(m.hit_time, color='red', lw=1.5, alpha=0.95)
                if m.overshoot_time is not None:
                    ax_time.axvline(m.overshoot_time, color='yellow', lw=1.5, alpha=0.7)
                if m.recovery_start_time is not None:
                    ax_time.axvline(m.recovery_start_time, color='cyan', lw=1.5, alpha=0.7)

            ax_time.set_xlabel('Time (s)', color='white', fontsize=10)
            ax_time.set_yticks([])

            # Debug panel
            try:
                ax_debug.clear()
                ax_debug.set_facecolor('black')

                debug_lines = []
                for m in self.missiles:
                    ml = min(frame, len(m.distance) - 1) if hasattr(m, 'distance') and len(m.distance) > 0 else 0

                    if ml < 0:
                        continue

                    has_hit = hasattr(m, 'hit_time') and m.hit_time is not None and ct >= m.hit_time
                    is_overshooting = m.overshoot_detected and ct >= m.overshoot_time and (not m.recovery_start_time or ct < m.recovery_start_time)
                    in_recovery = m.recovery_start_time and ct >= m.recovery_start_time

                    if m.launch_time is None:
                        phase = "TRACKING"
                    elif has_hit:
                        phase = f"HIT ({m.recovery_count}r)"
                    elif is_overshooting and not in_recovery:
                        phase = f"OVERSHOOT #{m.recovery_count}"
                    elif in_recovery:
                        phase = f"BODY-PRONAV #{m.recovery_count}"
                    elif ct >= m.launch_time + 1.5 and not in_recovery:
                        vc_val = m.vc[ml] if ml < len(m.vc) else 0.0
                        phase = "CLOSING" if vc_val > 0 else "OPENING"
                    elif ct >= m.launch_time:
                        vc_val = m.vc[ml] if ml < len(m.vc) else 0.0
                        phase = "CLOSING" if vc_val > 0 else "OPENING"
                    else:
                        phase = "TRACKING"

                    duty_time_str = "---"
                    launch_in_str = "---"
                    speed_show = 0
                    if m.launch_time is not None:
                        duty_time = max(0, ct - m.launch_time)
                        duty_time_str = f"{duty_time:.0f}s"
                        if ml < len(m.speed):
                            speed_show = m.speed[ml]
                        if ct < m.launch_time:
                            launch_in_str = f"{int(round(m.launch_time - ct))}"

                    los_body_deg = np.degrees(m.los_angle_body[ml]) if ml < len(m.los_angle_body) else 0.0
                    vc_val = m.vc[ml] if ml < len(m.vc) else 0.0
                    dist_val = m.distance[ml] if ml < len(m.distance) else 0.0
                    acc_val = m.acc_mag[ml] if ml < len(m.acc_mag) else 0.0

                    assigned_id = get_target_for_animation(m, frame, ct)
                    target_name = "----"
                    if assigned_id is not None and self.multi_target and assigned_id < len(self.multi_target.targets):
                        target_name = self.multi_target.targets[assigned_id]['config'].name

                    launch_field = f" | Launch in {launch_in_str:>2}s" if launch_in_str != "---" else ""
                    duty_field = f"Duty={duty_time_str:>3}" if m.launch_time is not None and ct >= m.launch_time else ""

                    debug_lines.append(
                        f"{m.config.name} | Target:{target_name:8s} | {phase:>5s} | d={dist_val:4.0f}m | "
                        f"Spd={speed_show:4.0f}m/s | Vc={vc_val:+4.0f}m/s | Acc={acc_val:4.0f}m/s² | LOS={los_body_deg:+6.1f}°{launch_field}{(' | ' + duty_field) if duty_field else ''}"
                    )

                target_status = []
                if self.multi_target:
                    for target in self.multi_target.targets:
                        idx = int(ct / self.multi_target.dt)
                        if idx >= len(target['x']):
                            idx = len(target['x']) - 1

                        if idx < 0 or idx >= len(target['z']):
                            target_status.append(f"{target['config'].name}: NO DATA")
                            continue

                        z_at_frame = target['z'][idx]

                        if z_at_frame <= 0:
                            hit_frame = 0
                            for fi, zi in enumerate(target['z']):
                                if zi <= 0:
                                    hit_frame = fi
                                    break
                            hit_time = hit_frame * self.multi_target.dt
                            target_status.append(f"{target['config'].name}: ✗ GROUND @ {hit_time:.1f}s")
                        elif target.get('hit', False):
                            target_status.append(f"{target['config'].name}: ✓ DESTROYED by {target.get('hit_by', '?'):.10s}")
                        else:
                            vx = target['vx'][idx] if idx < len(target['vx']) else 0
                            vy = target['vy'][idx] if idx < len(target['vy']) else 0
                            vz = target['vz'][idx] if idx < len(target['vz']) else 0
                            speed = np.sqrt(vx**2 + vy**2 + vz**2)
                            target_status.append(f"{target['config'].name}: ALT={z_at_frame:5.0f}m | SP={speed:5.0f}m/s")

                else:
                    target_status.append("No multi-target system")

                debug_text = f'=== TIME: {ct:.2f}s | Hits: {hits}/{len(self.missiles)} | Modes: ENG→OT→REC ===\n'

                if debug_lines:
                    debug_text += '\n'.join(debug_lines) + '\n\n'
                else:
                    debug_text += "(No missile data)\n\n"

                debug_text += '=== TARGETS ===\n'
                debug_text += '\n'.join(target_status)

                ax_debug.text(0.5, 0.5, debug_text,
                             ha='center', va='center', fontsize=8,
                             color='lime', family='monospace',
                             bbox=dict(facecolor='black', edgecolor='white', alpha=0.95, pad=1.1, boxstyle='round'))

                ax_debug.set_xlim(0, 1)
                ax_debug.set_ylim(0, 1)
                ax_debug.axis('off')

            except Exception as e:
                try:
                    ax_debug.clear()
                    ax_debug.text(0.5, 0.5, f'DEBUG ERROR:\n{str(e)}',
                                 ha='center', va='center', fontsize=10, color='red')
                    ax_debug.set_xlim(0, 1)
                    ax_debug.set_ylim(0, 1)
                    ax_debug.axis('off')
                except:
                    pass

            return []

        ani = FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=False)

        if save_video and writer:
            print(f"[VIDEO] Saving...")
            plt.tight_layout()
            ani.save(output_filename, writer=writer)
            print(f"[VIDEO] Done: {output_filename}")
            plt.close()
        else:
            print("\n[ANIMATION] Displaying...")
            plt.tight_layout()
            plt.show()

        return ani
