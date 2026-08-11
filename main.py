#!/usr/bin/env python3
"""Entry point for ProNav Multi-Target Simulation"""

from config import DT, SIM_TIME, N, SAVE_VIDEO, OUTPUT_FILE, VIDEO_FPS, LIVE_MODE, LIVE_FEED_TYPE, LIVE_DATA_PORT, LIVE_UDP_PORT, LIVE_LOG_FILE
from models import TargetConfig, MissileConfig
from target_system import MultiTargetSystem
from radar import Radar
from simulator import ProNavSimulator
from live_feed import create_live_feed

def main():
    print("=" * 60)
    print("3D PROPORTIONAL NAVIGATION SIMULATION (MULTI-MODE GUIDANCE)")
    print("=" * 60)

    target_configs = [
        TargetConfig(name="T1", x0=-600, y0=-600, z0=2200, speed=45, heading_deg=90, pitch_deg=15),
        TargetConfig(name="T2", x0=-300, y0= 600, z0=1800, speed=25, target_x=400, target_y=-400, target_z=0),
        TargetConfig(name="T3", x0=-100, y0= 500, z0=1900, speed=25, target_x=300, target_y=-400, target_z=0),
        TargetConfig(name="T4", x0= 100, y0= 600, z0=2000, speed=30, target_x=200, target_y=-400, target_z=0),
        TargetConfig(name="T5", x0= 300, y0= 500, z0=2100, speed=25, target_x=100, target_y=-400, target_z=0),
        TargetConfig(name="T6", x0= 400, y0= 600, z0=2300, speed=35, target_x=200, target_y=-400, target_z=0),
    ]

    missile_configs = [
        MissileConfig(name="Missile 1", x0=-400, y0=-400, z0=0, cruise_speed=55, N=N, launch_delay=1),
        MissileConfig(name="Missile 2", x0=-200, y0=-400, z0=0, cruise_speed=55, N=N, launch_delay=2),
        MissileConfig(name="Missile 3", x0= 200, y0=-400, z0=0, cruise_speed=55, N=N, launch_delay=3),
        MissileConfig(name="Missile 4", x0= 400, y0=-400, z0=0, cruise_speed=55, N=N, launch_delay=4),
        MissileConfig(name="Missile 5", x0=-400, y0= 400, z0=0, cruise_speed=55, N=N, launch_delay=1),
        MissileConfig(name="Missile 6", x0=-200, y0= 400, z0=0, cruise_speed=55, N=N, launch_delay=2),
        MissileConfig(name="Missile 7", x0= 200, y0= 400, z0=0, cruise_speed=55, N=N, launch_delay=3),
        MissileConfig(name="Missile 8", x0= 400, y0= 400, z0=0, cruise_speed=55, N=N, launch_delay=4),
    ]

    radar_configs = [
        Radar(x=-600, y=-600, sweep_freq=0.10, max_range=3000.0, noise_std=2, refresh_rate=0.1),
        Radar(x=-600, y= 600, sweep_freq=0.10, max_range=3000.0, noise_std=2, refresh_rate=0.1),
        Radar(x= 600, y=-600, sweep_freq=0.10, max_range=3000.0, noise_std=2, refresh_rate=0.1),
        Radar(x= 600, y= 600, sweep_freq=0.10, max_range=3000.0, noise_std=2, refresh_rate=0.1),
    ]

    multi_target = MultiTargetSystem(target_configs, dt=DT)
    sim = ProNavSimulator(dt=DT, N=N, sim_time=SIM_TIME,
                          multi_target_system=multi_target,
                          missile_configs=missile_configs)

    sim.radars = radar_configs

    live_feed = None
    if LIVE_MODE:
        print(f"[LIVE MODE] Enabling {LIVE_FEED_TYPE.upper()} feed...")

        if LIVE_FEED_TYPE == 'network':
            live_feed = create_live_feed('network', host='localhost', port=LIVE_DATA_PORT)
        elif LIVE_FEED_TYPE == 'file':
            live_feed = create_live_feed('file', filepath=LIVE_LOG_FILE, realtime=True)
        elif LIVE_FEED_TYPE == 'udp':
            live_feed = create_live_feed('udp', host='localhost', port=LIVE_UDP_PORT)
        else:
            print(f"[WARNING] Unknown live feed type: {LIVE_FEED_TYPE}")

        if live_feed:
            live_feed.start()

    sim.run_simulation(live_feed=live_feed)
    sim.animate(interval=50, save_video=SAVE_VIDEO, output_filename=OUTPUT_FILE, fps=VIDEO_FPS)

    if live_feed:
        live_feed.stop()

if __name__ == "__main__":
    main()
