"""3D Radar simulation with sweep visualization"""
import numpy as np

class Radar:
    def __init__(self, x, y, z=0.0, sweep_freq=1.0, max_range=1500.0, 
                 noise_std=5.0, refresh_rate=0.1):
        self.x = x
        self.y = y
        self.z = z
        self.sweep_freq = sweep_freq
        self.max_range = max_range
        self.noise_std = noise_std
        self.refresh_rate = refresh_rate
        self.last_detection_time = -np.inf
        self.detected_position = None
        self.detected_velocity = None
        self.detected_acceleration = None
        self.detection_count = 0

    def detect(self, target_x, target_y, target_z, target_vx, target_vy, target_vz,
               target_ax, target_ay, target_az, current_time):
        if current_time - self.last_detection_time < self.refresh_rate:
            return None
        
        detected_x = target_x + np.random.normal(0, self.noise_std)
        detected_y = target_y + np.random.normal(0, self.noise_std)
        detected_z = target_z + np.random.normal(0, self.noise_std)
        self.detected_position = (detected_x, detected_y, detected_z)
        self.detected_velocity = (target_vx, target_vy, target_vz)
        self.detected_acceleration = (target_ax, target_ay, target_az)
        
        self.last_detection_time = current_time
        self.target_detected = True
        self.detection_count += 1
        
        return self.detected_position, self.detected_velocity, self.detected_acceleration

    def get_latest_estimate(self):
        return self.detected_position, self.detected_velocity, self.detected_acceleration
