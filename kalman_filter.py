"""3D Kalman Filter for radar sensor fusion"""
import numpy as np

class KalmanFilter3D:
    def __init__(self, initial_pos, initial_vel=None, process_noise=None, measurement_noise=None):
        """
        State vector: [x, y, z, vx, vy, vz]
        """
        self.x = np.hstack((initial_pos, initial_vel if initial_vel is not None else np.zeros(3)))
        self.P = np.eye(6) * 100
        self.F = np.eye(6)
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        
        self.Q = np.eye(6) * (process_noise if process_noise is not None else 0.1)
        self.R = np.eye(3) * (measurement_noise if measurement_noise is not None else 10)
        
        self.predict_count = 0
        self.update_count = 0

    def predict(self, dt):
        self.F[:3, 3:] = np.eye(3) * dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.predict_count += 1

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(f"    [KF] Warning: Singular S matrix, skipping update")
            return
        
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.update_count += 1
