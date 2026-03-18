import numpy as np

class System:
    def __init__(self, positions, velocities, masses):
        self.positions = np.array(positions)  # shape (N, 2) for N bodies in 2D
        self.velocities = np.array(velocities)  # shape (N, 2) for N bodies in 2D
        self.masses = np.array(masses)  # shape (N,) for N bodies

class Trajectory:
    def __init__(self, system, steps):
        self.positions = np.zeros((steps + 1, system.positions.shape[0], 2))
        self.velocities = np.zeros((steps + 1, system.velocities.shape[0], 2))
        self.positions[0] = system.positions
        self.velocities[0] = system.velocities
        self.masses = system.masses
    def __len__(self):
        return self.positions.shape[0]