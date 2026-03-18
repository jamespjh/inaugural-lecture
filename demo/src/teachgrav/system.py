import numpy as np


class System:
    def __init__(self, positions, velocities, masses, immobile=None):
        self.positions = np.array(positions)  # shape (N, 2) for N bodies in 2D
        # shape (N, 2) for N bodies in 2D
        self.velocities = np.array(velocities)
        self.masses = np.array(masses)  # shape (N,) for N bodies
        self.immobile = (np.array(
            immobile) if immobile is not None
            else np.zeros_like(masses, dtype=bool))

    def update(self, new_positions, new_velocities):
        """Return a new System with updated positions and velocities."""
        return System(new_positions, new_velocities,
                      self.masses, self.immobile)


class Trajectory:
    def __init__(self, system, steps):
        self.positions = np.zeros((steps + 1, system.positions.shape[0], 2))
        self.velocities = np.zeros((steps + 1, system.velocities.shape[0], 2))
        self.positions[0] = system.positions
        self.velocities[0] = system.velocities
        self.masses = system.masses
        self.immobile = system.immobile

    def __len__(self):
        return self.positions.shape[0]

    def write(self, stream, format='csv'):
        """Write the trajectory data to a stream in the specified format."""
        if format == 'csv':
            np.savetxt(stream, fmt='%10.5f', X=self.positions.reshape(
                len(self), -1), delimiter=',')
        else:
            # TODO:: Write to HDF5
            raise ValueError(f"Unsupported format: {format}")
