import numpy as np


class System:
    def __init__(self, data, masses, immobile=None):
        self.data = np.array(data)  # shape (2, N, D) for N bodies
        # in D dimensions
        # First slice is positions, second slice is velocities
        self.D = self.data.shape[2]  # Number of dimensions (e.g., 2 for 2D)
        self.masses = np.array(masses)  # shape (N,) for N bodies
        self.immobile = (np.array(
            immobile) if immobile is not None
            else np.zeros_like(masses, dtype=bool))

    def positions(self) -> np.ndarray:
        return self.data[0]

    def velocities(self) -> np.ndarray:
        return self.data[1]

    def update(self, data):
        """Return a new System with updated positions and velocities."""
        return System(data, self.masses, self.immobile)

    def update_flat(self, flat_data):
        """Return a new System from a flat state vector."""
        N = len(self.positions())
        data = flat_data.reshape((2, N, self.D))
        return self.update(data)

    def __sub__(self, other):
        """Change representing the difference between systems."""
        return Change(self.data - other.data)

    def __add__(self, change):
        """Return a new System by applying a change to this system."""
        return self.displace(change)

    def displace(self, other):
        """Return a new System displaced by the change from another system."""
        return self.update(self.data + other.data)

    def __len__(self):
        return self.data.shape[1]  # Number of bodies


class Change:
    """ Represents the change in positions and velocities for a system."""

    def __init__(self, data):
        self.data = np.array(data)  # shape (2, N, D) for N bodies
        # in D dimensions
        # First slice is position changes, second slice is velocity changes


class Trajectory:
    def __init__(self, system, steps):
        self.data = np.zeros((
            steps + 1,
            2,
            system.data.shape[1],
            system.D))  # shape (steps+1, 2, N, D)
        self.data[0] = system.data
        self.masses = system.masses
        self.immobile = system.immobile
        self.D = system.D

    def __len__(self):
        return self.data.shape[0]

    def positions(self):
        return self.data[:, 0]

    def velocities(self):
        return self.data[:, 1]

    def write(self, stream, format='csv'):
        """Write the trajectory data to a stream in the specified format."""
        if format == 'csv':
            np.savetxt(stream, fmt='%10.5f', X=self.data.reshape(
                len(self), -1), delimiter=',')
        else:
            # TODO:: Write to HDF5
            raise ValueError(f"Unsupported format: {format}")
