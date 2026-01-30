"""Custom descriptor implementation (Global geometric descriptor)."""
from __future__ import annotations

import numpy as np
from .base import Descriptor

class YourDescriptor(Descriptor):
    """Simplified scan context-style global descriptor."""

    def __init__(
            self,
            num_rings: int = 20,
            num_sectors: int = 60,
            max_radius: float = 80.0,
            normalize: bool = True,
    ):
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_radius = max_radius
        self.normalize = normalize

    def compute(self, points):
        """Compute descriptor for given points."""
        if points is None or len(points) == 0:
            raise ValueError("Points array is empty.")
        
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("Points array must be of shape (N, 3+).*")
        
        xyz = points[:, :3]
        xy = xyz[:, :2]
        radii = np.linalg.norm(xy, axis=1)
        if self.max_radius <= 0:
            raise ValueError("max_radius must be positive.")
        mask = radii <= self.max_radius
        xyz = xyz[mask]
        if xyz.size == 0:
            return np.zeros(self.num_rings * self.num_sectors, dtype=float)
        
        xy = xyz[:, :2]
        radii = np.linalg.norm(xy, axis=1)
        angles = np.arctan2(xy[:, 1], xy[:, 0])
        angles = np.mod(angles + 2 * np.pi, 2 * np.pi)

        ring_size = self.max_radius / self.max_radius
        sector_size = 2 * np.pi / self.num_sectors
        ring_indices = np.floor(radii / ring_size).astype(int)
        sector_indices = np.floor(angles / sector_size).astype(int)
        ring_indices = np.clip(ring_indices, 0, self.num_rings - 1)
        sector_indices = np.clip(sector_indices, 0, self.num_sectors - 1)

        descriptor = np.zeros((self.num_rings, self.num_sectors), dtype=float)
        np.maximum.at(descriptor, (ring_indices, sector_indices), xyz[:, 2])

        descriptor = descriptor.flatten()
        if self.normalize:
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
        return descriptor
