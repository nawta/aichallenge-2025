"""Bird's Eye View (BEV) grid generator for lane boundaries.

This module generates BEV representations of lane boundaries in two modes:
1. Local BEV: Vehicle-centered, rotated to vehicle heading (for local perception)
2. Global BEV: Map-fixed coordinates, vehicle position marked (for global planning)

Used by:
- lib/data.py: For training data preparation (BEV models)
- train.py: For BEV model training

Related files:
- lib/map_loader.py: Lane boundary CSV loading
- lib/model.py: TinyLidarNetLocalBEV, TinyLidarNetGlobalBEV, TinyLidarNetDualBEV
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .map_loader import LaneBoundaries


class BEVGenerator:
    """Generates Bird's Eye View grids from lane boundary data.

    Supports two BEV modes:
    - Local: Centered on ego vehicle, rotated to vehicle heading
    - Global: Fixed map coordinates, vehicle position marked as channel

    Attributes:
        local_grid_size: Size of local BEV grid (e.g., 64 for 64x64).
        local_resolution: Meters per pixel for local BEV.
        global_grid_size: Size of global BEV grid (e.g., 128 for 128x128).
        global_resolution: Meters per pixel for global BEV.
    """

    def __init__(
        self,
        local_grid_size: int = 64,
        local_resolution: float = 1.0,
        global_grid_size: int = 128,
        global_resolution: float = 1.5,
        local_channels: int = 2,
        global_channels: int = 3
    ):
        """Initialize BEV generator.

        Args:
            local_grid_size: Size of local BEV grid in pixels.
            local_resolution: Meters per pixel for local BEV.
            global_grid_size: Size of global BEV grid in pixels.
            global_resolution: Meters per pixel for global BEV.
            local_channels: Number of channels for local BEV (default 2: left/right).
            global_channels: Number of channels for global BEV (default 3: left/right/ego).
        """
        # Local BEV settings
        self.local_grid_size = local_grid_size
        self.local_resolution = local_resolution
        self.local_half_size = local_grid_size // 2
        self.local_coverage = local_grid_size * local_resolution
        self.local_channels = local_channels

        # Global BEV settings
        self.global_grid_size = global_grid_size
        self.global_resolution = global_resolution
        self.global_half_size = global_grid_size // 2
        self.global_coverage = global_grid_size * global_resolution
        self.global_channels = global_channels

        # Map center for global BEV (will be set on first call or manually)
        self._map_center: Optional[Tuple[float, float]] = None

    def set_map_center(self, center_x: float, center_y: float) -> None:
        """Set the center point for global BEV.

        Args:
            center_x: X coordinate of map center (in offset-normalized coords).
            center_y: Y coordinate of map center (in offset-normalized coords).
        """
        self._map_center = (center_x, center_y)

    def auto_compute_map_center(self, lane_data: LaneBoundaries) -> Tuple[float, float]:
        """Automatically compute map center from lane data.

        Args:
            lane_data: LaneBoundaries object with boundary data.

        Returns:
            (center_x, center_y) tuple.
        """
        if len(lane_data.all_points) == 0:
            return (0.0, 0.0)

        center_x = float(np.mean(lane_data.all_points[:, 0]))
        center_y = float(np.mean(lane_data.all_points[:, 1]))
        self._map_center = (center_x, center_y)
        return self._map_center

    def generate_local(
        self,
        lane_data: LaneBoundaries,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        map_offset: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Generate Local BEV grid centered on ego vehicle.

        Local BEV characteristics:
        - Centered on ego vehicle position
        - Rotated to align with vehicle heading (forward = up)
        - Channel 0: left boundaries, Channel 1: right boundaries

        Args:
            lane_data: LaneBoundaries object with boundary data.
            ego_x: Ego vehicle x position in world coordinates.
            ego_y: Ego vehicle y position in world coordinates.
            ego_yaw: Ego vehicle yaw angle in radians (0 = +x direction).
            map_offset: Optional offset already applied to lane_data.
                If None, uses lane_data.offset.

        Returns:
            np.ndarray of shape (local_channels, local_grid_size, local_grid_size).
            Channel 0: left boundaries, Channel 1: right boundaries.
            Values are 1.0 where boundary exists, 0.0 elsewhere.
        """
        grid = np.zeros(
            (self.local_channels, self.local_grid_size, self.local_grid_size),
            dtype=np.float32
        )

        # Use lane_data offset if not provided
        if map_offset is None:
            map_offset = lane_data.offset

        # Convert ego position to normalized coordinates
        ego_x_norm = ego_x - map_offset[0]
        ego_y_norm = ego_y - map_offset[1]

        # Precompute rotation values
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)

        # Process each lanelet's boundaries
        for lanelet_id, bounds in lane_data.boundaries.items():
            for channel_idx, side in enumerate(['left', 'right']):
                points = bounds.get(side, [])
                if len(points) < 2:
                    continue

                # Draw line segments
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i + 1]

                    # Transform to vehicle-centered coordinates
                    v1 = self._world_to_vehicle(
                        p1[0], p1[1], ego_x_norm, ego_y_norm, cos_yaw, sin_yaw
                    )
                    v2 = self._world_to_vehicle(
                        p2[0], p2[1], ego_x_norm, ego_y_norm, cos_yaw, sin_yaw
                    )

                    # Convert to grid coordinates
                    g1 = self._vehicle_to_local_grid(v1[0], v1[1])
                    g2 = self._vehicle_to_local_grid(v2[0], v2[1])

                    # Draw line using Bresenham's algorithm
                    self._draw_line(grid[channel_idx], g1[0], g1[1], g2[0], g2[1],
                                   self.local_grid_size)

        return grid

    def generate_global(
        self,
        lane_data: LaneBoundaries,
        ego_x: float,
        ego_y: float,
        map_offset: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Generate Global BEV grid in map-fixed coordinates.

        Global BEV characteristics:
        - Fixed map coordinates (no rotation)
        - Channel 0: left boundaries, Channel 1: right boundaries
        - Channel 2: ego vehicle position marker

        Args:
            lane_data: LaneBoundaries object with boundary data.
            ego_x: Ego vehicle x position in world coordinates.
            ego_y: Ego vehicle y position in world coordinates.
            map_offset: Optional offset already applied to lane_data.
                If None, uses lane_data.offset.

        Returns:
            np.ndarray of shape (global_channels, global_grid_size, global_grid_size).
            Channel 0: left boundaries, Channel 1: right boundaries.
            Channel 2: ego vehicle position (3x3 marker).
            Values are 1.0 where boundary/ego exists, 0.0 elsewhere.
        """
        grid = np.zeros(
            (self.global_channels, self.global_grid_size, self.global_grid_size),
            dtype=np.float32
        )

        # Use lane_data offset if not provided
        if map_offset is None:
            map_offset = lane_data.offset

        # Compute map center if not set
        if self._map_center is None:
            self.auto_compute_map_center(lane_data)

        map_center_x, map_center_y = self._map_center

        # Convert ego position to normalized coordinates
        ego_x_norm = ego_x - map_offset[0]
        ego_y_norm = ego_y - map_offset[1]

        # Draw lane boundaries (no rotation)
        for lanelet_id, bounds in lane_data.boundaries.items():
            for channel_idx, side in enumerate(['left', 'right']):
                points = bounds.get(side, [])
                if len(points) < 2:
                    continue

                # Draw line segments
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i + 1]

                    # Convert to global grid coordinates (no rotation)
                    g1 = self._world_to_global_grid(p1[0], p1[1], map_center_x, map_center_y)
                    g2 = self._world_to_global_grid(p2[0], p2[1], map_center_x, map_center_y)

                    # Draw line
                    self._draw_line(grid[channel_idx], g1[0], g1[1], g2[0], g2[1],
                                   self.global_grid_size)

        # Draw ego position marker on channel 2 (3x3 marker)
        ego_grid = self._world_to_global_grid(ego_x_norm, ego_y_norm, map_center_x, map_center_y)
        ego_row, ego_col = ego_grid

        # Draw 3x3 marker centered on ego position
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = ego_row + dr, ego_col + dc
                if 0 <= r < self.global_grid_size and 0 <= c < self.global_grid_size:
                    grid[2, r, c] = 1.0

        return grid

    def generate_both(
        self,
        lane_data: LaneBoundaries,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        map_offset: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate both Local and Global BEV grids.

        Args:
            lane_data: LaneBoundaries object with boundary data.
            ego_x: Ego vehicle x position in world coordinates.
            ego_y: Ego vehicle y position in world coordinates.
            ego_yaw: Ego vehicle yaw angle in radians.
            map_offset: Optional offset.

        Returns:
            Tuple of (local_bev, global_bev) numpy arrays.
        """
        local_bev = self.generate_local(lane_data, ego_x, ego_y, ego_yaw, map_offset)
        global_bev = self.generate_global(lane_data, ego_x, ego_y, map_offset)
        return local_bev, global_bev

    def _world_to_vehicle(
        self,
        wx: float,
        wy: float,
        ego_x: float,
        ego_y: float,
        cos_yaw: float,
        sin_yaw: float
    ) -> Tuple[float, float]:
        """Transform world coordinates to vehicle-centered coordinates.

        Vehicle coordinate system:
        - Origin at ego vehicle
        - X-axis points forward (in direction of ego_yaw)
        - Y-axis points left

        Args:
            wx, wy: World coordinates (offset-normalized).
            ego_x, ego_y: Ego position (offset-normalized).
            cos_yaw, sin_yaw: Precomputed cos(-ego_yaw), sin(-ego_yaw).

        Returns:
            (vx, vy) in vehicle coordinates.
        """
        # Translate to ego-centered
        dx = wx - ego_x
        dy = wy - ego_y

        # Rotate to align with vehicle heading
        vx = dx * cos_yaw - dy * sin_yaw
        vy = dx * sin_yaw + dy * cos_yaw

        return (vx, vy)

    def _vehicle_to_local_grid(self, vx: float, vy: float) -> Tuple[int, int]:
        """Convert vehicle coordinates to local grid indices.

        Grid layout:
        - Row 0 is at the top (rear of vehicle in visualization)
        - Column 0 is at the left
        - Center of grid is ego position
        - Vehicle forward direction maps to "up" in grid

        Args:
            vx: X in vehicle coords (positive = forward).
            vy: Y in vehicle coords (positive = left).

        Returns:
            (row, col) grid indices.
        """
        # Convert meters to pixels
        px = vx / self.local_resolution
        py = vy / self.local_resolution

        # Grid layout: row = -vx direction, col = vy direction
        row = self.local_half_size - int(px)  # Flip so forward is up
        col = self.local_half_size + int(py)  # Left is positive in Y

        return (row, col)

    def _world_to_global_grid(
        self,
        wx: float,
        wy: float,
        center_x: float,
        center_y: float
    ) -> Tuple[int, int]:
        """Convert world coordinates to global grid indices.

        Grid layout:
        - Fixed map orientation (no rotation)
        - Center of grid is map center
        - +X maps to right (column+), +Y maps to up (row-)

        Args:
            wx, wy: World coordinates (offset-normalized).
            center_x, center_y: Map center coordinates.

        Returns:
            (row, col) grid indices.
        """
        # Translate to map-center
        dx = wx - center_x
        dy = wy - center_y

        # Convert meters to pixels
        px = dx / self.global_resolution
        py = dy / self.global_resolution

        # Grid layout: row = -Y direction (up is negative row), col = +X direction
        row = self.global_half_size - int(py)
        col = self.global_half_size + int(px)

        return (row, col)

    def _draw_line(
        self,
        grid: np.ndarray,
        r0: int, c0: int,
        r1: int, c1: int,
        grid_size: int
    ) -> None:
        """Draw a line on the grid using Bresenham's algorithm.

        Args:
            grid: 2D array to draw on.
            r0, c0: Start point (row, col).
            r1, c1: End point (row, col).
            grid_size: Size of the grid for bounds checking.
        """
        # Clip to grid bounds for early termination check
        if (r0 < 0 and r1 < 0) or (r0 >= grid_size and r1 >= grid_size):
            return
        if (c0 < 0 and c1 < 0) or (c0 >= grid_size and c1 >= grid_size):
            return

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0

        # Maximum iterations to prevent infinite loops
        max_iter = max(dr, dc) + 1

        for _ in range(max_iter):
            # Set pixel if within bounds
            if 0 <= r < grid_size and 0 <= c < grid_size:
                grid[r, c] = 1.0

            # Check if reached end point
            if r == r1 and c == c1:
                break

            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def get_local_coverage_range(self) -> float:
        """Get the local BEV coverage range in meters from ego position.

        Returns:
            Maximum distance from ego that is covered by the local grid.
        """
        return self.local_half_size * self.local_resolution

    def get_global_coverage_range(self) -> float:
        """Get the global BEV coverage range in meters from map center.

        Returns:
            Maximum distance from map center that is covered by the global grid.
        """
        return self.global_half_size * self.global_resolution


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion to yaw angle.

    Args:
        qx, qy, qz, qw: Quaternion components.

    Returns:
        Yaw angle in radians.
    """
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw
