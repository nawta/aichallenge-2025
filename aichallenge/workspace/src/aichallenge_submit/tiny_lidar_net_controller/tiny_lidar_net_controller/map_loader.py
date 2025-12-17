"""Map loader module for lane boundary data.

This module provides functions to load and parse lane boundary data from
CSV files generated from Lanelet2 maps.
"""

import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LaneBoundaries:
    """Container for lane boundary data.
    
    Attributes:
        boundaries: Dict mapping lanelet_id to {'left': [...], 'right': [...]}
        all_points: Flattened array of all boundary points for fast lookup
        offset: (x, y) offset applied to normalize coordinates
    """
    boundaries: Dict[int, Dict[str, List[Tuple[float, float]]]]
    all_points: np.ndarray
    offset: Tuple[float, float]


def load_lane_boundaries(
    csv_path: str,
    auto_offset: bool = True,
    manual_offset: Optional[Tuple[float, float]] = None
) -> LaneBoundaries:
    """Load lane boundaries from a CSV file.
    
    Args:
        csv_path: Path to the lane.csv file.
        auto_offset: If True, automatically compute offset from first point.
        manual_offset: Manual (x, y) offset to apply. Overrides auto_offset.
    
    Returns:
        LaneBoundaries object containing parsed boundary data.
    
    CSV Format Expected:
        lanelet_id,way_id,boundary_type,node_id,sequence_order,local_x,local_y,elevation,latitude,longitude
    """
    # First pass: collect all points grouped by (lanelet_id, way_id, boundary_type)
    raw_data: Dict[Tuple[int, int, str], List[Tuple[int, float, float]]] = {}
    all_points_list = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lanelet_id = int(row['lanelet_id'])
            way_id = int(row['way_id'])
            boundary_type = row['boundary_type']  # 'left' or 'right'
            sequence_order = int(row['sequence_order'])
            local_x = float(row['local_x'])
            local_y = float(row['local_y'])
            
            key = (lanelet_id, way_id, boundary_type)
            if key not in raw_data:
                raw_data[key] = []
            raw_data[key].append((sequence_order, local_x, local_y))
            all_points_list.append((local_x, local_y))
    
    if not all_points_list:
        raise ValueError(f"No data found in CSV file: {csv_path}")
    
    # Compute offset
    all_points_array = np.array(all_points_list)
    if manual_offset is not None:
        offset = manual_offset
    elif auto_offset:
        # Use the first point as offset (similar to laserscan_generator)
        offset = (all_points_list[0][0], all_points_list[0][1])
    else:
        offset = (0.0, 0.0)
    
    # Apply offset to all points
    all_points_normalized = all_points_array - np.array(offset)
    
    # Second pass: organize by lanelet_id with sorted points and applied offset
    boundaries: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    
    for (lanelet_id, way_id, boundary_type), points in raw_data.items():
        # Sort by sequence_order
        sorted_points = sorted(points, key=lambda x: x[0])
        
        # Apply offset and extract (x, y)
        normalized_points = [
            (x - offset[0], y - offset[1]) 
            for _, x, y in sorted_points
        ]
        
        if lanelet_id not in boundaries:
            boundaries[lanelet_id] = {'left': [], 'right': []}
        
        # Extend existing points (a lanelet may have multiple ways for same boundary)
        boundaries[lanelet_id][boundary_type].extend(normalized_points)
    
    return LaneBoundaries(
        boundaries=boundaries,
        all_points=all_points_normalized,
        offset=offset
    )


def get_nearby_boundaries(
    lane_data: LaneBoundaries,
    ego_x: float,
    ego_y: float,
    radius: float = 50.0
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """Get lane boundaries within a radius of the ego position.
    
    Args:
        lane_data: LaneBoundaries object from load_lane_boundaries.
        ego_x: Ego vehicle x position (offset-normalized).
        ego_y: Ego vehicle y position (offset-normalized).
        radius: Search radius in meters.
    
    Returns:
        Dict of lanelet_id -> {'left': [...], 'right': [...]} for nearby lanelets.
    """
    nearby = {}
    radius_sq = radius * radius
    
    for lanelet_id, bounds in lane_data.boundaries.items():
        # Check if any point of this lanelet is within radius
        is_nearby = False
        for side in ['left', 'right']:
            for px, py in bounds[side]:
                dist_sq = (px - ego_x) ** 2 + (py - ego_y) ** 2
                if dist_sq <= radius_sq:
                    is_nearby = True
                    break
            if is_nearby:
                break
        
        if is_nearby:
            nearby[lanelet_id] = bounds
    
    return nearby


def get_all_boundary_segments(
    lane_data: LaneBoundaries
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], 
           List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Extract all boundary line segments.
    
    Args:
        lane_data: LaneBoundaries object.
    
    Returns:
        Tuple of (left_segments, right_segments), where each segment is
        ((x1, y1), (x2, y2)).
    """
    left_segments = []
    right_segments = []
    
    for lanelet_id, bounds in lane_data.boundaries.items():
        for i in range(len(bounds['left']) - 1):
            p1 = bounds['left'][i]
            p2 = bounds['left'][i + 1]
            left_segments.append((p1, p2))
        
        for i in range(len(bounds['right']) - 1):
            p1 = bounds['right'][i]
            p2 = bounds['right'][i + 1]
            right_segments.append((p1, p2))
    
    return left_segments, right_segments


def compute_map_offset_from_csv(csv_path: str) -> Tuple[float, float]:
    """Compute the map offset from a CSV file (first point).
    
    This is useful for aligning the map coordinate system with the
    localization coordinate system.
    
    Args:
        csv_path: Path to the lane.csv file.
    
    Returns:
        Tuple (offset_x, offset_y).
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        return (float(first_row['local_x']), float(first_row['local_y']))
