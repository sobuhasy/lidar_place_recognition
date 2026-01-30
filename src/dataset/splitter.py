"""Dataset split utilities for NCLT scans."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.dataset.loader import get_scan_files, load_ground_truth_csv

def timestamp_from_path(path: Path) -> int:
    return int(path.stem)

def _resolve_pose_columns(columns: Iterable[str]) -> Tuple[str, str, str, str]:
    columns = list(columns)
    timestamp_candidates = ["timestamp", "time", "t"]
    timestamp_col = next((col for col in timestamp_candidates if col in columns), columns[0])
    if all(name in columns for name in ("x", "y", "z")):
        return timestamp_col, "x", "y", "z"
    if len(columns) >= 4:
        return timestamp_col, columns[1], columns[2], columns[3]
    raise ValueError("Ground truth CSV must contain timestamp and x, y, z columns.")

def align_scans_with_groundtruth(
        scan_files: List[Path],
        ground_truth_csv: str,
) -> Tuple[List[Path], np.ndarray]:
    """Align scans with ground truth using Nearest Neighbor search (Fast)."""
    
    print(f"Loading Ground Truth from {ground_truth_csv}...")
    # Read CSV efficiently
    df = pd.read_csv(ground_truth_csv)
    
    # Identify columns
    cols = [c.lower() for c in df.columns]
    # NCLT usually has 'utime', 'time', or 'timestamp'
    # We look for the first column that looks like time
    if 'utime' in cols:
        t_col = df.columns[cols.index('utime')]
    elif 'time' in cols:
        t_col = df.columns[cols.index('time')]
    else:
        t_col = df.columns[0] # Fallback to first column
        
    # Ensure sorted by time
    df = df.sort_values(by=t_col)
    
    # Get numpy arrays (Fast access)
    gt_times = df[t_col].values
    # Try to find x, y, z columns
    try:
        # NCLT CSV usually has explicit x,y,z headers
        gt_poses = df[['x', 'y', 'z']].values
    except KeyError:
        # Fallback: assume columns 1, 2, 3 are x, y, z
        gt_poses = df.iloc[:, 1:4].values

    print(f"Aligning {len(scan_files)} scans to {len(gt_times)} GT poses...")
    
    scan_times = np.array([int(p.stem) for p in scan_files])
    
    # --- MAGIC STEP: SEARCHSORTED (Binary Search) ---
    # Find the index of the closest GT timestamp for each scan
    # This is Instant (O(log N)) vs Iterrows (O(N))
    idxs = np.searchsorted(gt_times, scan_times)
    idxs = np.clip(idxs, 0, len(gt_times) - 1)
    
    # Refine: searchsorted finds the insertion point (right side)
    # Check if the left neighbor is actually closer
    left_idxs = np.clip(idxs - 1, 0, len(gt_times) - 1)
    dist_right = np.abs(gt_times[idxs] - scan_times)
    dist_left = np.abs(gt_times[left_idxs] - scan_times)
    
    # Pick the better neighbor
    use_left = dist_left < dist_right
    final_idxs = np.where(use_left, left_idxs, idxs)
    
    # FILTER: If the nearest match is too far (e.g., > 100ms), ignore it
    min_dists = np.minimum(dist_left, dist_right)
    TOLERANCE_US = 100000 # 100ms tolerance
    valid_mask = min_dists < TOLERANCE_US
    
    matched_scans = np.array(scan_files)[valid_mask].tolist()
    matched_poses = gt_poses[final_idxs][valid_mask]
    
    print(f"Success: Matched {len(matched_scans)} scans.")
    
    if len(matched_scans) == 0:
        raise ValueError("No matches found! Check CSV timestamps vs Filenames.")
        
    return matched_scans, matched_poses

def split_database_query(
        scan_files: List[Path],
        poses: np.ndarray,
        db_stride: int = 10,
        query_stride: int = 1,
        query_offset: int | None = None,
) -> Tuple[List[Path], List[Path], np.ndarray, np.ndarray]:
    """Split scans into database/query lists with disjoint indices."""
    if db_stride <= 0 or query_stride <= 0:
        raise ValueError("Stride values must be positive.")
    if len(scan_files) != len(poses):
        raise ValueError("scan_files and poses must have the same length.")
    
    if query_offset is None:
        query_offset = db_stride // 2

    database_indices = set(range(0, len(scan_files), db_stride))
    query_indices = [
        idx
        for idx in range(query_offset, len(scan_files), query_stride)
        if idx not in database_indices
    ]

    database_scans = [scan_files[idx] for idx in sorted(database_indices)]
    query_scans = [scan_files[idx] for idx in query_indices]
    database_poses = poses[sorted(database_indices)]
    query_poses = poses[query_indices]
    return database_scans, query_scans, database_poses, query_poses

def load_nclt_dataset(
        velodyne_dir: str,
        ground_truth_csv: str,
        db_stride: int = 10,
        query_stride: int = 1,
        query_offset: int | None = None,
        max_scans: int | None = None,
) -> Tuple[List[Path], List[Path], np.ndarray, np.ndarray]:
    """Load NCLT scans and split into database/query subsets."""
    scan_files = get_scan_files(velodyne_dir)
    if max_scans is not None:
        scan_files = scan_files[:max_scans]

    matched_scans, poses = align_scans_with_groundtruth(scan_files, ground_truth_csv)
    return split_database_query(
        matched_scans,
        poses,
        db_stride=db_stride,
        query_stride=query_stride,
        query_offset=query_offset,
    )
