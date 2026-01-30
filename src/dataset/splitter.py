"""Dataset split utilities for NCLT scans."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

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
    """Align scans with ground truth positions based on timestamps."""
    df = load_ground_truth_csv(ground_truth_csv)
    timestamp_col, x_col, y_col, z_col = _resolve_pose_columns(df.columns)
    poses_by_time = {
        int(row[timestamp_col]): np.array([row[x_col], row[y_col], row[z_col]], dtype=float)
        for _, row in df.iterrows()
    }

    matched_scans: List[Path] = []
    matched_poses: List[np.ndarray] = []
    for scan_file in scan_files:
        timestamp = timestamp_from_path(scan_file)
        pose = poses_by_time.get(timestamp)
        if pose is None:
            continue
        matched_scans.append(scan_file)
        matched_poses.append(pose)

    return matched_scans, np.vstack(matched_poses) if matched_poses else np.zeros((0, 3))

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
