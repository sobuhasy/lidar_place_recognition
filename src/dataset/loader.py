"""Dataset loading utilities for NCLT."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def load_scan(file_path: str) -> np.ndarray:
    """Load a single NCLT LiDAR scan from a binary file.
    
    Format: x, y, z, intensity (float32)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {file_path}")
    
    # Read binary data
    scan = np.fromfile(path, dtype=np.float32)
    
    # Reshape to (N, 4) -> [x, y, z, intensity]
    return scan.reshape((-1, 4))

def load_ground_truth_csv(csv_path: str) -> pd.DataFrame:
    """Load the NCLT ground truth CSV."""
    # NCLT GT format: timestamp, x, y, z, roll, pitch, yaw
    # We explicitly name columns because sometimes headers are messy
    df = pd.read_csv(csv_path)
    return df

def get_scan_files(velodyne_dir: str) -> List[Path]:
    """Get all .bin files sorted by timestamp (filename)."""
    p = Path(velodyne_dir)
    # NCLT filenames are timestamps (e.g., 1326031000123456.bin)
    files = sorted(list(p.glob("*.bin")))
    return files
