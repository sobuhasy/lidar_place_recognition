"""Generate database/query split lists from NCLT data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import RAW_DATA, SPLITS_DIR
from src.dataset.splitter import load_nclt_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Split NCLT scans into database/query lists.")
    parser.add_argument(
        "--velodyne-dir",
        type=str,
        default=str(RAW_DATA / "velodyne_data" / "velodyne_sync"),
    )
    parser.add_argument(
        "--groundtruth-csv",
        type=str,
        default=str(RAW_DATA / "groundtruth.csv"),
    )
    parser.add_argument("--db-stride", type=int, default=10)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--query-offset", type=int, default=None)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(SPLITS_DIR / "nclt_split.json"),
        help="Path to save the split JSON.",
    )
    args = parser.parse_args()

    database_scans, query_scans, database_poses, query_poses = load_nclt_dataset(
        velodyne_dir=args.velodyne_dir,
        ground_truth_csv=args.groundtruth_csv,
        db_stride=args.db_stride,
        query_stride=args.query_stride,
        query_offset=args.query_offset,
        max_scans=args.max_scans,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    payload = {
        "database_scans": [str(path) for path in database_scans],
        "query_scans": [str(path) for path in query_scans],
        "database_poses": database_poses.tolist(),
        "query_poses": query_poses.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved split to {output_path}")
    print(f"Database scans: {len(database_scans)}")
    print(f"Query scans: {len(query_scans)}")

    if __name__ == "__main__":
        main()