"""Run evaluation end-to-end (TODO)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from eval.metrics import average_precision, pr_curve, precision_recall_f1, roc_curve
from eval.sweeps import run_threshold_sweep_from_scores
from src.config import PLOTS_DIR, RAW_DATA, RESULTS_DIR
from src.dataset.loader import load_scan
from src.dataset.preprocess import downsample, filter_points, normalize
from src.descriptors.your_descriptor import YourDescriptor
from src.dataset.splitter import load_nclt_dataset
from src.matching.similarity import cosine_similarity

def _compute_descriptors(scan_paths, descriptor: YourDescriptor):
    descriptors = []
    for scan_path in scan_paths:
        points = load_scan(str(scan_path))
        filtered = filter_points(points)
        sampled = downsample(filtered)
        normalized = normalize(sampled)
        descriptors.append(descriptor.compute(normalized))
    return np.vstack(descriptors)

def _score_query_database(
        query_descriptors,
        database_descriptors,
        query_poses,
        database_poses,
        distance_threshold: float,
        ):
    """Score query/database descriptor pairs using pose distance as ground truth."""
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive.")
    
    scores = []
    labels = []
    threshold_sq = distance_threshold ** 2
    for query_idx, query_descriptor in enumerate(query_descriptors):
        query_pose = query_poses[query_idx]
        diffs = database_poses - query_pose
        diffs_sq = np.sum(diffs ** 2, axis = 1)
        is_match = dists_sq <= threshold_sq
        for db_idx, db_descriptor in enumerate(database_descriptors):
            labels.append(int(is_match[db_idx]))
            scores.append(cosine_similarity(query_descriptor, db_descriptor))

    return np.asarray(labels), np.asarray(scores)

def main():
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Run place recognition evaluation.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--distance-threshold", type=int, default=200)
    parser.add_argument(
        "--velodyne-dir",
        type=str,
        default=str(RAW_DATA/ "velodyne-data" / "velodyne_sync")
    )
    parser.add_argument(
        "--groundtruth-csv",
        type=str,
        default=str(RAW_DATA / "groundtruth.csv")
    )
    parser.add_argument("--db-stride", type=int, default=10)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--query-offset", type=int, default=None)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    database_scans, query_scans, database_poses, query_poses = load_nclt_dataset(
        velodyne_dir=args.velodyne_dir,
        ground_truth_csv=args.groundtruth_csv,
        db_stride=args.db_stride,
        query_stride=args.query_stride,
        query_offset=args.query_offset,
        max_scans=args.max_scans,
    )
    descriptor = YourDescriptor()
    database_descriptors = _compute_descriptors(database_scans, descriptor)
    query_descriptors = _compute_descriptors(query_scans, descriptor)
    y_true, y_scores = _score_query_database(
        query_descriptors,
        database_descriptors,
        query_poses,
        database_poses,
        args.distance_threshold,
    )

    y_pred = (y_scores >= args.threshold).astype(int)

    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    ap = average_precision(y_true, y_scores)
    recalls, precisions, _ = pr_curve(y_true, y_scores)
    fprs, tprs, _ = roc_curve(y_true, y_scores)
    sweep_metrics = run_threshold_sweep_from_scores(y_true, y_scores)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "eval_metrics.json"
    metrics_payload = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "threshold_sweep": sweep_metrics,
        "num_pairs": int(y_true.size),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

# --- Plot 1: Precision-Recall Curve ---
    pr_path = PLOTS_DIR / "pr_curve.png"
    plt.figure()
    plt.plot(recalls, precisions, marker=".", label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    # --- Plot 2: ROC Curve ---
    roc_path = PLOTS_DIR / "roc_curve.png"  # <--- DEFINING THE MISSING VARIABLE
    plt.figure()
    plt.plot(fprs, tprs, marker=".", color="#FFE5B4", label="ROC")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved PR curve to {pr_path}")
    print(f"Saved ROC curve to {roc_path}")


if __name__ == "__main__":
    main()
