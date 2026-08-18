from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from phase_d_tcn import extract_tcn_windows


SUBJECTS = (1, 2, 3, 5)
TCN_CACHE_VERSION = "phase-d-tcn-v1"


def tcn_cache_path(cache_dir: Path, subject: int) -> Path:
    return cache_dir / f"{TCN_CACHE_VERSION}_subject_{subject}.joblib"


def run(data_dir: Path, cache_dir: Path, force: bool = False) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for subject in SUBJECTS:
        csv_path = data_dir / f"keypoints_with_labels_{subject}.csv"
        stat = csv_path.stat()
        signature = {
            "version": TCN_CACHE_VERSION,
            "subject": subject,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "training_window": 150,
            "training_stride": 75,
            "training_majority_threshold": 0.70,
            "inference_stride": 30,
        }
        destination = tcn_cache_path(cache_dir, subject)
        payload = None
        if destination.exists() and not force:
            try:
                candidate = joblib.load(destination)
                if candidate.get("signature") == signature:
                    payload = candidate
            except (EOFError, OSError, ValueError):
                payload = None
        cache_hit = payload is not None
        if payload is None:
            frame = pd.read_csv(csv_path)
            training = extract_tcn_windows(
                frame,
                subject_id=subject,
                window_size=150,
                stride=75,
                majority_threshold=0.70,
                labeled=True,
            )
            inference = extract_tcn_windows(
                frame,
                subject_id=subject,
                window_size=150,
                stride=30,
                labeled=False,
            )
            payload = {"signature": signature, "training": training, "inference": inference}
            temporary = destination.with_suffix(".joblib.tmp")
            joblib.dump(payload, temporary, compress=3)
            temporary.replace(destination)
        record = {
            "subject": subject,
            "training_windows": len(payload["training"].labels),
            "inference_windows": len(payload["inference"].labels),
            "cache_hit": cache_hit,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (cache_dir / "tcn_cache_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resumable Phase D TCN caches.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/phase_d/tcn_cache"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.cache_dir, force=args.force)
