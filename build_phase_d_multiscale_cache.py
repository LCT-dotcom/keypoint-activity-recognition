from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from phase_d_multiscale import COMPACT_SCHEMA_VERSION, MULTISCALE_WINDOWS, extract_compact_windows


BASE_STRIDE = 15
SUBJECTS = (1, 2, 3, 5)


def base_cache_path(cache_dir: Path, subject_id: int, window_size: int) -> Path:
    return cache_dir / (
        f"compact_{COMPACT_SCHEMA_VERSION}_subject_{subject_id}"
        f"_w{window_size}_s{BASE_STRIDE}_unlabeled.joblib"
    )


def source_signature(csv_path: Path, subject_id: int, window_size: int) -> dict:
    stat = csv_path.stat()
    return {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "subject_id": subject_id,
        "window_size": window_size,
        "stride": BASE_STRIDE,
        "labeled": False,
        "source_path": str(csv_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def load_or_build_base_cache(
    csv_path: Path,
    subject_id: int,
    window_size: int,
    cache_dir: Path,
    force: bool = False,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = base_cache_path(cache_dir, subject_id, window_size)
    signature = source_signature(csv_path, subject_id, window_size)
    if destination.exists() and not force:
        payload = joblib.load(destination)
        if payload.get("signature") == signature:
            return payload["windows"], True

    frame = pd.read_csv(csv_path)
    windows = extract_compact_windows(
        frame,
        subject_id=subject_id,
        window_size=window_size,
        stride=BASE_STRIDE,
        labeled=False,
    )
    windows.x = windows.x.astype("float32")
    payload = {
        "signature": signature,
        "windows": windows,
        "rows": len(windows.x),
        "features": len(windows.x.columns),
    }
    joblib.dump(payload, destination, compress=3)
    return windows, False


def run(data_dir: Path, cache_dir: Path, force: bool = False) -> list[dict]:
    records: list[dict] = []
    for subject in SUBJECTS:
        csv_path = data_dir / f"keypoints_with_labels_{subject}.csv"
        for window_size in MULTISCALE_WINDOWS:
            windows, cache_hit = load_or_build_base_cache(
                csv_path, subject, window_size, cache_dir, force=force
            )
            record = {
                "subject": subject,
                "window_size": window_size,
                "rows": len(windows.x),
                "features": len(windows.x.columns),
                "cache_hit": cache_hit,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    (cache_dir / "compact_cache_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resumable unlabeled Phase D compact caches.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/phase_d/compact_cache")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.cache_dir, force=args.force)
