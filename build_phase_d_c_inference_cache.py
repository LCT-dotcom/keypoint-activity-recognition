from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from abc_experiment_pipeline import extract_experiment_unlabeled_windows


def cache_path(cache_dir: Path, subject: int) -> Path:
    return cache_dir / f"c_unlabeled_stride75_subject_{subject}.joblib"


def run(
    data_dir: Path,
    s4_shared_path: Path,
    cache_dir: Path,
    force: bool = False,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for subject in (1, 2, 3, 4, 5):
        source = (
            s4_shared_path
            if subject == 4
            else data_dir / f"keypoints_with_labels_{subject}.csv"
        )
        stat = source.stat()
        signature = {
            "source_path": str(source.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "subject": subject,
            "experiment": "C",
            "window_size": 150,
            "stride": 75,
            "labeled": False,
        }
        destination = cache_path(cache_dir, subject)
        payload = None
        if destination.exists() and not force:
            try:
                candidate = joblib.load(destination)
                if candidate.get("signature") == signature:
                    payload = candidate
            except (EOFError, OSError, ValueError):
                payload = None
        hit = payload is not None
        if payload is None:
            windows = extract_experiment_unlabeled_windows(
                pd.read_csv(source),
                subject_id=subject,
                experiment="C",
                window_size=150,
                stride=75,
            )
            payload = {"signature": signature, "windows": windows}
            temporary = destination.with_suffix(".joblib.tmp")
            joblib.dump(payload, temporary, compress=3)
            temporary.replace(destination)
        record = {"subject": subject, "windows": len(payload["windows"].x), "cache_hit": hit}
        records.append(record)
        print(json.dumps(record), flush=True)
    (cache_dir / "c_unlabeled_stride75_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label-independent C inference caches.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--s4-shared-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/phase_d/c_inference_cache"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.s4_shared_path, args.cache_dir, args.force)
