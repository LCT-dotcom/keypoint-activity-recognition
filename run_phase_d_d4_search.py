from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_phase_d_tcn_cache import SUBJECTS, tcn_cache_path
from phase_d_evaluation import aggregate_window_probabilities
from phase_d_tcn import TCN_CANDIDATES, concatenate_tcn_windows, train_tcn
from tsfel_histgb_pipeline import CLASSES, clean_labels, evaluate_predictions


D4_SEARCH_VERSION = "phase-d4-tcn-search-v1"


def checkpoint_path(
    output_dir: Path,
    outer_subject: int,
    validation_subject: int,
    candidate_index: int,
) -> Path:
    return output_dir / (
        f"model_outer_{outer_subject}_validation_{validation_subject}"
        f"_candidate_{candidate_index:02d}.joblib"
    )


def run(
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    random_state: int = 42,
    max_epochs: int = 20,
    patience: int = 4,
    outer_subjects: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = {subject: joblib.load(tcn_cache_path(cache_dir, subject)) for subject in SUBJECTS}
    truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    selected_outers = outer_subjects or SUBJECTS
    if not selected_outers or any(subject not in SUBJECTS for subject in selected_outers):
        raise ValueError("D4 outer filter contains an unknown subject")
    rows: list[dict] = []
    audit = {"model_fits": 0, "cache_hits": 0, "completed_models": 0}
    for outer in selected_outers:
        outer_train = tuple(subject for subject in SUBJECTS if subject != outer)
        for validation_subject in outer_train:
            train_subjects = tuple(
                subject for subject in outer_train if subject != validation_subject
            )
            train = concatenate_tcn_windows(
                [cached[subject]["training"] for subject in train_subjects]
            )
            validation_training = cached[validation_subject]["training"]
            validation_inference = cached[validation_subject]["inference"]
            for candidate_index, config in enumerate(TCN_CANDIDATES):
                signature = {
                    "version": D4_SEARCH_VERSION,
                    "outer_subject": outer,
                    "train_subjects": list(train_subjects),
                    "validation_subject": validation_subject,
                    "candidate_index": candidate_index,
                    "config": asdict(config),
                    "random_state": random_state,
                    "max_epochs": max_epochs,
                    "patience": patience,
                }
                destination = checkpoint_path(
                    output_dir, outer, validation_subject, candidate_index
                )
                payload = None
                if destination.exists():
                    try:
                        candidate_payload = joblib.load(destination)
                        if candidate_payload.get("signature") == signature:
                            payload = candidate_payload
                    except (EOFError, OSError, ValueError):
                        payload = None
                cache_hit = payload is not None
                if payload is None:
                    fitted = train_tcn(
                        train,
                        validation_training,
                        config,
                        CLASSES,
                        random_state=(
                            random_state
                            + outer * 1000
                            + validation_subject * 100
                            + candidate_index
                        ),
                        max_epochs=max_epochs,
                        patience=patience,
                        batch_size=64,
                        device="cuda",
                    )
                    probabilities = fitted.predict_proba(
                        validation_inference, batch_size=128, device="cuda"
                    )
                    frame_result = aggregate_window_probabilities(
                        validation_inference.meta,
                        probabilities,
                        n_frames=len(truth[validation_subject]),
                        classes=CLASSES,
                    )
                    valid = np.asarray(
                        [label in set(CLASSES) for label in truth[validation_subject]],
                        dtype=bool,
                    )
                    metrics = evaluate_predictions(
                        truth[validation_subject][valid], frame_result.labels[valid]
                    )
                    row = {
                        "outer_held_out_subject": outer,
                        "train_subjects": "_".join(map(str, train_subjects)),
                        "validation_subject": validation_subject,
                        "candidate_index": candidate_index,
                        "best_epoch": fitted.best_epoch,
                        "validation_window_accuracy": fitted.validation_accuracy,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "abnormal_f1": metrics["abnormal_f1"],
                    }
                    payload = {"signature": signature, "model": fitted, "row": row}
                    temporary = destination.with_name(
                        f"{destination.name}.{os.getpid()}.tmp"
                    )
                    joblib.dump(payload, temporary, compress=3)
                    temporary.replace(destination)
                    audit["model_fits"] += 1
                else:
                    audit["cache_hits"] += 1
                rows.append(payload["row"])
                audit["completed_models"] += 1
                print(
                    json.dumps(
                        {
                            "outer": outer,
                            "validation": validation_subject,
                            "candidate": candidate_index,
                            "accuracy": payload["row"]["accuracy"],
                            "cache_hit": cache_hit,
                        }
                    ),
                    flush=True,
                )
    scores = pd.DataFrame(rows).sort_values(
        ["outer_held_out_subject", "candidate_index", "validation_subject"],
        kind="mergesort",
    ).reset_index(drop=True)
    suffix = "" if outer_subjects is None else "_" + "_".join(map(str, selected_outers))
    scores.to_csv(output_dir / f"d4_inner_scores{suffix}.csv", index=False)
    audit.update(
        {
            "version": D4_SEARCH_VERSION,
            "candidate_count": len(TCN_CANDIDATES),
            "expected_models": len(selected_outers) * 3 * len(TCN_CANDIDATES),
            "outer_subjects": list(selected_outers),
            "max_epochs": max_epochs,
            "patience": patience,
        }
    )
    (output_dir / f"d4_search_audit{suffix}.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return scores, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable nested D4 TCN search.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/phase_d/tcn_cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/phase_d/d4_search")
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--outers", help="Optional comma-separated outer subjects, for example 1,2."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, audit = run(
        args.data_dir,
        args.cache_dir,
        args.output_dir,
        args.random_state,
        args.max_epochs,
        args.patience,
        tuple(map(int, args.outers.split(","))) if args.outers else None,
    )
    print(json.dumps(audit, indent=2))
