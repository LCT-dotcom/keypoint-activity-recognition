from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_phase_d_tcn_cache import SUBJECTS, tcn_cache_path
from phase_d_evaluation import aggregate_window_probabilities
from phase_d_runner import CandidateResult, _build_outer_result, select_phase_d_winner
from phase_d_tcn import TCN_CANDIDATES, concatenate_tcn_windows, train_tcn_fixed_epochs
from run_phase_d_d1 import _candidate_from_artifacts, frame_prediction_table
from tsfel_histgb_pipeline import CLASSES, clean_labels, evaluate_predictions


def select_outer_config(scores: pd.DataFrame, outer: int) -> tuple[int, int, pd.DataFrame]:
    inner = scores[scores["outer_held_out_subject"] == outer].copy()
    if len(inner) != 3 * len(TCN_CANDIDATES):
        raise ValueError(f"Outer {outer} has an incomplete D4 inner score table")
    summary = inner.groupby("candidate_index", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_abnormal_f1=("abnormal_f1", "mean"),
    )
    best_index = int(
        summary.sort_values(
            ["mean_accuracy", "worst_accuracy", "mean_macro_f1", "mean_abnormal_f1"],
            ascending=False,
            kind="mergesort",
        ).iloc[0]["candidate_index"]
    )
    selected_epochs = inner[inner["candidate_index"] == best_index]["best_epoch"] + 1
    epochs = max(1, int(round(float(selected_epochs.median()))))
    return best_index, epochs, inner


def run(data_dir: Path, cache_dir: Path, phase_dir: Path, random_state: int = 42) -> dict:
    scores = pd.read_csv(phase_dir / "d4_search" / "d4_inner_scores.csv")
    cached = {subject: joblib.load(tcn_cache_path(cache_dir, subject)) for subject in SUBJECTS}
    truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    audits: dict[str, dict] = {}
    for outer in SUBJECTS:
        candidate_index, epochs, inner = select_outer_config(scores, outer)
        config = TCN_CANDIDATES[candidate_index]
        train_subjects = tuple(subject for subject in SUBJECTS if subject != outer)
        train = concatenate_tcn_windows(
            [cached[subject]["training"] for subject in train_subjects]
        )
        fitted = train_tcn_fixed_epochs(
            train,
            config,
            CLASSES,
            epochs=epochs,
            random_state=random_state + outer,
            batch_size=64,
            device="cuda",
        )
        test = cached[outer]["inference"]
        probabilities = fitted.predict_proba(test, batch_size=128, device="cuda")
        frame_result = aggregate_window_probabilities(
            test.meta,
            probabilities,
            n_frames=len(truth[outer]),
            classes=CLASSES,
        )
        inner_splits = [
            {
                "train_subjects": [
                    subject
                    for subject in train_subjects
                    if subject != int(validation)
                ],
                "validation_subject": int(validation),
            }
            for validation in sorted(inner["validation_subject"].unique())
        ]
        audit = {
            "outer_held_out_subject": outer,
            "outer_train_subjects": list(train_subjects),
            "inner_splits": inner_splits,
            "candidate": "D4-shallow-multistream-TCN",
            "selected_candidate_index": candidate_index,
            "selected_config": asdict(config),
            "fixed_outer_epochs": epochs,
            "epoch_rule": "median(best_inner_epoch_plus_one)",
        }
        fold = _build_outer_result(
            "D4-shallow-multistream-TCN",
            outer,
            frame_result,
            truth[outer],
            audit,
            fitted,
        )
        joblib.dump(
            {"model": fitted, "audit": audit},
            phase_dir / f"d4_model_outer_subject_{outer}.joblib",
            compress=3,
        )
        rows.append({"held_out_subject": outer, **fold.metrics})
        pooled_true.append(fold.y_true)
        pooled_pred.append(fold.y_pred)
        audits[str(outer)] = audit
        fold.confusion.to_csv(phase_dir / f"d4_confusion_subject_{outer}.csv")
        fold.classification_report.to_csv(
            phase_dir / f"d4_classification_report_subject_{outer}.csv"
        )
        frame_prediction_table(frame_result).to_csv(
            phase_dir / f"d4_frame_predictions_subject_{outer}.csv", index=False
        )
        np.savez_compressed(
            phase_dir / f"d4_frame_probabilities_subject_{outer}.npz",
            probabilities=frame_result.probabilities,
            coverage=frame_result.coverage,
        )
        (phase_dir / f"d4_audit_subject_{outer}.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(phase_dir / "d4_fold_metrics.csv", index=False)
    pooled = evaluate_predictions(np.concatenate(pooled_true), np.concatenate(pooled_pred))
    summary = {
        "candidate": "D4-shallow-multistream-TCN",
        "pooled_accuracy": pooled["accuracy"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_abnormal_f1": pooled["abnormal_f1"],
        "worst_subject_accuracy": float(fold_metrics["accuracy"].min()),
        "worst_subject": int(
            fold_metrics.loc[fold_metrics["accuracy"].idxmin(), "held_out_subject"]
        ),
        "outer_audits": audits,
    }
    (phase_dir / "d4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    contenders: list[CandidateResult] = []
    for name, prefix, baseline in (
        ("D0-C", "d0", True),
        ("D1-regularized-C", "d1", False),
        ("D2-multiscale-geometry", "d2", False),
        ("D3-D1-soft-gate-viterbi", "d3", False),
        ("D4-shallow-multistream-TCN", "d4", False),
    ):
        contenders.append(
            _candidate_from_artifacts(
                name,
                pd.read_csv(phase_dir / f"{prefix}_fold_metrics.csv"),
                json.loads((phase_dir / f"{prefix}_summary.json").read_text(encoding="utf-8")),
                baseline,
            )
        )
    winner = select_phase_d_winner(contenders)
    comparison = {
        "locked_winner": winner.name,
        "candidates": {
            candidate.name: {
                "eligible": candidate.eligible,
                "eligibility_reasons": candidate.eligibility_reasons,
                "pooled_accuracy": candidate.pooled_accuracy,
                "pooled_macro_f1": candidate.pooled_macro_f1,
                "pooled_abnormal_f1": candidate.pooled_abnormal_f1,
                "worst_subject_accuracy": candidate.worst_subject_accuracy,
            }
            for candidate in contenders
        },
    }
    (phase_dir / "phase_d_final_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected D4 outer LOSO evaluation.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/phase_d/tcn_cache"))
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.data_dir, args.cache_dir, args.phase_dir, args.random_state), indent=2))
