from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_phase_d_multiscale_cache import SUBJECTS, base_cache_path
from phase_d_classical import phase_d_parameter_candidates
from phase_d_d2 import (
    fit_d2_outer_models,
    prepare_d2_window_caches,
    reconstruct_inner_scale_probabilities,
    select_fusion_weight,
    select_scale_settings,
    top_d1_candidate_indices,
)
from phase_d_multiscale import MULTISCALE_WINDOWS
from phase_d_runner import CandidateResult, _build_outer_result, select_phase_d_winner
from phase_d_search import select_outer_configs_from_pair_scores
from run_phase_d_d1 import _candidate_from_artifacts, frame_prediction_table
from tsfel_histgb_pipeline import clean_labels, evaluate_predictions


def run(data_dir: Path, cache_dir: Path, phase_dir: Path, random_state: int = 42) -> dict:
    candidates = phase_d_parameter_candidates(random_state)
    d1_scores = pd.read_csv(phase_dir / "d1_pair_search" / "pair_search_scores.csv")
    d1_selections, _ = select_outer_configs_from_pair_scores(d1_scores, SUBJECTS, candidates)
    allowed = {
        subject: top_d1_candidate_indices(selection.inner_metrics, top_k=2)
        for subject, selection in d1_selections.items()
    }
    d2_search_dir = phase_dir / "d2_pair_search"
    d2_scores = pd.read_csv(d2_search_dir / "d2_pair_search_scores.csv")
    base_windows = {
        subject: {
            scale: joblib.load(base_cache_path(cache_dir, subject, scale))["windows"]
            for scale in MULTISCALE_WINDOWS
        }
        for subject in SUBJECTS
    }
    frame_truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    training, inference = prepare_d2_window_caches(base_windows, frame_truth)

    fold_rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    outer_audits: dict[str, dict] = {}
    for outer in SUBJECTS:
        scale_selections, selection_audit = select_scale_settings(
            d2_scores,
            SUBJECTS,
            outer_held_out_subject=outer,
            allowed_candidate_indices=allowed[outer],
        )
        inner_probabilities, inner_truth = reconstruct_inner_scale_probabilities(
            inference,
            frame_truth,
            scale_selections,
            selection_audit,
            d2_search_dir,
        )
        fusion_weights, fusion_metrics = select_fusion_weight(
            inner_probabilities, inner_truth
        )
        fitted_models, frame_result, outer_model_audit = fit_d2_outer_models(
            training,
            inference,
            frame_truth,
            candidates,
            outer,
            scale_selections,
            fusion_weights,
            random_state=random_state,
        )
        audit = dict(selection_audit)
        audit.update(outer_model_audit)
        audit.update(
            {
                "candidate": "D2-multiscale-geometry",
                "allowed_d1_candidate_indices": allowed[outer],
            }
        )
        fold = _build_outer_result(
            "D2-multiscale-geometry",
            outer,
            frame_result,
            frame_truth[outer],
            audit,
            fitted_models,
        )
        joblib.dump(
            {
                "models_by_scale": fitted_models,
                "scale_selections": scale_selections,
                "fusion_weights": fusion_weights,
                "classes": frame_result.classes,
                "audit": audit,
            },
            phase_dir / f"d2_model_outer_subject_{outer}.joblib",
            compress=3,
        )
        fold_rows.append({"held_out_subject": outer, **fold.metrics})
        pooled_true.append(fold.y_true)
        pooled_pred.append(fold.y_pred)
        outer_audits[str(outer)] = audit
        fold.confusion.to_csv(phase_dir / f"d2_confusion_subject_{outer}.csv")
        fold.classification_report.to_csv(
            phase_dir / f"d2_classification_report_subject_{outer}.csv"
        )
        frame_prediction_table(frame_result).to_csv(
            phase_dir / f"d2_frame_predictions_subject_{outer}.csv", index=False
        )
        np.savez_compressed(
            phase_dir / f"d2_frame_probabilities_subject_{outer}.npz",
            probabilities=frame_result.probabilities,
            coverage=frame_result.coverage,
        )
        fusion_metrics.to_csv(
            phase_dir / f"d2_inner_fusion_metrics_outer_subject_{outer}.csv", index=False
        )
        (phase_dir / f"d2_audit_subject_{outer}.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(phase_dir / "d2_fold_metrics.csv", index=False)
    pooled = evaluate_predictions(np.concatenate(pooled_true), np.concatenate(pooled_pred))
    summary = {
        "candidate": "D2-multiscale-geometry",
        "pooled_accuracy": pooled["accuracy"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_abnormal_f1": pooled["abnormal_f1"],
        "worst_subject_accuracy": float(fold_metrics["accuracy"].min()),
        "worst_subject": int(
            fold_metrics.loc[fold_metrics["accuracy"].idxmin(), "held_out_subject"]
        ),
        "outer_audits": outer_audits,
    }
    (phase_dir / "d2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    candidates_for_winner: list[CandidateResult] = []
    for name, prefix, baseline in (
        ("D0-C", "d0", True),
        ("D1-regularized-C", "d1", False),
        ("D2-multiscale-geometry", "d2", False),
    ):
        metrics = pd.read_csv(phase_dir / f"{prefix}_fold_metrics.csv")
        candidate_summary = json.loads(
            (phase_dir / f"{prefix}_summary.json").read_text(encoding="utf-8")
        )
        candidates_for_winner.append(
            _candidate_from_artifacts(name, metrics, candidate_summary, baseline)
        )
    winner = select_phase_d_winner(candidates_for_winner)
    comparison = {
        "current_winner": winner.name,
        "candidates": {
            candidate.name: {
                "eligible": candidate.eligible,
                "eligibility_reasons": candidate.eligibility_reasons,
                "pooled_accuracy": candidate.pooled_accuracy,
                "pooled_macro_f1": candidate.pooled_macro_f1,
                "pooled_abnormal_f1": candidate.pooled_abnormal_f1,
                "worst_subject_accuracy": candidate.worst_subject_accuracy,
            }
            for candidate in candidates_for_winner
        },
    }
    (phase_dir / "d0_d1_d2_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and evaluate D2 outer LOSO folds.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/phase_d/compact_cache")
    )
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.data_dir, args.cache_dir, args.phase_dir, args.random_state), indent=2))
