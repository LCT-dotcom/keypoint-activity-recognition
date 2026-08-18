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

from phase_d_classical import phase_d_parameter_candidates
from phase_d_evaluation import FrameProbabilityResult
from phase_d_runner import CandidateResult, run_d1_fold_with_selection, select_phase_d_winner
from phase_d_search import select_outer_configs_from_pair_scores
from tsfel_histgb_pipeline import clean_labels, evaluate_predictions


SUBJECTS = (1, 2, 3, 5)


def frame_prediction_table(result: FrameProbabilityResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": np.arange(len(result.labels)),
            "predicted_label": result.labels,
            "confidence": result.confidence,
            "coverage": result.coverage,
        }
    )


def _candidate_from_artifacts(name: str, fold_metrics: pd.DataFrame, summary: dict, baseline: bool) -> CandidateResult:
    def metric(key: str) -> float:
        pooled_key = f"pooled_{key}"
        if pooled_key in summary:
            return float(summary[pooled_key])
        return float(summary[key])

    return CandidateResult(
        name=name,
        fold_metrics=fold_metrics,
        pooled_accuracy=metric("accuracy"),
        pooled_macro_f1=metric("macro_f1"),
        pooled_abnormal_f1=metric("abnormal_f1"),
        is_baseline=baseline,
    )


def compare_d0_d1_from_artifacts(output_dir: Path) -> dict:
    d0_metrics = pd.read_csv(output_dir / "d0_fold_metrics.csv")
    d0_summary = json.loads((output_dir / "d0_summary.json").read_text(encoding="utf-8"))
    d1_metrics = pd.read_csv(output_dir / "d1_fold_metrics.csv")
    d1_summary = json.loads((output_dir / "d1_summary.json").read_text(encoding="utf-8"))
    d0 = _candidate_from_artifacts("D0-C", d0_metrics, d0_summary, True)
    d1 = _candidate_from_artifacts("D1-regularized-C", d1_metrics, d1_summary, False)
    winner = select_phase_d_winner([d0, d1])
    comparison = {
        "current_winner": winner.name,
        "d1_eligible": d1.eligible,
        "d1_eligibility_reasons": d1.eligibility_reasons,
        "accuracy_change": d1.pooled_accuracy - d0.pooled_accuracy,
        "macro_f1_change": d1.pooled_macro_f1 - d0.pooled_macro_f1,
        "abnormal_f1_change": d1.pooled_abnormal_f1 - d0.pooled_abnormal_f1,
        "worst_subject_accuracy_change": d1.worst_subject_accuracy - d0.worst_subject_accuracy,
    }
    (output_dir / "d0_d1_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return comparison


def run(data_dir: Path, cache_dir: Path, output_dir: Path, random_state: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = {
        subject: joblib.load(cache_dir / f"abc_c_subject_{subject}_windows.joblib")["windows"]
        for subject in SUBJECTS
    }
    truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    candidates = phase_d_parameter_candidates(random_state)
    pair_scores = pd.read_csv(output_dir / "d1_pair_search" / "pair_search_scores.csv")
    selections, selection_audit = select_outer_configs_from_pair_scores(
        pair_scores, SUBJECTS, candidates
    )

    fold_rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    selected_configs: dict[str, dict] = {}
    for subject in SUBJECTS:
        selection = selections[subject]
        fold = run_d1_fold_with_selection(
            windows,
            truth,
            held_out_subject=subject,
            config=selection.config,
            selection_audit=selection.audit,
            random_state=random_state,
        )
        fold_rows.append({"held_out_subject": subject, **fold.metrics})
        pooled_true.append(fold.y_true)
        pooled_pred.append(fold.y_pred)
        selected_configs[str(subject)] = {
            "candidate_index": selection.audit["selected_candidate_index"],
            "config": selection.config.to_dict(),
        }

        joblib.dump(fold.fitted_model, output_dir / f"d1_model_outer_subject_{subject}.joblib", compress=3)
        fold.confusion.to_csv(output_dir / f"d1_confusion_subject_{subject}.csv")
        fold.classification_report.to_csv(output_dir / f"d1_classification_report_subject_{subject}.csv")
        frame_prediction_table(fold.frame_result).to_csv(
            output_dir / f"d1_frame_predictions_subject_{subject}.csv", index=False
        )
        np.savez_compressed(
            output_dir / f"d1_frame_probabilities_subject_{subject}.npz",
            probabilities=fold.frame_result.probabilities,
            coverage=fold.frame_result.coverage,
        )
        (output_dir / f"d1_audit_subject_{subject}.json").write_text(
            json.dumps(fold.audit, indent=2), encoding="utf-8"
        )
        selection.inner_metrics.to_csv(
            output_dir / f"d1_inner_metrics_outer_subject_{subject}.csv", index=False
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "d1_fold_metrics.csv", index=False)
    pooled = evaluate_predictions(np.concatenate(pooled_true), np.concatenate(pooled_pred))
    summary = {
        "candidate": "D1-regularized-C",
        "pooled_accuracy": pooled["accuracy"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_abnormal_f1": pooled["abnormal_f1"],
        "worst_subject_accuracy": float(fold_metrics["accuracy"].min()),
        "worst_subject": int(fold_metrics.loc[fold_metrics["accuracy"].idxmin(), "held_out_subject"]),
        "selection_audit": selection_audit,
        "selected_configs": selected_configs,
    }
    (output_dir / "d1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comparison = compare_d0_d1_from_artifacts(output_dir)
    return {"summary": summary, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe Phase D1 outer LOSO evaluation.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/abc_ablation/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase_d"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.data_dir, args.cache_dir, args.output_dir, args.random_state)
    print(json.dumps(result, indent=2))
