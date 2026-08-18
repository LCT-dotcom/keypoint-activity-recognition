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

from phase_d_classical import fit_classical_model, phase_d_parameter_candidates
from phase_d_d3 import GROUP_CLASSES, select_gate_decoder, to_group_labels
from phase_d_evaluation import (
    FrameProbabilityResult,
    aggregate_window_probabilities,
    estimate_transition_model,
    soft_group_fusion,
    viterbi_decode,
)
from phase_d_protocol import load_unlabeled_windows
from phase_d_runner import CandidateResult, _build_outer_result, select_phase_d_winner
from phase_d_search import select_outer_configs_from_pair_scores
from run_phase_d_d1 import _candidate_from_artifacts, frame_prediction_table
from tsfel_histgb_pipeline import (
    ABNORMAL_CLASSES,
    CLASSES,
    WindowFeatures,
    clean_labels,
    concatenate_window_features,
    evaluate_predictions,
)


SUBJECTS = (1, 2, 3, 5)


def group_windows(windows: WindowFeatures) -> WindowFeatures:
    return WindowFeatures(
        x=windows.x,
        y=pd.Series(to_group_labels(windows.y), name="group_label", dtype="string"),
        meta=windows.meta,
    )


def binary_model_path(output_dir: Path, train_subjects: tuple[int, ...], candidate_index: int) -> Path:
    pair = "_".join(map(str, train_subjects))
    return output_dir / f"d3_binary_pair_{pair}_candidate_{candidate_index:02d}.joblib"


def _atomic_dump(payload: object, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def run(data_dir: Path, c_cache_dir: Path, phase_dir: Path, random_state: int = 42) -> dict:
    training_windows = {
        subject: joblib.load(c_cache_dir / f"abc_c_subject_{subject}_windows.joblib")["windows"]
        for subject in SUBJECTS
    }
    inference_windows = {
        subject: load_unlabeled_windows(
            phase_dir
            / "c_inference_cache"
            / f"c_unlabeled_stride75_subject_{subject}.joblib"
        )
        for subject in SUBJECTS
    }
    truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    candidates = phase_d_parameter_candidates(random_state)
    d1_search_dir = phase_dir / "d1_pair_search"
    d1_scores = pd.read_csv(
        d1_search_dir / "pair_search_scores_unlabeled_inference.csv"
    )
    selections, _ = select_outer_configs_from_pair_scores(d1_scores, SUBJECTS, candidates)

    fold_rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    selection_records: dict[str, dict] = {}
    for outer in SUBJECTS:
        d1_selection = selections[outer]
        candidate_index = int(d1_selection.audit["selected_candidate_index"])
        config = d1_selection.config
        flat_by_split: dict[str, np.ndarray] = {}
        group_by_split: dict[str, np.ndarray] = {}
        truth_by_split: dict[str, np.ndarray] = {}
        transitions_by_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for split in d1_selection.audit["inner_splits"]:
            train_subjects = tuple(sorted(int(subject) for subject in split["train_subjects"]))
            validation_subject = int(split["validation_subject"])
            split_key = f"train_{'_'.join(map(str, train_subjects))}_val_{validation_subject}"
            flat_payload = joblib.load(
                d1_search_dir
                / (
                    f"model_pair_{'_'.join(map(str, train_subjects))}"
                    f"_candidate_{candidate_index:02d}.joblib"
                )
            )
            flat_model = flat_payload["model"]
            validation = inference_windows[validation_subject]
            flat_window_probabilities = flat_model.predict_proba(validation.x, CLASSES)
            flat_frame = aggregate_window_probabilities(
                validation.meta,
                flat_window_probabilities,
                n_frames=len(truth[validation_subject]),
                classes=CLASSES,
            )

            binary_file = binary_model_path(phase_dir, train_subjects, candidate_index)
            binary_payload = None
            if binary_file.exists():
                try:
                    binary_payload = joblib.load(binary_file)
                except (EOFError, OSError, ValueError):
                    binary_payload = None
            signature = {
                "train_subjects": list(train_subjects),
                "candidate_index": candidate_index,
                "config": config.to_dict(),
                "selected_columns": list(flat_model.selector.selected_columns),
            }
            if binary_payload is None or binary_payload.get("signature") != signature:
                train = concatenate_window_features(
                    [training_windows[subject] for subject in train_subjects]
                )
                binary_model = fit_classical_model(
                    group_windows(train),
                    config=config,
                    random_state=random_state + candidate_index + sum(train_subjects) * 100,
                    prefit_selector=flat_model.selector,
                )
                binary_payload = {"signature": signature, "model": binary_model}
                _atomic_dump(binary_payload, binary_file)
            binary_model = binary_payload["model"]
            group_window_probabilities = binary_model.predict_proba(
                validation.x, GROUP_CLASSES
            )
            group_frame = aggregate_window_probabilities(
                validation.meta,
                group_window_probabilities,
                n_frames=len(truth[validation_subject]),
                classes=GROUP_CLASSES,
            )
            flat_by_split[split_key] = flat_frame.probabilities
            group_by_split[split_key] = group_frame.probabilities
            truth_by_split[split_key] = truth[validation_subject]
            transitions_by_split[split_key] = estimate_transition_model(
                [truth[subject] for subject in train_subjects], CLASSES, laplace=1.0
            )

        gate_selection, inner_metrics = select_gate_decoder(
            flat_by_split,
            group_by_split,
            truth_by_split,
            transitions_by_split,
        )
        outer_train = tuple(subject for subject in SUBJECTS if subject != outer)
        flat_outer_model = joblib.load(phase_dir / f"d1u_model_outer_subject_{outer}.joblib")
        test = inference_windows[outer]
        flat_archive = np.load(phase_dir / f"d1u_frame_probabilities_subject_{outer}.npz")
        flat_probabilities = flat_archive["probabilities"]
        outer_binary_model = None
        if gate_selection.alpha == 0.0:
            fused = flat_probabilities
        else:
            outer_binary_train = concatenate_window_features(
                [training_windows[subject] for subject in outer_train]
            )
            outer_binary_model = fit_classical_model(
                group_windows(outer_binary_train),
                config=config,
                random_state=random_state + outer,
                prefit_selector=flat_outer_model.selector,
            )
            group_window_probabilities = outer_binary_model.predict_proba(
                test.x, GROUP_CLASSES
            )
            group_frame = aggregate_window_probabilities(
                test.meta,
                group_window_probabilities,
                n_frames=len(truth[outer]),
                classes=GROUP_CLASSES,
            )
            fused = soft_group_fusion(
                flat_probabilities,
                group_frame.probabilities,
                gate_selection.alpha,
                CLASSES,
                ABNORMAL_CLASSES,
            )
        initial, transition = estimate_transition_model(
            [truth[subject] for subject in outer_train], CLASSES, laplace=1.0
        )
        decoded = viterbi_decode(
            fused,
            transition,
            initial,
            strength=gate_selection.transition_strength,
        )
        labels = np.asarray(CLASSES, dtype=object)[decoded]
        result = FrameProbabilityResult(
            probabilities=fused,
            coverage=flat_archive["coverage"],
            labels=labels,
            confidence=fused[np.arange(len(fused)), decoded],
            classes=tuple(CLASSES),
        )
        audit = dict(d1_selection.audit)
        audit.update(
            {
                "candidate": "D3u-D1u-soft-gate-viterbi",
                "base_candidate": "D1u-regularized-C-unlabeled-inference",
                "selected_config": config.to_dict(),
                "alpha": gate_selection.alpha,
                "transition_strength": gate_selection.transition_strength,
                "inner_summary": gate_selection.inner_summary,
                "held_out_labels_used_for_windowing": False,
            }
        )
        fold = _build_outer_result(
            "D3u-D1u-soft-gate-viterbi",
            outer,
            result,
            truth[outer],
            audit,
            {"flat": flat_outer_model, "binary": outer_binary_model},
        )
        _atomic_dump(
            {
                "flat_model": flat_outer_model,
                "binary_model": outer_binary_model,
                "alpha": gate_selection.alpha,
                "initial_probabilities": initial,
                "transition_probabilities": transition,
                "transition_strength": gate_selection.transition_strength,
                "classes": tuple(CLASSES),
                "group_classes": GROUP_CLASSES,
                "audit": audit,
            },
            phase_dir / f"d3u_model_outer_subject_{outer}.joblib",
        )
        fold_rows.append({"held_out_subject": outer, **fold.metrics})
        pooled_true.append(fold.y_true)
        pooled_pred.append(fold.y_pred)
        selection_records[str(outer)] = audit
        fold.confusion.to_csv(phase_dir / f"d3u_confusion_subject_{outer}.csv")
        fold.classification_report.to_csv(
            phase_dir / f"d3u_classification_report_subject_{outer}.csv"
        )
        frame_prediction_table(result).to_csv(
            phase_dir / f"d3u_frame_predictions_subject_{outer}.csv", index=False
        )
        np.savez_compressed(
            phase_dir / f"d3u_frame_probabilities_subject_{outer}.npz",
            probabilities=result.probabilities,
            coverage=result.coverage,
        )
        inner_metrics.to_csv(
            phase_dir / f"d3u_inner_metrics_outer_subject_{outer}.csv", index=False
        )
        (phase_dir / f"d3u_audit_subject_{outer}.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(phase_dir / "d3u_fold_metrics.csv", index=False)
    pooled = evaluate_predictions(np.concatenate(pooled_true), np.concatenate(pooled_pred))
    summary = {
        "candidate": "D3u-D1u-soft-gate-viterbi",
        "pooled_accuracy": pooled["accuracy"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_abnormal_f1": pooled["abnormal_f1"],
        "worst_subject_accuracy": float(fold_metrics["accuracy"].min()),
        "worst_subject": int(
            fold_metrics.loc[fold_metrics["accuracy"].idxmin(), "held_out_subject"]
        ),
        "outer_selections": selection_records,
        "held_out_windowing": "unlabeled_continuous_stride75",
    }
    (phase_dir / "d3u_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    contenders: list[CandidateResult] = []
    for name, prefix, baseline in (
        ("D0u-C-unlabeled-inference", "d0u", True),
        ("D1u-regularized-C-unlabeled-inference", "d1u", False),
        ("D3u-D1u-soft-gate-viterbi", "d3u", False),
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
            for candidate in contenders
        },
    }
    (phase_dir / "d0u_d1u_d3u_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-free nested D3 soft gate and Viterbi evaluation."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--c-cache-dir", type=Path, default=Path("artifacts/abc_ablation/cache")
    )
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.data_dir, args.c_cache_dir, args.phase_dir, args.random_state), indent=2))
