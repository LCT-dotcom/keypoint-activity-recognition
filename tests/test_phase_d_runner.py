from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_d_classical import ClassicalConfig
from phase_d_runner import (
    CandidateResult,
    run_d0_fold,
    run_d1_fold_with_selection,
    run_d1_fold,
    select_classical_config,
    select_phase_d_winner,
    validate_outer_audit,
)
from tsfel_histgb_pipeline import WindowFeatures


def candidate(
    name: str,
    accuracies: dict[int, float],
    pooled_accuracy: float,
    abnormal_f1: float,
    macro_f1: float,
) -> CandidateResult:
    return CandidateResult(
        name=name,
        fold_metrics=pd.DataFrame(
            [{"held_out_subject": subject, "accuracy": accuracy} for subject, accuracy in accuracies.items()]
        ),
        pooled_accuracy=pooled_accuracy,
        pooled_macro_f1=macro_f1,
        pooled_abnormal_f1=abnormal_f1,
        is_baseline=name == "D0-C",
    )


def make_subject_windows(subject: int, rows: int = 32) -> tuple[WindowFeatures, np.ndarray]:
    rng = np.random.default_rng(subject)
    labels = np.resize(np.array(["Attacking", "Walking"], dtype=object), rows)
    useful = (labels == "Attacking").astype(float) + rng.normal(0, 0.05, rows)
    x = pd.DataFrame({"useful": useful, "noise": rng.normal(size=rows)})
    starts = np.arange(rows) * 10
    meta = pd.DataFrame(
        {
            "subject_id": subject,
            "start_pos": starts,
            "end_pos": starts + 20,
        }
    )
    truth = np.resize(labels.repeat(10), int(starts[-1] + 20))
    return WindowFeatures(x=x, y=pd.Series(labels, dtype="string"), meta=meta), truth


def test_ineligible_accuracy_gain_cannot_beat_baseline() -> None:
    baseline = candidate("D0-C", {1: 0.50, 2: 0.60}, 0.56, 0.70, 0.55)
    high_accuracy_low_abnormal = candidate("bad", {1: 0.60, 2: 0.70}, 0.66, 0.60, 0.65)

    winner = select_phase_d_winner([baseline, high_accuracy_low_abnormal])

    assert winner.name == "D0-C"
    assert not high_accuracy_low_abnormal.eligible


def test_eligible_candidates_rank_by_accuracy_then_worst_subject() -> None:
    baseline = candidate("D0-C", {1: 0.50, 2: 0.60}, 0.56, 0.70, 0.55)
    candidate_a = candidate("candidate_a", {1: 0.53, 2: 0.66}, 0.60, 0.70, 0.60)
    candidate_b = candidate("candidate_b", {1: 0.55, 2: 0.65}, 0.60, 0.71, 0.59)

    winner = select_phase_d_winner([baseline, candidate_a, candidate_b])

    assert winner.name == "candidate_b"
    assert candidate_a.eligible and candidate_b.eligible


def test_outer_audit_rejects_held_out_subject_in_fit_records() -> None:
    valid = {
        "outer_held_out_subject": 3,
        "outer_train_subjects": [1, 2, 5],
        "inner_splits": [
            {"train_subjects": [2, 5], "validation_subject": 1},
            {"train_subjects": [1, 5], "validation_subject": 2},
            {"train_subjects": [1, 2], "validation_subject": 5},
        ],
    }
    validate_outer_audit(valid)

    invalid = dict(valid)
    invalid["outer_train_subjects"] = [1, 2, 3, 5]
    with pytest.raises(ValueError, match="held-out subject leaked"):
        validate_outer_audit(invalid)


def test_d0_fold_produces_frame_metrics_and_leakage_audit() -> None:
    prepared = {subject: make_subject_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}

    fold = run_d0_fold(windows, truth, held_out_subject=3, random_state=42)

    assert fold.held_out_subject == 3
    assert len(fold.frame_result.labels) == len(truth[3])
    assert 0 <= fold.metrics["accuracy"] <= 1
    assert fold.audit["outer_train_subjects"] == [1, 2, 5]
    validate_outer_audit(fold.audit)


def test_nested_classical_selection_uses_only_inner_subjects() -> None:
    prepared = {subject: make_subject_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}
    config = ClassicalConfig(2, 0.1, 15, 5, 1.0, "none")

    selection = select_classical_config(
        windows,
        truth,
        outer_held_out_subject=3,
        candidates=[config],
        random_state=42,
    )

    assert selection.config == config
    assert selection.audit["outer_held_out_subject"] == 3
    validate_outer_audit(selection.audit)


def test_nested_selection_fits_feature_ranking_once_per_inner_split() -> None:
    prepared = {subject: make_subject_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}
    candidates = [
        ClassicalConfig(1, 0.1, 15, 5, 1.0, "none"),
        ClassicalConfig(2, 0.1, 15, 5, 1.0, "none"),
    ]

    selection = select_classical_config(
        windows,
        truth,
        outer_held_out_subject=3,
        candidates=candidates,
        random_state=42,
    )

    assert selection.audit["selector_fit_count"] == 3


def test_d1_fold_uses_nested_selected_config_and_outer_audit() -> None:
    prepared = {subject: make_subject_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}
    config = ClassicalConfig(2, 0.1, 15, 5, 1.0, "none")
    selector_cache: dict = {}
    model_cache: dict = {}

    fold = run_d1_fold(
        windows,
        truth,
        held_out_subject=3,
        candidates=[config],
        random_state=42,
        selector_cache=selector_cache,
        model_cache=model_cache,
    )

    assert fold.candidate_name == "D1-regularized-C"
    assert fold.audit["selected_config"] == config.to_dict()
    assert len(selector_cache) == 3 and len(model_cache) == 3
    validate_outer_audit(fold.audit)


def test_d1_fold_with_selection_skips_inner_search_and_preserves_audit() -> None:
    prepared = {subject: make_subject_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}
    config = ClassicalConfig(2, 0.1, 15, 5, 1.0, "none")
    audit = {
        "outer_held_out_subject": 3,
        "outer_train_subjects": [1, 2, 5],
        "inner_splits": [
            {"train_subjects": [2, 5], "validation_subject": 1},
            {"train_subjects": [1, 5], "validation_subject": 2},
            {"train_subjects": [1, 2], "validation_subject": 5},
        ],
        "selected_candidate_index": 0,
    }

    fold = run_d1_fold_with_selection(
        windows,
        truth,
        held_out_subject=3,
        config=config,
        selection_audit=audit,
        random_state=42,
    )

    assert fold.audit["selected_candidate_index"] == 0
    assert fold.audit["selected_config"] == config.to_dict()
    assert fold.audit["outer_train_subjects"] == [1, 2, 5]
    validate_outer_audit(fold.audit)


def test_nested_selection_reuses_training_pair_selector_and_model_caches() -> None:
    prepared = {subject: make_subject_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}
    config = ClassicalConfig(2, 0.1, 15, 5, 1.0, "none")
    selector_cache: dict = {}
    model_cache: dict = {}

    select_classical_config(
        windows,
        truth,
        outer_held_out_subject=3,
        candidates=[config],
        selector_cache=selector_cache,
        model_cache=model_cache,
    )
    second = select_classical_config(
        windows,
        truth,
        outer_held_out_subject=2,
        candidates=[config],
        selector_cache=selector_cache,
        model_cache=model_cache,
    )

    assert len(selector_cache) == 5
    assert len(model_cache) == 5
    assert second.audit["selector_cache_hits"] == 1
    assert second.audit["model_cache_hits"] == 1
