from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_d_classical import (
    ClassicalConfig,
    TrainingOnlyFeatureSelector,
    build_inner_subject_splits,
    fit_classical_model,
    phase_d_parameter_candidates,
)
from tsfel_histgb_pipeline import WindowFeatures


def make_windows(rows: int = 80) -> WindowFeatures:
    rng = np.random.default_rng(42)
    labels = np.resize(np.array(["Attacking", "Walking"], dtype=object), rows)
    signal = (labels == "Attacking").astype(float)
    x = pd.DataFrame(
        {
            "useful": signal + rng.normal(0, 0.05, rows),
            "duplicate": signal + rng.normal(0, 0.0001, rows),
            "noise": rng.normal(size=rows),
            "constant": 1.0,
        }
    )
    meta = pd.DataFrame(
        {
            "subject_id": 1,
            "start_pos": np.arange(rows) * 10,
            "end_pos": np.arange(rows) * 10 + 10,
        }
    )
    return WindowFeatures(x=x, y=pd.Series(labels, dtype="string"), meta=meta)


def test_selector_fit_uses_training_rows_only() -> None:
    windows = make_windows()
    selector = TrainingOnlyFeatureSelector(feature_budget=2, random_state=42)
    selector.fit(windows.x.iloc[:60], windows.y.iloc[:60])
    selected_before = selector.selected_columns

    altered_test_x = windows.x.iloc[60:].copy() * 1_000_000
    transformed = selector.transform(altered_test_x)

    assert selector.selected_columns == selected_before
    assert transformed.columns.tolist() == list(selected_before)
    assert "constant" not in selected_before


def test_transform_rejects_missing_selected_columns() -> None:
    windows = make_windows()
    selector = TrainingOnlyFeatureSelector(feature_budget=2, random_state=42).fit(
        windows.x,
        windows.y,
    )

    with pytest.raises(ValueError, match="missing selected features"):
        selector.transform(windows.x.drop(columns=[selector.selected_columns[0]]))


def test_classical_model_produces_normalized_probabilities_in_requested_order() -> None:
    windows = make_windows()
    config = ClassicalConfig(
        feature_budget=2,
        learning_rate=0.1,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        weighting="none",
    )

    fitted = fit_classical_model(windows, config=config, random_state=42)
    probabilities = fitted.predict_proba(
        windows.x.iloc[:5],
        class_order=["Attacking", "Walking", "Biting"],
    )

    assert probabilities.shape == (5, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, 2] == 0)
    assert fitted.classifier.early_stopping is True


def test_parameter_candidates_are_bounded_and_deterministic() -> None:
    first = phase_d_parameter_candidates()
    second = phase_d_parameter_candidates()

    assert len(first) == 24
    assert first == second
    assert {candidate.weighting for candidate in first} <= {"none", "sqrt_balanced"}
    assert {candidate.feature_budget for candidate in first} <= {128, 256, 512, 1024}


def test_fitted_selector_can_create_stable_budget_prefix_without_refitting() -> None:
    windows = make_windows()
    selector = TrainingOnlyFeatureSelector(feature_budget=4, random_state=42).fit(windows.x, windows.y)

    smaller = selector.restrict_budget(1)

    assert smaller.selected_columns == selector.selected_columns[:1]
    assert selector.selected_columns != ()
    assert smaller.fill_values_.equals(selector.fill_values_)


def test_classical_model_can_reuse_a_prefit_selector() -> None:
    windows = make_windows()
    selector = TrainingOnlyFeatureSelector(feature_budget=2, random_state=42).fit(windows.x, windows.y)
    selected_before = selector.selected_columns
    config = ClassicalConfig(2, 0.1, 15, 10, 1.0, "none")

    fitted = fit_classical_model(
        windows,
        config=config,
        random_state=42,
        prefit_selector=selector,
    )

    assert fitted.selector.selected_columns == selected_before


def test_inner_subject_splits_never_include_outer_held_out_subject() -> None:
    splits = build_inner_subject_splits([1, 2, 3, 5], outer_held_out_subject=3)

    assert {validation for _, validation in splits} == {1, 2, 5}
    for train_subjects, validation_subject in splits:
        assert 3 not in train_subjects
        assert validation_subject != 3
        assert validation_subject not in train_subjects
