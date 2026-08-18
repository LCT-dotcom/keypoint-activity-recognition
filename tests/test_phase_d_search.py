from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase_d_classical import ClassicalConfig
from phase_d_search import run_resumable_pair_search, select_outer_configs_from_pair_scores
from tsfel_histgb_pipeline import WindowFeatures


def make_windows(subject: int, rows: int = 24) -> tuple[WindowFeatures, np.ndarray]:
    rng = np.random.default_rng(subject)
    labels = np.resize(np.array(["Attacking", "Walking"], dtype=object), rows)
    x = pd.DataFrame(
        {
            "signal": (labels == "Attacking").astype(float) + rng.normal(0, 0.05, rows),
            "noise": rng.normal(size=rows),
        }
    )
    starts = np.arange(rows) * 5
    meta = pd.DataFrame({"subject_id": subject, "start_pos": starts, "end_pos": starts + 10})
    truth = np.resize(labels.repeat(5), int(starts[-1] + 10))
    return WindowFeatures(x=x, y=pd.Series(labels, dtype="string"), meta=meta), truth


def test_pair_search_is_complete_and_resumable() -> None:
    prepared = {subject: make_windows(subject) for subject in [1, 2, 3, 5]}
    windows = {subject: item[0] for subject, item in prepared.items()}
    truth = {subject: item[1] for subject, item in prepared.items()}
    config = ClassicalConfig(2, 0.1, 15, 5, 1.0, "none")
    output_dir = Path("artifacts/phase_d/test_pair_search")

    first_scores, first_audit = run_resumable_pair_search(
        windows,
        truth,
        candidates=[config],
        output_dir=output_dir,
        random_state=42,
        force=True,
    )
    second_scores, second_audit = run_resumable_pair_search(
        windows,
        truth,
        candidates=[config],
        output_dir=output_dir,
        random_state=42,
        force=False,
    )

    assert len(first_scores) == 12
    assert first_audit["selector_fits"] == 6
    assert first_audit["model_fits"] == 6
    assert first_audit["completed_pair_candidates"] == 6
    assert second_audit["selector_fits"] == 0
    assert second_audit["model_fits"] == 0
    assert second_audit["cache_hits"] == 6
    pd.testing.assert_frame_equal(first_scores, second_scores)


def test_outer_config_selection_uses_only_development_subjects() -> None:
    candidates = [
        ClassicalConfig(2, 0.1, 15, 5, 1.0, "none"),
        ClassicalConfig(2, 0.05, 31, 10, 0.5, "sqrt_balanced"),
    ]
    rows = []
    subjects = [1, 2, 3, 5]
    for train_subjects in [(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 5)]:
        for validation_subject in sorted(set(subjects) - set(train_subjects)):
            for candidate_index in range(2):
                # Candidate 1 wins all legitimate inner folds. An artificially high
                # score involving outer subject 5 must never influence outer-5 selection.
                accuracy = 0.7 if candidate_index == 1 else 0.6
                if validation_subject == 5 and candidate_index == 0:
                    accuracy = 0.99
                rows.append(
                    {
                        "train_subjects": "_".join(map(str, train_subjects)),
                        "candidate_index": candidate_index,
                        "validation_subject": validation_subject,
                        "accuracy": accuracy,
                        "macro_f1": accuracy,
                        "abnormal_f1": accuracy,
                    }
                )

    selections, audit = select_outer_configs_from_pair_scores(
        pd.DataFrame(rows), subjects, candidates
    )

    assert set(selections) == set(subjects)
    assert selections[5].config == candidates[1]
    assert all(5 not in split["train_subjects"] for split in selections[5].audit["inner_splits"])
    assert all(split["validation_subject"] != 5 for split in selections[5].audit["inner_splits"])
    assert audit["outer_fold_count"] == 4
    assert audit["leakage_check_passed"] is True
