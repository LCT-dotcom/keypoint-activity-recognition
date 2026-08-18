import numpy as np
import pandas as pd

from phase_d_evaluation import FrameProbabilityResult
from run_phase_d_d1 import _candidate_from_artifacts, frame_prediction_table


def test_frame_prediction_table_uses_integer_coverage_counts() -> None:
    result = FrameProbabilityResult(
        probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]),
        coverage=np.array([1, 3]),
        labels=np.array(["A", "B"], dtype=object),
        confidence=np.array([0.8, 0.9]),
        classes=("A", "B"),
    )

    table = frame_prediction_table(result)

    assert table["coverage"].tolist() == [1, 3]
    assert table["confidence"].tolist() == [0.8, 0.9]


def test_candidate_adapter_accepts_legacy_d0_summary_keys() -> None:
    folds = pd.DataFrame([{"held_out_subject": 1, "accuracy": 0.5}])
    summary = {"accuracy": 0.5, "macro_f1": 0.4, "abnormal_f1": 0.7}

    result = _candidate_from_artifacts("D0-C", folds, summary, True)

    assert result.pooled_accuracy == 0.5
    assert result.pooled_macro_f1 == 0.4
    assert result.pooled_abnormal_f1 == 0.7
