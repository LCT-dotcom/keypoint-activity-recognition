import pandas as pd

from run_phase_d_lock_s4 import select_global_d1_candidate


def test_global_lock_selection_uses_all_inner_rows_and_declared_ranking() -> None:
    scores = pd.DataFrame(
        [
            {"candidate_index": 0, "accuracy": 0.6, "macro_f1": 0.5, "abnormal_f1": 0.7},
            {"candidate_index": 0, "accuracy": 0.6, "macro_f1": 0.5, "abnormal_f1": 0.7},
            {"candidate_index": 1, "accuracy": 0.7, "macro_f1": 0.6, "abnormal_f1": 0.7},
            {"candidate_index": 1, "accuracy": 0.4, "macro_f1": 0.6, "abnormal_f1": 0.7},
        ]
    )

    selected, ranked = select_global_d1_candidate(scores)

    assert selected == 0
    assert ranked.iloc[0]["mean_accuracy"] == 0.6
