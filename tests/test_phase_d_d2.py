import numpy as np
import pandas as pd

from phase_d_d2 import select_fusion_weight, select_scale_settings
from tsfel_histgb_pipeline import CLASSES


def test_scale_selection_excludes_outer_held_out_subject() -> None:
    subjects = [1, 2, 3, 5]
    rows = []
    for train_pair in [(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 5)]:
        for validation_subject in sorted(set(subjects) - set(train_pair)):
            for scale in (60, 150, 300):
                for candidate_index in (0, 1):
                    for threshold in (0.70, 0.85):
                        for stride in (15, 30):
                            accuracy = 0.7 if candidate_index == 1 else 0.6
                            if validation_subject == 5 and candidate_index == 0:
                                accuracy = 0.99
                            rows.append(
                                {
                                    "train_subjects": "_".join(map(str, train_pair)),
                                    "scale": scale,
                                    "candidate_index": candidate_index,
                                    "majority_threshold": threshold,
                                    "inference_stride": stride,
                                    "validation_subject": validation_subject,
                                    "accuracy": accuracy,
                                    "macro_f1": accuracy,
                                    "abnormal_f1": accuracy,
                                }
                            )

    selections, audit = select_scale_settings(
        pd.DataFrame(rows),
        subject_ids=subjects,
        outer_held_out_subject=5,
        allowed_candidate_indices=[0, 1],
    )

    assert set(selections) == {60, 150, 300}
    assert all(selection.candidate_index == 1 for selection in selections.values())
    assert audit["leakage_check_passed"] is True
    assert all(split["validation_subject"] != 5 for split in audit["inner_splits"])


def test_fusion_selection_prefers_scale_with_best_inner_frame_predictions() -> None:
    truth = np.array([CLASSES[0], CLASSES[1], CLASSES[0]], dtype=object)
    perfect = np.full((3, len(CLASSES)), 0.01)
    perfect[np.arange(3), [0, 1, 0]] = 0.93
    wrong = np.full((3, len(CLASSES)), 0.01)
    wrong[:, 2] = 0.93
    probabilities = {
        "split-a": {60: perfect, 150: wrong, 300: wrong},
        "split-b": {60: perfect, 150: wrong, 300: wrong},
    }
    truths = {"split-a": truth, "split-b": truth}

    selection, metrics = select_fusion_weight(probabilities, truths)

    assert selection == (1.0, 0.0, 0.0)
    assert len(metrics) == 20
