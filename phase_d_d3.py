from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from phase_d_evaluation import soft_group_fusion, viterbi_decode
from tsfel_histgb_pipeline import ABNORMAL_CLASSES, CLASSES, evaluate_predictions


GATE_ALPHAS = (0.0, 0.5, 1.0, 2.0)
TRANSITION_STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)
GROUP_CLASSES = ("normal", "abnormal")


@dataclass(frozen=True)
class GateDecoderSelection:
    alpha: float
    transition_strength: float
    inner_summary: dict[str, float]


def to_group_labels(labels: Sequence[str]) -> np.ndarray:
    return np.asarray(
        ["abnormal" if label in ABNORMAL_CLASSES else "normal" for label in labels],
        dtype=object,
    )


def select_gate_decoder(
    flat_probabilities_by_split: dict[object, np.ndarray],
    group_probabilities_by_split: dict[object, np.ndarray],
    truth_by_split: dict[object, Sequence[str]],
    transition_models_by_split: dict[object, tuple[np.ndarray, np.ndarray]],
) -> tuple[GateDecoderSelection, pd.DataFrame]:
    split_keys = set(flat_probabilities_by_split)
    if not split_keys:
        raise ValueError("Gate/decoder selection requires at least one inner split")
    if not (
        split_keys
        == set(group_probabilities_by_split)
        == set(truth_by_split)
        == set(transition_models_by_split)
    ):
        raise ValueError("Gate/decoder inputs must contain identical inner splits")

    rows: list[dict] = []
    valid_classes = set(CLASSES)
    for alpha_index, alpha in enumerate(GATE_ALPHAS):
        for strength_index, strength in enumerate(TRANSITION_STRENGTHS):
            for split_key in flat_probabilities_by_split:
                flat = np.asarray(flat_probabilities_by_split[split_key], dtype=float)
                groups = np.asarray(group_probabilities_by_split[split_key], dtype=float)
                truth = np.asarray(truth_by_split[split_key], dtype=object)
                if len(flat) != len(groups) or len(flat) != len(truth):
                    raise ValueError("Gate/decoder frame inputs must have equal lengths")
                fused = soft_group_fusion(
                    flat,
                    groups,
                    alpha=alpha,
                    classes=CLASSES,
                    abnormal_classes=ABNORMAL_CLASSES,
                )
                initial, transition = transition_models_by_split[split_key]
                decoded_indices = viterbi_decode(
                    fused,
                    transition,
                    initial,
                    strength=strength,
                )
                predicted = np.asarray(CLASSES, dtype=object)[decoded_indices]
                valid = np.asarray([label in valid_classes for label in truth], dtype=bool)
                metrics = evaluate_predictions(truth[valid], predicted[valid])
                rows.append(
                    {
                        "alpha_index": alpha_index,
                        "strength_index": strength_index,
                        "alpha": alpha,
                        "transition_strength": strength,
                        "split": str(split_key),
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "abnormal_f1": metrics["abnormal_f1"],
                    }
                )
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(
        ["alpha_index", "strength_index", "alpha", "transition_strength"],
        as_index=False,
    ).agg(
        mean_accuracy=("accuracy", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_abnormal_f1=("abnormal_f1", "mean"),
    )
    best = summary.sort_values(
        ["mean_accuracy", "worst_accuracy", "mean_macro_f1", "mean_abnormal_f1"],
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    selection = GateDecoderSelection(
        alpha=float(best["alpha"]),
        transition_strength=float(best["transition_strength"]),
        inner_summary={
            "mean_accuracy": float(best["mean_accuracy"]),
            "worst_accuracy": float(best["worst_accuracy"]),
            "mean_macro_f1": float(best["mean_macro_f1"]),
            "mean_abnormal_f1": float(best["mean_abnormal_f1"]),
        },
    )
    return selection, metrics
