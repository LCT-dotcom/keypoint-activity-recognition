from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in minimal inference environments
    njit = None

from tsfel_histgb_pipeline import evaluate_predictions


PROBABILITY_EPSILON = 1e-12


def _viterbi_kernel_impl(
    log_emissions: np.ndarray,
    log_transition: np.ndarray,
    log_initial: np.ndarray,
) -> np.ndarray:
    n_frames, n_classes = log_emissions.shape
    backpointers = np.zeros((n_frames, n_classes), dtype=np.int64)
    previous_scores = log_emissions[0] + log_initial
    for frame_index in range(1, n_frames):
        current_scores = np.empty(n_classes, dtype=np.float64)
        for current in range(n_classes):
            best_previous = 0
            best_score = previous_scores[0] + log_transition[0, current]
            for previous in range(1, n_classes):
                score = previous_scores[previous] + log_transition[previous, current]
                if score > best_score:
                    best_score = score
                    best_previous = previous
            backpointers[frame_index, current] = best_previous
            current_scores[current] = best_score + log_emissions[frame_index, current]
        previous_scores = current_scores
    decoded = np.empty(n_frames, dtype=np.int64)
    decoded[-1] = int(np.argmax(previous_scores))
    for frame_index in range(n_frames - 2, -1, -1):
        decoded[frame_index] = backpointers[frame_index + 1, decoded[frame_index + 1]]
    return decoded


_viterbi_kernel = njit(cache=True)(_viterbi_kernel_impl) if njit is not None else _viterbi_kernel_impl


@dataclass(frozen=True)
class FrameProbabilityResult:
    probabilities: np.ndarray
    coverage: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    classes: tuple[str, ...]


def _normalized_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError(f"{name} rows must have positive sums")
    return probabilities / row_sums


def _fill_nearest_covered(probabilities: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    covered = np.flatnonzero(coverage > 0)
    if len(covered) == 0:
        raise ValueError("No frame is covered by a valid window")
    missing = np.flatnonzero(coverage == 0)
    if len(missing) == 0:
        return probabilities

    insertions = np.searchsorted(covered, missing)
    left_positions = np.clip(insertions - 1, 0, len(covered) - 1)
    right_positions = np.clip(insertions, 0, len(covered) - 1)
    left = covered[left_positions]
    right = covered[right_positions]
    nearest = np.where(missing - left <= right - missing, left, right)
    probabilities[missing] = probabilities[nearest]
    return probabilities


def aggregate_window_probabilities(
    meta: pd.DataFrame,
    window_probabilities: np.ndarray,
    n_frames: int,
    classes: Sequence[str],
) -> FrameProbabilityResult:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    class_order = tuple(classes)
    if not class_order or len(set(class_order)) != len(class_order):
        raise ValueError("classes must be non-empty and unique")
    required = {"start_pos", "end_pos"}
    missing_columns = sorted(required - set(meta.columns))
    if missing_columns:
        raise ValueError(f"Window metadata is missing columns: {missing_columns}")

    probabilities = _normalized_probabilities(window_probabilities, "window_probabilities")
    if len(meta) != len(probabilities):
        raise ValueError("Window metadata and probabilities must have equal row counts")
    if probabilities.shape[1] != len(class_order):
        raise ValueError("Probability column count does not match classes")

    totals = np.zeros((n_frames, len(class_order)), dtype=float)
    coverage = np.zeros(n_frames, dtype=np.int64)
    for row_index, row in enumerate(meta[["start_pos", "end_pos"]].itertuples(index=False)):
        start = max(0, int(row.start_pos))
        end = min(n_frames, int(row.end_pos))
        if start >= end:
            raise ValueError(f"Window {row_index} has an empty frame interval")
        totals[start:end] += probabilities[row_index]
        coverage[start:end] += 1

    covered = coverage > 0
    totals[covered] /= coverage[covered, None]
    totals = _fill_nearest_covered(totals, coverage)
    totals = _normalized_probabilities(totals, "frame_probabilities")
    winner_indices = totals.argmax(axis=1)
    labels = np.asarray(class_order, dtype=object)[winner_indices]
    confidence = totals[np.arange(n_frames), winner_indices]
    return FrameProbabilityResult(
        probabilities=totals,
        coverage=coverage,
        labels=labels,
        confidence=confidence,
        classes=class_order,
    )


def soft_group_fusion(
    flat_probabilities: np.ndarray,
    group_probabilities: np.ndarray,
    alpha: float,
    classes: Sequence[str],
    abnormal_classes: set[str],
) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    class_order = tuple(classes)
    flat = _normalized_probabilities(flat_probabilities, "flat_probabilities")
    groups = _normalized_probabilities(group_probabilities, "group_probabilities")
    if flat.shape[0] != groups.shape[0] or flat.shape[1] != len(class_order):
        raise ValueError("Flat/group rows and flat class columns must match")
    if groups.shape[1] != 2:
        raise ValueError("group_probabilities must use [normal, abnormal] column order")
    unknown = set(abnormal_classes) - set(class_order)
    if unknown:
        raise ValueError(f"Unknown abnormal classes: {sorted(unknown)}")

    group_indices = np.asarray(
        [1 if label in abnormal_classes else 0 for label in class_order],
        dtype=int,
    )
    adjusted = flat * np.power(np.clip(groups[:, group_indices], PROBABILITY_EPSILON, 1.0), alpha)
    adjusted = np.clip(adjusted, PROBABILITY_EPSILON, None)
    return _normalized_probabilities(adjusted, "fused_probabilities")


def estimate_transition_model(
    label_sequences: Iterable[Sequence[str]],
    classes: Sequence[str],
    laplace: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if laplace <= 0:
        raise ValueError("laplace must be positive")
    class_order = tuple(classes)
    if not class_order or len(set(class_order)) != len(class_order):
        raise ValueError("classes must be non-empty and unique")
    class_to_index = {label: index for index, label in enumerate(class_order)}
    initial_counts = np.full(len(class_order), laplace, dtype=float)
    transition_counts = np.full((len(class_order), len(class_order)), laplace, dtype=float)

    for sequence in label_sequences:
        previous: int | None = None
        for label in sequence:
            current = class_to_index.get(label)
            if current is None:
                previous = None
                continue
            if previous is None:
                initial_counts[current] += 1
            else:
                transition_counts[previous, current] += 1
            previous = current

    initial = initial_counts / initial_counts.sum()
    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return initial, transition


def viterbi_decode(
    emission_probabilities: np.ndarray,
    transition_probabilities: np.ndarray,
    initial_probabilities: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    if strength < 0:
        raise ValueError("strength must be non-negative")
    emissions = _normalized_probabilities(emission_probabilities, "emission_probabilities")
    transition = _normalized_probabilities(transition_probabilities, "transition_probabilities")
    initial = np.asarray(initial_probabilities, dtype=float)
    if initial.ndim != 1 or len(initial) != emissions.shape[1]:
        raise ValueError("initial_probabilities must match the emission class count")
    if transition.shape != (emissions.shape[1], emissions.shape[1]):
        raise ValueError("transition_probabilities must be square and match emissions")
    if not np.isfinite(initial).all() or (initial < 0).any() or initial.sum() <= 0:
        raise ValueError("initial_probabilities must be finite, non-negative, and non-empty")
    initial = initial / initial.sum()
    if strength == 0:
        return emissions.argmax(axis=1)

    log_emissions = np.log(np.clip(emissions, PROBABILITY_EPSILON, 1.0))
    log_transition = np.log(np.clip(transition, PROBABILITY_EPSILON, 1.0)) * strength
    log_initial = strength * np.log(np.clip(initial, PROBABILITY_EPSILON, 1.0))
    return _viterbi_kernel(log_emissions, log_transition, log_initial)


def evaluate_frame_result(
    y_true: Sequence[str],
    result: FrameProbabilityResult,
    valid_classes: Sequence[str] | None = None,
) -> dict:
    truth = np.asarray(y_true, dtype=object)
    if len(truth) != len(result.labels):
        raise ValueError("Ground truth and frame predictions must have equal lengths")
    allowed = set(valid_classes or result.classes)
    mask = np.asarray([label in allowed for label in truth], dtype=bool)
    if not mask.any():
        raise ValueError("Ground truth contains no valid target-class frames")
    return evaluate_predictions(truth[mask], result.labels[mask])
