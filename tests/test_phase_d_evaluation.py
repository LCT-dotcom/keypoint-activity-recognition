from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

from phase_d_evaluation import (
    aggregate_window_probabilities,
    estimate_transition_model,
    soft_group_fusion,
    viterbi_decode,
)


def test_overlapping_probabilities_are_averaged_and_edges_are_covered() -> None:
    meta = pd.DataFrame({"start_pos": [0, 2], "end_pos": [4, 6]})
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])

    result = aggregate_window_probabilities(
        meta,
        probabilities,
        n_frames=6,
        classes=["A", "B"],
    )

    np.testing.assert_allclose(result.probabilities[2:4], [[0.5, 0.5], [0.5, 0.5]])
    assert result.coverage.tolist() == [1, 1, 2, 2, 1, 1]
    assert result.labels.tolist() == ["A", "A", "A", "A", "B", "B"]


def test_uncovered_frames_copy_the_nearest_covered_probability() -> None:
    meta = pd.DataFrame({"start_pos": [1, 3], "end_pos": [2, 4]})
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])

    result = aggregate_window_probabilities(
        meta,
        probabilities,
        n_frames=5,
        classes=["A", "B"],
    )

    np.testing.assert_allclose(result.probabilities[0], probabilities[0])
    np.testing.assert_allclose(result.probabilities[2], probabilities[0])
    np.testing.assert_allclose(result.probabilities[4], probabilities[1])
    assert result.coverage.tolist() == [0, 1, 0, 1, 0]


def test_soft_gate_reweights_but_never_removes_a_class() -> None:
    flat = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=float)
    gate = np.array([[0.9, 0.1], [0.25, 0.75]], dtype=float)

    fused = soft_group_fusion(
        flat,
        gate,
        alpha=1.0,
        classes=["Normal", "Abnormal"],
        abnormal_classes={"Abnormal"},
    )

    assert np.all(fused > 0)
    np.testing.assert_allclose(fused.sum(axis=1), 1.0)
    assert fused[0, 0] > flat[0, 0]
    assert fused[1, 1] > flat[1, 1]


def test_zero_transition_strength_matches_frame_argmax() -> None:
    emissions = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
        ]
    )
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    initial = np.array([0.5, 0.5])

    decoded = viterbi_decode(emissions, transition, initial, strength=0.0)

    assert decoded.tolist() == emissions.argmax(axis=1).tolist()


def test_transition_estimation_is_smoothed_and_row_normalized() -> None:
    initial, transition = estimate_transition_model(
        [np.array(["A", "A", "B"]), np.array(["B", "A"])],
        classes=["A", "B"],
        laplace=1.0,
    )

    assert np.all(initial > 0)
    assert np.all(transition > 0)
    np.testing.assert_allclose(initial.sum(), 1.0)
    np.testing.assert_allclose(transition.sum(axis=1), np.ones(2))


def test_invalid_labels_break_transition_sequences() -> None:
    initial, transition = estimate_transition_model(
        [np.array(["A", None, "B"], dtype=object)],
        classes=["A", "B"],
        laplace=1.0,
    )

    np.testing.assert_allclose(initial, [0.5, 0.5])
    np.testing.assert_allclose(transition, [[0.5, 0.5], [0.5, 0.5]])


def test_viterbi_matches_brute_force_path_score() -> None:
    emissions = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.array([0.55, 0.45])
    strength = 0.5

    decoded = viterbi_decode(emissions, transition, initial, strength=strength)
    paths = list(product(range(2), repeat=len(emissions)))
    scores = []
    for path in paths:
        score = np.log(emissions[0, path[0]]) + strength * np.log(initial[path[0]])
        for frame in range(1, len(path)):
            score += np.log(emissions[frame, path[frame]])
            score += strength * np.log(transition[path[frame - 1], path[frame]])
        scores.append(score)

    assert decoded.tolist() == list(paths[int(np.argmax(scores))])
