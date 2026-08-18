import numpy as np

from phase_d_d3 import select_gate_decoder, to_group_labels
from tsfel_histgb_pipeline import CLASSES


def test_group_labels_use_declared_abnormal_classes() -> None:
    labels = np.array(["Walking", "Attacking", "Using phone", "Biting"], dtype=object)

    grouped = to_group_labels(labels)

    assert grouped.tolist() == ["normal", "abnormal", "normal", "abnormal"]


def test_gate_decoder_grid_returns_declared_parameters_and_all_rows() -> None:
    truth = np.array([CLASSES[0], CLASSES[1], CLASSES[0]], dtype=object)
    flat = np.full((3, len(CLASSES)), 0.01)
    flat[np.arange(3), [0, 1, 0]] = 0.93
    groups = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    transition = np.ones((len(CLASSES), len(CLASSES))) / len(CLASSES)
    initial = np.ones(len(CLASSES)) / len(CLASSES)

    selection, metrics = select_gate_decoder(
        {"split": flat},
        {"split": groups},
        {"split": truth},
        {"split": (initial, transition)},
    )

    assert selection.alpha in {0.0, 0.5, 1.0, 2.0}
    assert selection.transition_strength in {0.0, 0.25, 0.5, 1.0, 2.0}
    assert len(metrics) == 20
