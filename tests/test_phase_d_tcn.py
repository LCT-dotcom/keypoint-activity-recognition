import numpy as np
import pandas as pd
import torch

from phase_d_tcn import (
    MultiStreamTCN,
    TCNConfig,
    augment_coordinate_batch,
    build_tcn_streams,
    extract_tcn_windows,
    focal_cross_entropy,
    streams_from_coordinate_tensor,
)
from tsfel_histgb_pipeline import JOINTS


def make_normalized_pose(frames: int = 150) -> pd.DataFrame:
    time = np.arange(frames, dtype=float)
    values = {}
    for index, joint in enumerate(JOINTS):
        values[f"{joint}_x"] = np.sin(time / 20 + index) * 0.1
        values[f"{joint}_y"] = np.cos(time / 20 + index) * 0.1
    return pd.DataFrame(values)


def test_tcn_streams_have_stable_channel_time_layout() -> None:
    streams = build_tcn_streams(make_normalized_pose())

    assert streams["coordinates"].shape == (34, 150)
    assert streams["bones"].shape[1] == 150
    assert streams["velocity"].shape == (34, 150)
    assert all(np.isfinite(stream).all() for stream in streams.values())


def test_multistream_tcn_outputs_eight_class_and_binary_logits() -> None:
    config = TCNConfig(channels=16, blocks=3, dropout=0.2, loss="cross_entropy", aux_weight=0.2)
    model = MultiStreamTCN(
        {"coordinates": 34, "bones": 42, "velocity": 34}, config, num_classes=8
    )
    inputs = {
        "coordinates": torch.randn(4, 34, 150),
        "bones": torch.randn(4, 42, 150),
        "velocity": torch.randn(4, 34, 150),
    }

    class_logits, group_logits = model(inputs)

    assert class_logits.shape == (4, 8)
    assert group_logits.shape == (4, 2)


def test_tcn_window_extraction_keeps_subject_metadata_and_labels() -> None:
    pose = make_normalized_pose(frames=360)
    pose.insert(0, "frame_id", np.arange(len(pose)))
    pose["Action Label"] = np.where(
        np.arange(len(pose)) < 180, "Walking", "Attacking"
    )

    windows = extract_tcn_windows(
        pose,
        subject_id=7,
        window_size=150,
        stride=75,
        majority_threshold=0.70,
        labeled=True,
    )

    assert windows.streams["coordinates"].shape[2] == 150
    assert len(windows.labels) == len(windows.meta)
    assert windows.meta["subject_id"].eq(7).all()
    assert set(windows.labels) <= {"Walking", "Attacking"}


def test_tcn_training_augmentation_preserves_stream_shapes() -> None:
    generator = torch.Generator().manual_seed(42)
    coordinates = torch.randn(4, 34, 150)

    augmented = augment_coordinate_batch(coordinates, generator)
    streams = streams_from_coordinate_tensor(augmented)

    assert augmented.shape == coordinates.shape
    assert streams["coordinates"].shape == (4, 34, 150)
    assert streams["bones"].shape == (4, 42, 150)
    assert streams["velocity"].shape == (4, 34, 150)
    assert all(torch.isfinite(values).all() for values in streams.values())


def test_focal_loss_penalizes_wrong_confident_prediction_more() -> None:
    targets = torch.tensor([0])
    easy = focal_cross_entropy(torch.tensor([[5.0, -5.0]]), targets)
    wrong = focal_cross_entropy(torch.tensor([[-5.0, 5.0]]), targets)

    assert easy.item() < wrong.item()
