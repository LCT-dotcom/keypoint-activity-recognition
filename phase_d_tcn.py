from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from abc_experiment_pipeline import normalize_for_experiment
from phase_d_multiscale import BONES
from tsfel_histgb_pipeline import (
    COORD_COLUMNS,
    FS,
    JOINTS,
    _window_metadata,
    majority_label,
    prepare_pose_frame,
    window_starts,
)


@dataclass(frozen=True)
class TCNConfig:
    channels: int
    blocks: int
    dropout: float
    loss: str
    aux_weight: float

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.blocks <= 0:
            raise ValueError("TCN channels and blocks must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("TCN dropout must be in [0, 1)")
        if self.loss not in {"cross_entropy", "focal"}:
            raise ValueError("TCN loss must be cross_entropy or focal")
        if self.aux_weight < 0:
            raise ValueError("TCN auxiliary weight must be non-negative")


TCN_CANDIDATES = (
    TCNConfig(32, 3, 0.2, "cross_entropy", 0.0),
    TCNConfig(32, 3, 0.4, "focal", 0.2),
    TCNConfig(32, 4, 0.2, "cross_entropy", 0.2),
    TCNConfig(64, 3, 0.2, "cross_entropy", 0.2),
    TCNConfig(64, 3, 0.4, "focal", 0.2),
    TCNConfig(64, 4, 0.4, "focal", 0.0),
)


@dataclass
class TCNWindows:
    streams: dict[str, np.ndarray]
    labels: np.ndarray
    meta: pd.DataFrame


def concatenate_tcn_windows(items: Sequence[TCNWindows]) -> TCNWindows:
    windows = list(items)
    if not windows:
        raise ValueError("At least one TCN window collection is required")
    stream_names = set(windows[0].streams)
    if any(set(item.streams) != stream_names for item in windows):
        raise ValueError("TCN window collections have inconsistent streams")
    return TCNWindows(
        streams={
            name: np.concatenate([item.streams[name] for item in windows], axis=0)
            for name in sorted(stream_names)
        },
        labels=np.concatenate([item.labels for item in windows]),
        meta=pd.concat([item.meta for item in windows], ignore_index=True),
    )


def extract_tcn_windows(
    df: pd.DataFrame,
    subject_id: int,
    window_size: int = 150,
    stride: int = 75,
    majority_threshold: float = 0.70,
    labeled: bool = True,
) -> TCNWindows:
    if window_size <= 1 or stride <= 0:
        raise ValueError("TCN window_size must exceed one and stride must be positive")
    prepared = prepare_pose_frame(df)
    if labeled and "Action Label" not in prepared:
        raise ValueError("Labeled TCN extraction requires an Action Label column")
    normalized = normalize_for_experiment(prepared, "C")
    full_streams = build_tcn_streams(normalized)
    valid_starts: list[int] = []
    labels: list[str | None] = []
    for start in window_starts(len(prepared), window_size, stride, cover_tail=not labeled):
        label = None
        if labeled:
            label = majority_label(
                prepared["Action Label"].iloc[start : start + window_size],
                threshold=majority_threshold,
            )
            if label is None:
                continue
        valid_starts.append(start)
        labels.append(label)
    if not valid_starts:
        raise ValueError(f"Subject {subject_id} produced no TCN windows")
    window_streams = {
        name: np.stack(
            [values[:, start : start + window_size] for start in valid_starts], axis=0
        ).astype(np.float32, copy=False)
        for name, values in full_streams.items()
    }
    return TCNWindows(
        streams=window_streams,
        labels=np.asarray(labels, dtype=object),
        meta=_window_metadata(prepared, subject_id, valid_starts, window_size),
    )


def build_tcn_streams(normalized_pose: pd.DataFrame, fs: int = FS) -> dict[str, np.ndarray]:
    missing = sorted(set(COORD_COLUMNS) - set(normalized_pose.columns))
    if missing:
        raise ValueError(f"Normalized pose is missing TCN coordinates: {missing}")
    coordinates = normalized_pose[COORD_COLUMNS].to_numpy(dtype=np.float32).T
    velocity = np.diff(coordinates, axis=1, prepend=coordinates[:, :1]) * float(fs)
    bone_channels: list[np.ndarray] = []
    for _, first, second in BONES:
        dx = (
            normalized_pose[f"{second}_x"].to_numpy(dtype=np.float32)
            - normalized_pose[f"{first}_x"].to_numpy(dtype=np.float32)
        )
        dy = (
            normalized_pose[f"{second}_y"].to_numpy(dtype=np.float32)
            - normalized_pose[f"{first}_y"].to_numpy(dtype=np.float32)
        )
        bone_channels.extend((dx, dy, np.hypot(dx, dy)))
    bones = np.stack(bone_channels).astype(np.float32, copy=False)
    streams = {
        "coordinates": coordinates,
        "bones": bones,
        "velocity": velocity.astype(np.float32, copy=False),
    }
    if not all(np.isfinite(values).all() for values in streams.values()):
        raise ValueError("TCN streams contain non-finite values")
    return streams


def streams_from_coordinate_tensor(coordinates: torch.Tensor, fs: int = FS) -> dict[str, torch.Tensor]:
    if coordinates.ndim != 3 or coordinates.shape[1] != len(COORD_COLUMNS):
        raise ValueError("Coordinate tensor must have shape [batch, 34, time]")
    batch, _, frames = coordinates.shape
    points = coordinates.reshape(batch, len(JOINTS), 2, frames)
    velocity = torch.diff(coordinates, dim=2, prepend=coordinates[:, :, :1]) * float(fs)
    joint_index = {joint: index for index, joint in enumerate(JOINTS)}
    bone_channels: list[torch.Tensor] = []
    for _, first, second in BONES:
        delta = points[:, joint_index[second]] - points[:, joint_index[first]]
        length = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        bone_channels.extend((delta[:, 0:1], delta[:, 1:2], length))
    bones = torch.cat(bone_channels, dim=1)
    return {"coordinates": coordinates, "bones": bones, "velocity": velocity}


def augment_coordinate_batch(
    coordinates: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if coordinates.ndim != 3 or coordinates.shape[1] != len(COORD_COLUMNS):
        raise ValueError("Coordinate tensor must have shape [batch, 34, time]")
    batch, _, frames = coordinates.shape
    points = coordinates.clone().reshape(batch, len(JOINTS), 2, frames)

    # Temporal crop and resample retains the fixed model input length.
    resampled: list[torch.Tensor] = []
    for sample in points:
        fraction = float(torch.empty(1).uniform_(0.90, 1.0, generator=generator).item())
        crop_length = max(2, int(round(frames * fraction)))
        max_start = frames - crop_length
        start = int(torch.randint(max_start + 1, (1,), generator=generator).item())
        crop = sample[:, :, start : start + crop_length].reshape(1, len(JOINTS) * 2, crop_length)
        restored = F.interpolate(crop, size=frames, mode="linear", align_corners=False)
        resampled.append(restored.reshape(len(JOINTS), 2, frames))
    points = torch.stack(resampled)

    scale = torch.empty(batch, 1, 1, 1).uniform_(0.95, 1.05, generator=generator)
    angle = torch.empty(batch).uniform_(-np.deg2rad(10), np.deg2rad(10), generator=generator)
    cosine = torch.cos(angle).reshape(batch, 1, 1)
    sine = torch.sin(angle).reshape(batch, 1, 1)
    x = points[:, :, 0].clone()
    y = points[:, :, 1].clone()
    points[:, :, 0] = cosine * x - sine * y
    points[:, :, 1] = sine * x + cosine * y
    points *= scale

    mirror_mask = torch.rand(batch, generator=generator) < 0.5
    left_right_pairs = [
        (index, JOINTS.index(joint.replace("left_", "right_")))
        for index, joint in enumerate(JOINTS)
        if joint.startswith("left_") and joint.replace("left_", "right_") in JOINTS
    ]
    for sample_index in torch.nonzero(mirror_mask, as_tuple=False).flatten().tolist():
        mirrored = points[sample_index].clone()
        mirrored[:, 0] *= -1
        for left, right in left_right_pairs:
            mirrored[[left, right]] = mirrored[[right, left]]
        points[sample_index] = mirrored

    joint_dropout = torch.rand(batch, len(JOINTS), 1, 1, generator=generator) < 0.05
    points = points.masked_fill(joint_dropout, 0.0)
    noise = torch.randn(points.shape, generator=generator, dtype=points.dtype) * 0.005
    points = points + noise
    return points.reshape(batch, len(COORD_COLUMNS), frames)


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if gamma < 0:
        raise ValueError("Focal gamma must be non-negative")
    losses = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    true_probabilities = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
    return (((1.0 - true_probabilities) ** gamma) * losses).mean()


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.network(values))


class MultiStreamTCN(nn.Module):
    def __init__(
        self,
        stream_channels: dict[str, int],
        config: TCNConfig,
        num_classes: int,
    ) -> None:
        super().__init__()
        expected = {"coordinates", "bones", "velocity"}
        if set(stream_channels) != expected:
            raise ValueError(f"TCN stream channels must contain {sorted(expected)}")
        if num_classes <= 1:
            raise ValueError("TCN requires at least two output classes")
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv1d(channels, config.channels, kernel_size=1),
                    nn.BatchNorm1d(config.channels),
                    nn.GELU(),
                )
                for name, channels in stream_channels.items()
            }
        )
        self.blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(config.channels, dilation=2**index, dropout=config.dropout)
                for index in range(config.blocks)
            ]
        )
        self.class_head = nn.Linear(config.channels, num_classes)
        self.group_head = nn.Linear(config.channels, 2)

    def forward(self, streams: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if set(streams) != set(self.embeddings):
            raise ValueError("TCN input streams do not match configured embeddings")
        embedded = [self.embeddings[name](streams[name]) for name in self.embeddings]
        values = torch.stack(embedded, dim=0).mean(dim=0)
        values = self.blocks(values).mean(dim=-1)
        return self.class_head(values), self.group_head(values)


@dataclass
class FittedTCN:
    config: TCNConfig
    state_dict: dict[str, torch.Tensor]
    stream_channels: dict[str, int]
    classes: tuple[str, ...]
    best_epoch: int
    validation_accuracy: float

    def build_model(self, device: torch.device) -> MultiStreamTCN:
        model = MultiStreamTCN(self.stream_channels, self.config, len(self.classes))
        model.load_state_dict(self.state_dict)
        return model.to(device).eval()

    def predict_proba(
        self,
        windows: TCNWindows,
        batch_size: int = 128,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = self.build_model(selected_device)
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(windows.labels), batch_size):
                end = min(start + batch_size, len(windows.labels))
                streams = {
                    name: torch.from_numpy(values[start:end]).to(selected_device)
                    for name, values in windows.streams.items()
                }
                logits, _ = model(streams)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities)


def _encoded_labels(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    class_to_index = {label: index for index, label in enumerate(classes)}
    unknown = sorted(set(labels) - set(class_to_index))
    if unknown:
        raise ValueError(f"TCN labels contain unknown classes: {unknown}")
    return np.asarray([class_to_index[label] for label in labels], dtype=np.int64)


def train_tcn(
    train: TCNWindows,
    validation: TCNWindows,
    config: TCNConfig,
    classes: Sequence[str],
    random_state: int = 42,
    max_epochs: int = 40,
    patience: int = 6,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> FittedTCN:
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("TCN epochs, patience, and batch size must be positive")
    class_order = tuple(classes)
    train_targets = _encoded_labels(train.labels, class_order)
    validation_targets = _encoded_labels(validation.labels, class_order)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(random_state)

    stream_channels = {name: values.shape[1] for name, values in train.streams.items()}
    model = MultiStreamTCN(stream_channels, config, len(class_order)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train_targets, minlength=len(class_order)).astype(float)
    weights = np.sqrt(len(train_targets) / (len(class_order) * np.maximum(counts, 1.0)))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=selected_device)
    train_coordinates = torch.from_numpy(train.streams["coordinates"])
    targets_tensor = torch.from_numpy(train_targets)
    group_targets = torch.tensor(
        [1 if class_order[index] in {"Attacking", "Biting", "Head banging", "Throwing things"} else 0 for index in train_targets],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(random_state)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -np.inf
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(train_targets), generator=generator)
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            augmented = augment_coordinate_batch(train_coordinates[indices], generator)
            streams = {
                name: values.to(selected_device)
                for name, values in streams_from_coordinate_tensor(augmented).items()
            }
            targets = targets_tensor[indices].to(selected_device)
            groups = group_targets[indices].to(selected_device)
            class_logits, group_logits = model(streams)
            if config.loss == "focal":
                main_loss = focal_cross_entropy(
                    class_logits, targets, gamma=2.0, weight=class_weights
                )
            else:
                main_loss = F.cross_entropy(class_logits, targets, weight=class_weights)
            loss = main_loss + config.aux_weight * F.cross_entropy(group_logits, groups)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        predicted: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(validation_targets), batch_size * 2):
                end = min(start + batch_size * 2, len(validation_targets))
                streams = {
                    name: torch.from_numpy(values[start:end]).to(selected_device)
                    for name, values in validation.streams.items()
                }
                logits, _ = model(streams)
                predicted.append(logits.argmax(dim=1).cpu().numpy())
        validation_accuracy = float(
            np.mean(np.concatenate(predicted) == validation_targets)
        )
        if validation_accuracy > best_accuracy + 1e-12:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("TCN training failed to produce a checkpoint")
    return FittedTCN(
        config=config,
        state_dict=best_state,
        stream_channels=stream_channels,
        classes=class_order,
        best_epoch=best_epoch,
        validation_accuracy=best_accuracy,
    )


def train_tcn_fixed_epochs(
    train: TCNWindows,
    config: TCNConfig,
    classes: Sequence[str],
    epochs: int,
    random_state: int = 42,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> FittedTCN:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("TCN fixed epochs and batch size must be positive")
    class_order = tuple(classes)
    train_targets = _encoded_labels(train.labels, class_order)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(random_state)

    stream_channels = {name: values.shape[1] for name, values in train.streams.items()}
    model = MultiStreamTCN(stream_channels, config, len(class_order)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train_targets, minlength=len(class_order)).astype(float)
    weights = np.sqrt(len(train_targets) / (len(class_order) * np.maximum(counts, 1.0)))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=selected_device)
    coordinates = torch.from_numpy(train.streams["coordinates"])
    targets_tensor = torch.from_numpy(train_targets)
    abnormal = {"Attacking", "Biting", "Head banging", "Throwing things"}
    group_targets = torch.tensor(
        [1 if class_order[index] in abnormal else 0 for index in train_targets],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(random_state)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_targets), generator=generator)
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            augmented = augment_coordinate_batch(coordinates[indices], generator)
            streams = {
                name: values.to(selected_device)
                for name, values in streams_from_coordinate_tensor(augmented).items()
            }
            targets = targets_tensor[indices].to(selected_device)
            groups = group_targets[indices].to(selected_device)
            class_logits, group_logits = model(streams)
            if config.loss == "focal":
                main_loss = focal_cross_entropy(
                    class_logits, targets, gamma=2.0, weight=class_weights
                )
            else:
                main_loss = F.cross_entropy(class_logits, targets, weight=class_weights)
            loss = main_loss + config.aux_weight * F.cross_entropy(group_logits, groups)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    return FittedTCN(
        config=config,
        state_dict=state,
        stream_channels=stream_channels,
        classes=class_order,
        best_epoch=epochs - 1,
        validation_accuracy=float("nan"),
    )
