from __future__ import annotations

import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASSES = [
    "Attacking",
    "Biting",
    "Eating snacks",
    "Head banging",
    "Sitting quietly",
    "Throwing things",
    "Using phone",
    "Walking",
]

ABNORMAL_CLASSES = {"Attacking", "Biting", "Head banging", "Throwing things"}

LABEL_ALIASES = {
    "Throwing": "Throwing things",
    "Throwing object": "Throwing things",
    "Throwing objects": "Throwing things",
    "Biting Nails": "Biting",
    "Biting nails": "Biting",
    "Head-Banging": "Head banging",
    "Sitting Quietly": "Sitting quietly",
    "Eating Snacks": "Eating snacks",
    "Using Phone": "Using phone",
}

JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COORD_COLUMNS = [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y")]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_labels(values: pd.Series) -> pd.Series:
    labels = values.astype("string").str.strip().replace(LABEL_ALIASES)
    labels = labels.where(labels.isin(CLASSES), "None")
    return labels.fillna("None")


def validate_pose_columns(df: pd.DataFrame) -> None:
    missing = [column for column in COORD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} pose columns: {missing}")


def prepare_pose_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_pose_columns(df)
    out = df.copy()
    out[COORD_COLUMNS] = out[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out[COORD_COLUMNS] = out[COORD_COLUMNS].interpolate(limit_direction="both").ffill().bfill()
    if out[COORD_COLUMNS].isna().any().any():
        bad = out[COORD_COLUMNS].columns[out[COORD_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Pose columns contain only missing values: {bad}")
    if "Action Label" in out.columns:
        out["Action Label"] = clean_labels(out["Action Label"])
    return out


def pose_features(df: pd.DataFrame) -> np.ndarray:
    """Return hip-centered, torso-scaled coordinates plus first-order velocity."""
    frame = prepare_pose_frame(df)
    coords = frame[COORD_COLUMNS].to_numpy(np.float32).reshape(-1, len(JOINTS), 2)

    left_hip = JOINTS.index("left_hip")
    right_hip = JOINTS.index("right_hip")
    left_shoulder = JOINTS.index("left_shoulder")
    right_shoulder = JOINTS.index("right_shoulder")

    hip_mid = (coords[:, left_hip] + coords[:, right_hip]) / 2.0
    shoulder_mid = (coords[:, left_shoulder] + coords[:, right_shoulder]) / 2.0
    torso = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    shoulder_width = np.linalg.norm(coords[:, left_shoulder] - coords[:, right_shoulder], axis=1)
    scale = np.where(torso > 1e-5, torso, shoulder_width)
    finite_scale = scale[np.isfinite(scale) & (scale > 1e-5)]
    fallback = float(np.median(finite_scale)) if len(finite_scale) else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, fallback)

    normalized = (coords - hip_mid[:, None, :]) / scale[:, None, None]
    normalized = normalized.reshape(len(frame), -1)
    velocity = np.diff(normalized, axis=0, prepend=normalized[[0]])
    features = np.concatenate([normalized, velocity], axis=1).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def window_starts(length: int, window_size: int, stride: int, cover_tail: bool = False) -> list[int]:
    if length <= 0:
        return []
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    tail = length - window_size
    if cover_tail and starts[-1] != tail:
        starts.append(tail)
    return starts


def pad_window(array: np.ndarray, window_size: int) -> np.ndarray:
    if len(array) >= window_size:
        return array[:window_size]
    if len(array) == 0:
        raise ValueError("Cannot pad an empty sequence")
    pad = np.repeat(array[[-1]], window_size - len(array), axis=0)
    return np.concatenate([array, pad], axis=0)


def majority_label(labels: Iterable[str], threshold: float = 0.70) -> str | None:
    values = list(labels)
    valid = [value for value in values if value in CLASSES]
    if not values or not valid:
        return None
    label, count = collections.Counter(valid).most_common(1)[0]
    return label if count / len(values) >= threshold else None


@dataclass
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def make_labeled_windows(
    df: pd.DataFrame,
    subject_id: int,
    window_size: int,
    stride: int,
    majority_threshold: float = 0.70,
) -> WindowSet:
    frame = prepare_pose_frame(df)
    if "Action Label" not in frame.columns:
        raise ValueError("Training CSV must contain an 'Action Label' column")

    features = pose_features(frame)
    labels = frame["Action Label"].to_numpy(object)
    windows: list[np.ndarray] = []
    targets: list[int] = []

    for start in window_starts(len(frame), window_size, stride):
        end = min(start + window_size, len(frame))
        label = majority_label(labels[start:end], majority_threshold)
        if label is None:
            continue
        windows.append(pad_window(features[start:end], window_size))
        targets.append(CLASSES.index(label))

    if not windows:
        raise ValueError(f"Subject {subject_id} produced no valid labeled windows")
    x = np.stack(windows).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    subjects = np.full(len(y), subject_id, dtype=np.int64)
    return WindowSet(x=x, y=y, subjects=subjects)


def concatenate_window_sets(items: list[WindowSet]) -> WindowSet:
    return WindowSet(
        x=np.concatenate([item.x for item in items]),
        y=np.concatenate([item.y for item in items]),
        subjects=np.concatenate([item.subjects for item in items]),
    )


def channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


class PoseWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.x = x
        self.y = y
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = (self.x[index] - self.mean) / self.std
        if self.augment:
            if np.random.random() < 0.5:
                sample = sample + np.random.normal(0.0, 0.012, sample.shape).astype(np.float32)
            if np.random.random() < 0.3:
                scale = np.float32(np.random.uniform(0.92, 1.08))
                sample = sample * scale
        tensor = torch.from_numpy(np.ascontiguousarray(sample.T, dtype=np.float32))
        return tensor, torch.tensor(self.y[index], dtype=torch.long)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel_size = 5
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class KeypointTCN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_channels: int = 96,
        dropout: float = 0.20,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualTemporalBlock(hidden_channels, dilation, dropout) for dilation in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.head(pooled)


def make_model(model_config: dict) -> KeypointTCN:
    return KeypointTCN(
        input_channels=int(model_config["input_channels"]),
        num_classes=int(model_config["num_classes"]),
        hidden_channels=int(model_config.get("hidden_channels", 96)),
        dropout=float(model_config.get("dropout", 0.20)),
        dilations=tuple(model_config.get("dilations", [1, 2, 4, 8])),
    )


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    train_windows: WindowSet,
    mean: np.ndarray,
    std: np.ndarray,
    model_config: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> KeypointTCN:
    set_seed(seed)
    dataset = PoseWindowDataset(train_windows.x, train_windows.y, mean, std, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = make_model(model_config).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_windows.y, len(CLASSES)).to(device), label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(targets)
        scheduler.step()
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch {epoch:02d}/{epochs}: loss={total_loss/seen:.4f}, train_accuracy={correct/seen:.4f}")
    return model


@torch.inference_mode()
def predict_windows(
    model: nn.Module,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    dummy_y = np.zeros(len(x), dtype=np.int64)
    dataset = PoseWindowDataset(x, dummy_y, mean, std, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    logits = []
    for inputs, _ in loader:
        logits.append(model(inputs.to(device)).cpu())
    all_logits = torch.cat(logits).numpy()
    probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    return all_logits.argmax(axis=1), probabilities


def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_labels = [CLASSES[index] for index in y_true]
    pred_labels = [CLASSES[index] for index in y_pred]
    true_abnormal = np.asarray([label in ABNORMAL_CLASSES for label in true_labels], dtype=int)
    pred_abnormal = np.asarray([label in ABNORMAL_CLASSES for label in pred_labels], dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "abnormal_f1": float(f1_score(true_abnormal, pred_abnormal, zero_division=0)),
        "abnormal_precision": float(precision_score(true_abnormal, pred_abnormal, zero_division=0)),
        "abnormal_recall": float(recall_score(true_abnormal, pred_abnormal, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    model_config: dict,
    preprocessing: dict,
    training_metadata: dict,
) -> None:
    checkpoint = {
        "format_version": 1,
        "architecture": "KeypointTCN",
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model_config,
        "classes": CLASSES,
        "abnormal_classes": sorted(ABNORMAL_CLASSES),
        "coordinate_columns": COORD_COLUMNS,
        "feature_names": [f"normalized_{column}" for column in COORD_COLUMNS]
        + [f"velocity_{column}" for column in COORD_COLUMNS],
        "channel_mean": torch.from_numpy(mean),
        "channel_std": torch.from_numpy(std),
        "preprocessing": preprocessing,
        "training_metadata": training_metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[KeypointTCN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "KeypointTCN":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.get('architecture')}")
    if checkpoint.get("classes") != CLASSES:
        raise ValueError("Checkpoint class order does not match this inference code")
    model = make_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
