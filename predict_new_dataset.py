from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from keypoint_pt_pipeline import (
    CLASSES,
    clean_labels,
    evaluation_metrics,
    load_checkpoint,
    pad_window,
    pose_features,
    predict_windows,
    window_starts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict eight activities for a completely new keypoint CSV")
    parser.add_argument("--model", type=Path, required=True, help="Path to exported .pt checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="New keypoint CSV")
    parser.add_argument("--output", type=Path, default=Path("submission_predictions.csv"))
    parser.add_argument("--participant-id", default="unseen_participant")
    parser.add_argument("--participant-column", default=None)
    parser.add_argument("--timestamp-column", default=None)
    parser.add_argument("--group-column", default=None, help="Optional sample/segment id column for pre-segmented data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-confidence", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def choose_column(df: pd.DataFrame, explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested column '{explicit}' does not exist")
        return explicit
    return next((column for column in candidates if column in df.columns), None)


def make_windows(features: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, list[int]]:
    starts = window_starts(len(features), window_size, stride, cover_tail=True)
    windows = [pad_window(features[start : start + window_size], window_size) for start in starts]
    return np.stack(windows).astype(np.float32), starts


def predict_continuous(
    df: pd.DataFrame,
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = pose_features(df)
    window_size = int(checkpoint["preprocessing"]["window_size"])
    stride = int(checkpoint["preprocessing"]["inference_stride"])
    windows, starts = make_windows(features, window_size, stride)
    _, probabilities = predict_windows(
        model,
        windows,
        checkpoint["channel_mean"].numpy(),
        checkpoint["channel_std"].numpy(),
        device,
        batch_size,
    )

    frame_scores = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    frame_votes = np.zeros(len(df), dtype=np.float32)
    for index, start in enumerate(starts):
        end = min(start + window_size, len(df))
        frame_scores[start:end] += probabilities[index]
        frame_votes[start:end] += 1.0
    frame_scores /= np.maximum(frame_votes[:, None], 1.0)
    pred = frame_scores.argmax(axis=1)
    confidence = frame_scores.max(axis=1)
    return pred, confidence


def evaluate_optional_labels(df: pd.DataFrame, pred: np.ndarray) -> None:
    if "Action Label" not in df.columns:
        return
    labels = clean_labels(df["Action Label"])
    mask = labels.isin(CLASSES).to_numpy()
    if not mask.any():
        print("Input contains no scorable labels among the eight target classes")
        return
    y_true = np.asarray([CLASSES.index(label) for label in labels[mask]], dtype=np.int64)
    metrics = evaluation_metrics(y_true, pred[mask])
    print(
        f"Evaluation on labeled rows: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, abnormal_f1={metrics['abnormal_f1']:.4f}, "
        f"abnormal_precision={metrics['abnormal_precision']:.4f}, "
        f"abnormal_recall={metrics['abnormal_recall']:.4f}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint = load_checkpoint(args.model, device)
    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input CSV is empty")

    participant_column = choose_column(
        df,
        args.participant_column,
        ["participant_id", "Participant", "participant", "subject_id", "Subject", "ID"],
    )
    timestamp_column = choose_column(
        df,
        args.timestamp_column,
        ["timestamp", "Timestamp", "time", "frame_id"],
    )

    if args.group_column:
        if args.group_column not in df.columns:
            raise ValueError(f"Group column '{args.group_column}' does not exist")
        output_rows = []
        for group_value, group in df.groupby(args.group_column, sort=False, dropna=False):
            pred, confidence = predict_continuous(group.reset_index(drop=True), model, checkpoint, device, args.batch_size)
            counts = np.bincount(pred, minlength=len(CLASSES))
            label_index = int(counts.argmax())
            row = {
                "participant_id": group[participant_column].iloc[0] if participant_column else args.participant_id,
                "timestamp": group[timestamp_column].iloc[0] if timestamp_column else group_value,
                "predicted_label": CLASSES[label_index],
            }
            if args.include_confidence:
                row["confidence"] = float(np.mean(confidence[pred == label_index]))
            output_rows.append(row)
        submission = pd.DataFrame(output_rows)
    else:
        pred, confidence = predict_continuous(df, model, checkpoint, device, args.batch_size)
        evaluate_optional_labels(df, pred)
        if participant_column:
            participant = df[participant_column].to_numpy()
        else:
            participant = np.full(len(df), args.participant_id, dtype=object)
        if timestamp_column:
            timestamp = df[timestamp_column].to_numpy()
        else:
            fps = float(checkpoint["preprocessing"]["fps"])
            timestamp = np.arange(len(df), dtype=float) / fps
        submission = pd.DataFrame(
            {
                "participant_id": participant,
                "timestamp": timestamp,
                "predicted_label": [CLASSES[index] for index in pred],
            }
        )
        if args.include_confidence:
            submission["confidence"] = confidence

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Model: {args.model.resolve()}")
    print(f"Input: {args.input.resolve()} ({len(df)} rows)")
    print(f"Saved: {args.output.resolve()} ({len(submission)} predictions)")
    print("Predicted distribution:")
    print(submission["predicted_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
