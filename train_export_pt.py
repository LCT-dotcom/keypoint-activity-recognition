from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from keypoint_pt_pipeline import (
    CLASSES,
    WindowSet,
    channel_stats,
    concatenate_window_sets,
    dump_json,
    evaluation_metrics,
    make_labeled_windows,
    predict_windows,
    save_checkpoint,
    train_model,
)


DEFAULT_DATA_DIR = Path(
    r"E:\DATA\Keypoint\Train Data-20260812T151048Z-1-001\Train Data\keypointlabel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LOSO keypoint TCN and export a portable .pt checkpoint")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--output", type=Path, default=Path("artifacts/keypoint_tcn_1_2_3_5.pt"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--fs", type=int, default=30)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--train-overlap", type=float, default=0.50)
    parser.add_argument("--majority-threshold", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--loso-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_windows(args: argparse.Namespace) -> dict[int, WindowSet]:
    window_size = int(round(args.fs * args.window_sec))
    train_stride = max(1, int(round(window_size * (1.0 - args.train_overlap))))
    windows: dict[int, WindowSet] = {}
    for subject in args.subjects:
        path = args.data_dir / f"keypoints_with_labels_{subject}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        windows[subject] = make_labeled_windows(
            frame,
            subject_id=subject,
            window_size=window_size,
            stride=train_stride,
            majority_threshold=args.majority_threshold,
        )
        counts = np.bincount(windows[subject].y, minlength=len(CLASSES))
        distribution = {CLASSES[index]: int(count) for index, count in enumerate(counts)}
        print(f"Subject {subject}: {len(frame)} frames, {len(windows[subject].y)} windows, {distribution}")
    return windows


def model_config(input_channels: int) -> dict:
    return {
        "input_channels": input_channels,
        "num_classes": len(CLASSES),
        "hidden_channels": 96,
        "dropout": 0.20,
        "dilations": [1, 2, 4, 8],
    }


def run_loso(args: argparse.Namespace, windows: dict[int, WindowSet], device: torch.device) -> dict:
    fold_metrics = []
    all_true = []
    all_pred = []
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    for held_out in args.subjects:
        print(f"\nLOSO holdout subject {held_out}")
        train = concatenate_window_sets([windows[s] for s in args.subjects if s != held_out])
        test = windows[held_out]
        mean, std = channel_stats(train.x)
        config = model_config(train.x.shape[-1])
        model = train_model(
            train,
            mean,
            std,
            config,
            epochs=args.loso_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed + held_out,
        )
        pred, _ = predict_windows(model, test.x, mean, std, device)
        metrics = evaluation_metrics(test.y, pred)
        row = {key: value for key, value in metrics.items() if key != "classification_report"}
        row["held_out_subject"] = held_out
        row["n_train_windows"] = len(train.y)
        row["n_test_windows"] = len(test.y)
        fold_metrics.append(row)
        all_true.append(test.y)
        all_pred.append(pred)
        dump_json(args.metrics_dir / f"loso_subject_{held_out}.json", metrics)
        print(
            f"  accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"abnormal_f1={metrics['abnormal_f1']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = evaluation_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    summary = {
        "folds": fold_metrics,
        "mean_fold_accuracy": float(np.mean([row["accuracy"] for row in fold_metrics])),
        "mean_fold_macro_f1": float(np.mean([row["macro_f1"] for row in fold_metrics])),
        "mean_fold_abnormal_f1": float(np.mean([row["abnormal_f1"] for row in fold_metrics])),
        "pooled": combined,
    }
    pd.DataFrame(fold_metrics).to_csv(args.metrics_dir / "loso_fold_metrics.csv", index=False)
    dump_json(args.metrics_dir / "loso_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if not 0 <= args.train_overlap < 1:
        raise ValueError("--train-overlap must be in [0, 1)")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    windows = load_windows(args)
    loso_summary = None if args.skip_loso else run_loso(args, windows, device)

    print("\nTraining final model on all supplied subjects")
    full_train = concatenate_window_sets([windows[subject] for subject in args.subjects])
    mean, std = channel_stats(full_train.x)
    config = model_config(full_train.x.shape[-1])
    final_model = train_model(
        full_train,
        mean,
        std,
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
    )

    preprocessing = {
        "fps": args.fs,
        "window_seconds": args.window_sec,
        "window_size": int(round(args.fs * args.window_sec)),
        "train_overlap": args.train_overlap,
        "train_stride": max(1, int(round(args.fs * args.window_sec * (1.0 - args.train_overlap)))),
        "inference_stride": int(round(args.fs * args.window_sec)),
        "majority_threshold": args.majority_threshold,
        "normalization": "hip midpoint center; torso length scale; shoulder-width fallback",
        "input_features": "34 normalized coordinates + 34 first-order velocities",
        "none_is_target_class": False,
        "label_aliases": {"Throwing": "Throwing things"},
    }
    training_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": args.subjects,
        "source_data_dir": str(args.data_dir),
        "n_training_windows": int(len(full_train.y)),
        "epochs": args.epochs,
        "seed": args.seed,
        "loso_summary_file": None if args.skip_loso else str(args.metrics_dir / "loso_summary.json"),
        "mean_loso_accuracy": None if loso_summary is None else loso_summary["mean_fold_accuracy"],
        "mean_loso_abnormal_f1": None if loso_summary is None else loso_summary["mean_fold_abnormal_f1"],
    }
    save_checkpoint(args.output, final_model, mean, std, config, preprocessing, training_metadata)
    print(f"\nSaved portable checkpoint: {args.output.resolve()}")
    print(f"Size: {args.output.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
