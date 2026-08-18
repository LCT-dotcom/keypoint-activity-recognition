from __future__ import annotations

import argparse
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from abc_experiment_pipeline import extract_experiment_unlabeled_windows
from phase_d_evaluation import aggregate_window_probabilities, viterbi_decode


def predict(model_path: Path, input_path: Path, output_path: Path, participant_id: int) -> Path:
    artifact = joblib.load(model_path)
    frame = pd.read_csv(input_path)
    preprocessing = artifact["preprocessing"]
    classes = tuple(artifact["classes"])
    windows = extract_experiment_unlabeled_windows(
        frame,
        subject_id=participant_id,
        experiment=preprocessing["experiment"],
        window_size=int(preprocessing["window_size"]),
        stride=int(preprocessing["inference_stride"]),
    )
    window_probabilities = artifact["model"].predict_proba(windows.x, classes)
    frame_result = aggregate_window_probabilities(
        windows.meta, window_probabilities, n_frames=len(frame), classes=classes
    )
    decoder = artifact.get("decoder", {})
    strength = float(decoder.get("transition_strength", 0.0))
    if strength > 0.0:
        indices = viterbi_decode(
            frame_result.probabilities,
            np.asarray(decoder["transition_probabilities"], dtype=float),
            np.asarray(decoder["initial_probabilities"], dtype=float),
            strength=strength,
        )
        labels = np.asarray(classes, dtype=object)[indices]
    else:
        labels = frame_result.labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    result["predicted_label"] = labels
    result["prediction_confidence"] = frame_result.probabilities.max(axis=1)
    result.to_csv(output_path, index=False)

    timestamp = frame["frame_id"] if "frame_id" in frame else np.arange(len(frame))
    submission = pd.DataFrame(
        {
            "participant_id": participant_id,
            "timestamp": timestamp,
            "predicted_label": labels,
        }
    )
    submission.to_csv(output_path.with_name(f"{output_path.stem}_submission.csv"), index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a new keypoint CSV with Phase D.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--participant-id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(predict(args.model, args.input, args.output, args.participant_id).resolve())
