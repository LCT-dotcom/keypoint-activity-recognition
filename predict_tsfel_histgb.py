from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsfel_histgb_pipeline import load_artifact, predict_frame_labels, write_prediction_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict an unseen keypoint CSV with a saved TSFEL + HistGradientBoosting model."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tsfel_histgb"))
    parser.add_argument("--participant-id", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.model)
    frame = pd.read_csv(args.input)
    prediction = predict_frame_labels(frame, artifact, args.participant_id)
    paths = write_prediction_outputs(args.output_dir, frame, prediction, args.participant_id)
    print(pd.Series(prediction.frame_labels).value_counts().sort_index().to_string())
    print(f"Filled CSV: {paths.filled.resolve()}")
    print(f"Submission CSV: {paths.submission.resolve()}")


if __name__ == "__main__":
    main()
