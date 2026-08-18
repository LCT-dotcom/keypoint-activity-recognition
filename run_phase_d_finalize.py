from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase_d_runner import select_phase_d_winner
from run_phase_d_d1 import _candidate_from_artifacts


CANDIDATES = (
    ("D0u-C-unlabeled-inference", "d0u", True),
    ("D1u-regularized-C-unlabeled-inference", "d1u", False),
    ("D2-multiscale-geometry", "d2", False),
    ("D3u-D1u-soft-gate-viterbi", "d3u", False),
    ("D4-shallow-multistream-TCN", "d4", False),
)


def finalize(phase_dir: Path) -> dict:
    contenders = []
    for name, prefix, baseline in CANDIDATES:
        fold_metrics = pd.read_csv(phase_dir / f"{prefix}_fold_metrics.csv")
        summary = json.loads(
            (phase_dir / f"{prefix}_summary.json").read_text(encoding="utf-8")
        )
        contenders.append(
            _candidate_from_artifacts(name, fold_metrics, summary, baseline)
        )
    winner = select_phase_d_winner(contenders)
    result = {
        "winner": winner.name,
        "selection_protocol": {
            "development_subjects": [1, 2, 3, 5],
            "held_out_windowing": "unlabeled_continuous",
            "abnormal_f1_tolerance": 0.005,
            "maximum_per_subject_accuracy_loss": 0.05,
            "ranking": [
                "pooled_accuracy",
                "worst_subject_accuracy",
                "pooled_macro_f1",
                "pooled_abnormal_f1",
            ],
            "s4_used_for_selection": False,
        },
        "candidates": {
            candidate.name: {
                "eligible": candidate.eligible,
                "eligibility_reasons": candidate.eligibility_reasons,
                "pooled_accuracy": candidate.pooled_accuracy,
                "pooled_macro_f1": candidate.pooled_macro_f1,
                "pooled_abnormal_f1": candidate.pooled_abnormal_f1,
                "worst_subject_accuracy": candidate.worst_subject_accuracy,
                "subject_accuracies": candidate.subject_accuracies,
            }
            for candidate in contenders
        },
    }
    (phase_dir / "final_comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the leakage-free Phase D winner on S1/S2/S3/S5."
    )
    parser.add_argument("--phase-dir", type=Path, default=Path("artifacts/phase_d"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(finalize(args.phase_dir), indent=2))
