from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from build_phase_d_multiscale_cache import SUBJECTS, base_cache_path
from phase_d_classical import phase_d_parameter_candidates
from phase_d_d2 import run_resumable_d2_pair_search, top_d1_candidate_indices
from phase_d_multiscale import MULTISCALE_WINDOWS
from phase_d_search import select_outer_configs_from_pair_scores
from tsfel_histgb_pipeline import clean_labels


def parse_pairs(value: str | None) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    return [tuple(map(int, pair.split("_"))) for pair in value.split(",")]


def run(
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    random_state: int,
    force: bool,
    train_pairs: list[tuple[int, int]] | None = None,
) -> dict:
    candidates = phase_d_parameter_candidates(random_state)
    d1_scores = pd.read_csv(output_dir.parent / "d1_pair_search" / "pair_search_scores.csv")
    d1_selections, d1_audit = select_outer_configs_from_pair_scores(
        d1_scores, SUBJECTS, candidates
    )
    allowed = {
        subject: top_d1_candidate_indices(selection.inner_metrics, top_k=2)
        for subject, selection in d1_selections.items()
    }
    base_windows = {
        subject: {
            scale: joblib.load(base_cache_path(cache_dir, subject, scale))["windows"]
            for scale in MULTISCALE_WINDOWS
        }
        for subject in SUBJECTS
    }
    frame_truth = {
        subject: clean_labels(
            pd.read_csv(data_dir / f"keypoints_with_labels_{subject}.csv")["Action Label"]
        ).to_numpy(dtype=object)
        for subject in SUBJECTS
    }
    scores, audit = run_resumable_d2_pair_search(
        base_windows,
        frame_truth,
        candidates,
        allowed,
        output_dir,
        random_state=random_state,
        force=force,
        train_pairs=train_pairs,
    )
    result = {
        "d1_selection_audit": d1_audit,
        "allowed_candidates_by_outer": {str(key): value for key, value in allowed.items()},
        "search_audit": audit,
        "score_rows": len(scores),
    }
    suffix = "" if train_pairs is None else "_" + "__".join(
        "_".join(map(str, pair)) for pair in train_pairs
    )
    (output_dir / f"d2_search_manifest{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable leakage-safe D2 pair search.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/phase_d/compact_cache")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/phase_d/d2_pair_search")
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--pairs",
        help="Optional comma-separated training pairs, for example 1_2,1_3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(
                args.data_dir,
                args.cache_dir,
                args.output_dir,
                args.random_state,
                args.force,
                parse_pairs(args.pairs),
            ),
            indent=2,
        )
    )
