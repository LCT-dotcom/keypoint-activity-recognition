from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from tsfel_histgb_pipeline import WindowFeatures


def validate_unlabeled_windows(
    payload: dict[str, Any], source: Path | str = "cache payload"
) -> WindowFeatures:
    signature = payload.get("signature", {})
    windows = payload.get("windows")
    if not isinstance(windows, WindowFeatures):
        raise ValueError(f"Invalid inference cache payload: {source}")
    if signature.get("labeled") is not False or windows.y.notna().any():
        raise ValueError(
            f"Expected an unlabeled inference cache with no window labels: {source}"
        )
    return windows


def load_unlabeled_windows(path: Path) -> WindowFeatures:
    """Load an inference cache and reject any label-dependent window set."""
    return validate_unlabeled_windows(joblib.load(path), path)
