from __future__ import annotations

import pandas as pd
import pytest

from phase_d_protocol import validate_unlabeled_windows
from tsfel_histgb_pipeline import WindowFeatures


def _payload(*, labeled: bool, labels: list[object]) -> dict:
    windows = WindowFeatures(
        x=pd.DataFrame({"feature": [1.0, 2.0]}),
        y=pd.Series(labels, name="Action Label", dtype="object"),
        meta=pd.DataFrame(
            {
                "subject_id": [1, 1],
                "start": [0, 75],
                "end": [150, 225],
                "center": [75, 150],
                "label": [None, None],
            }
        ),
    )
    return {"signature": {"labeled": labeled}, "windows": windows}


def test_validate_unlabeled_windows_accepts_label_free_cache() -> None:
    windows = validate_unlabeled_windows(_payload(labeled=False, labels=[None, None]))

    assert windows.x.shape == (2, 1)


@pytest.mark.parametrize(
    ("labeled", "labels"),
    [(True, [None, None]), (False, ["Walking", None])],
)
def test_validate_unlabeled_windows_rejects_label_dependent_cache(
    labeled: bool, labels: list[object]
) -> None:
    with pytest.raises(ValueError, match="unlabeled inference"):
        validate_unlabeled_windows(_payload(labeled=labeled, labels=labels))
