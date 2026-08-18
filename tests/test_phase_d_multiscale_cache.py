from pathlib import Path

from build_phase_d_multiscale_cache import base_cache_path


def test_base_compact_cache_paths_are_unique_by_subject_and_scale() -> None:
    root = Path("cache")
    paths = {
        base_cache_path(root, subject, scale)
        for subject in (1, 2, 3, 5)
        for scale in (60, 150, 300)
    }

    assert len(paths) == 12
    assert all("unlabeled" in path.name for path in paths)
