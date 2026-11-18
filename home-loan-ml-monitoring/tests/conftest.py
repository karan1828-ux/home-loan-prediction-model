"""Test configuration and shared fixtures.

Provides test-level isolation for the global metrics store so automated test runs
never mutate production metrics persistence files.
"""

from pathlib import Path
import pytest

from monitoring.metrics import metrics_tracker


@pytest.fixture(autouse=True)
def isolate_metrics_tracker(tmp_path: Path):
    """Isolate global metrics_tracker to a temporary JSON file for every test."""
    orig_path = metrics_tracker.filepath
    orig_autosave = metrics_tracker.auto_save
    test_file = tmp_path / "test_metrics.json"

    metrics_tracker.filepath = test_file
    metrics_tracker.auto_save = True
    metrics_tracker.reset()

    try:
        yield
    finally:
        # Disable auto_save before resetting state to prevent redundant disk writes
        metrics_tracker.auto_save = False
        metrics_tracker.reset()
        if test_file.exists():
            try:
                test_file.unlink()
            except OSError:
                pass
        # Restore original filepath and auto_save settings
        metrics_tracker.filepath = orig_path
        metrics_tracker.auto_save = orig_autosave

