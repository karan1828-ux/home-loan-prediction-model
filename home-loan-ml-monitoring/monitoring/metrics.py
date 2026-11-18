"""Metrics collection, persistent JSON store, and monitoring utilities.

This module provides a lightweight, thread-safe JSON-backed metrics store
for tracking operational serving statistics, data drift events, automated
retraining attempts, and model promotion history.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any, Dict, List, Optional, Union

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH: Path = REPORTS_DIR / "metrics.json"


class MetricsStore:
    """Thread-safe, file-backed metrics repository tracking system health and ML lifecycle events.

    Note on concurrency:
    File persistence uses an atomic write (tempfile + os.replace) to ensure reading
    processes never observe partially written or truncated JSON files. However, atomic
    replacement alone does not resolve multi-process update races (read-modify-write
    conflicts across independent processes).
    """

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
    ) -> None:
        self._lock = Lock()
        self.filepath: Path = Path(filepath) if filepath else DEFAULT_METRICS_PATH
        self.auto_save: bool = auto_save

        # Operational serving state variables
        self.start_time: float = time.time()
        self.total_predictions: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self.approved_count: int = 0
        self.rejected_count: int = 0
        self.total_latency_ms: float = 0.0

        # Drift tracking state variables
        self.drift_events_count: int = 0
        self.last_drift_score: float = 0.0
        self.last_drift_status: str = "NO DRIFT DETECTED"
        self.last_drift_timestamp: Optional[str] = None
        self.last_drifted_features: List[str] = []

        # Retraining & model lifecycle state variables
        self.retraining_attempts: int = 0
        self.successful_promotions: int = 0
        self.current_model_version: str = "champion-v1"
        self.last_retraining_time: Optional[str] = None
        self.champion_performance: Optional[float] = None
        self.candidate_performance: Optional[float] = None
        self.retraining_history: List[Dict[str, Any]] = []

        # Ensure directory exists and load existing file if available
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            if self.filepath.exists():
                self.load_from_json()
        except Exception as exc:
            logger.warning(
                "Could not initialize metrics persistence directory or file at '%s': %s",
                self.filepath,
                exc,
            )

    def record_prediction(
        self,
        loan_status: Optional[str] = None,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record an incoming inference request event."""
        with self._lock:
            self.total_predictions += 1
            if success:
                self.successful_predictions += 1
                self.total_latency_ms += latency_ms
                if loan_status in ("Y", "1", 1, "Approved", True):
                    self.approved_count += 1
                elif loan_status in ("N", "0", 0, "Rejected", False):
                    self.rejected_count += 1
            else:
                self.failed_predictions += 1

        if self.auto_save:
            try:
                self.save_to_json()
            except Exception as exc:
                logger.error("Auto-save failed in record_prediction: %s", exc)

    def record_drift_event(
        self,
        is_drift_detected: bool,
        drift_percentage: float,
        drifted_features: List[str],
        status: Optional[str] = None,
    ) -> None:
        """Record an evaluated data drift detection event."""
        with self._lock:
            self.drift_events_count += 1
            self.last_drift_score = round(float(drift_percentage), 2)
            if status is not None:
                self.last_drift_status = str(status)
            else:
                self.last_drift_status = (
                    "DRIFT DETECTED" if is_drift_detected else "NO DRIFT DETECTED"
                )
            self.last_drift_timestamp = datetime.now(timezone.utc).isoformat()
            self.last_drifted_features = list(drifted_features)

        if self.auto_save:
            try:
                self.save_to_json()
            except Exception as exc:
                logger.error("Auto-save failed in record_drift_event: %s", exc)

    def record_retraining_event(
        self,
        success: bool,
        promoted: bool,
        champion_score: Optional[float] = None,
        candidate_score: Optional[float] = None,
        new_model_version: Optional[str] = None,
        evaluation_metric: str = "f1_score",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a retraining execution and candidate promotion outcome."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.retraining_attempts += 1
            self.last_retraining_time = timestamp
            if champion_score is not None:
                self.champion_performance = round(float(champion_score), 4)
            if candidate_score is not None:
                self.candidate_performance = round(float(candidate_score), 4)

            if promoted:
                self.successful_promotions += 1
                if new_model_version:
                    self.current_model_version = str(new_model_version)

            fallback_status = "promoted" if promoted else ("success" if success else "failed")
            record_status = (details or {}).get("status") or fallback_status

            record = {
                "timestamp": timestamp,
                "retraining_run_id": self.retraining_attempts,
                "status": str(record_status),
                "success": bool(success),
                "promoted": bool(promoted),
                "metric_name": str(evaluation_metric),
                "champion_score": (
                    round(float(champion_score), 4)
                    if champion_score is not None
                    else None
                ),
                "candidate_score": (
                    round(float(candidate_score), 4)
                    if candidate_score is not None
                    else None
                ),
                "active_model_version": self.current_model_version,
                "details": details or {},
            }
            self.retraining_history.append(record)

        if self.auto_save:
            try:
                self.save_to_json()
            except Exception as exc:
                logger.error("Auto-save failed in record_retraining_event: %s", exc)

    def get_metrics(self) -> Dict[str, Any]:
        """Compute aggregated metrics dictionary matching API and dashboard requirements."""
        with self._lock:
            uptime = max(0.0, time.time() - self.start_time)
            avg_latency = (
                (self.total_latency_ms / self.successful_predictions)
                if self.successful_predictions > 0
                else 0.0
            )
            approval_rate = (
                (self.approved_count / self.successful_predictions)
                if self.successful_predictions > 0
                else 0.0
            )
            rejection_rate = (
                (self.rejected_count / self.successful_predictions)
                if self.successful_predictions > 0
                else 0.0
            )

            return {
                "total_predictions": self.total_predictions,
                "successful_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "approved_predictions": self.approved_count,
                "rejected_predictions": self.rejected_count,
                "approval_rate": round(approval_rate, 4),
                "rejection_rate": round(rejection_rate, 4),
                "average_latency_ms": round(avg_latency, 2),
                "uptime_seconds": round(uptime, 2),
                "drift_events_count": self.drift_events_count,
                "last_drift_score": self.last_drift_score,
                "last_drift_status": self.last_drift_status,
                "last_drift_timestamp": self.last_drift_timestamp,
                "last_drifted_features": list(self.last_drifted_features),
                "retraining_attempts": self.retraining_attempts,
                "successful_promotions": self.successful_promotions,
                "current_model_version": self.current_model_version,
                "last_retraining_time": self.last_retraining_time,
                "champion_performance": self.champion_performance,
                "candidate_performance": self.candidate_performance,
                "retraining_history": list(self.retraining_history[-10:]),
            }

    def save_to_json(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persist the current state to a JSON file atomically using a temporary file.

        Writes to a temporary file on the same filesystem/directory and replaces
        the target file via os.replace to prevent partial-write reads.
        """
        target_path = Path(filepath) if filepath else self.filepath
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create directory for '%s': %s", target_path, exc)
            return target_path

        data = self.get_metrics()
        # Include full retraining history in persistent storage
        with self._lock:
            data["retraining_history"] = list(self.retraining_history)

        temp_file_path: Optional[str] = None
        try:
            # Create temporary file in the target directory to ensure same filesystem for os.replace
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(target_path.parent),
                prefix="metrics_",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tf:
                temp_file_path = tf.name
                json.dump(data, tf, indent=2)

            os.replace(temp_file_path, str(target_path))
        except Exception as exc:
            logger.error(
                "Failed to persist metrics to JSON '%s': %s", target_path, exc
            )
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

        return target_path

    def load_from_json(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """Restore metrics from a JSON file safely handling missing, empty, or corrupted files."""
        target_path = Path(filepath) if filepath else self.filepath
        if not target_path.exists():
            return False

        try:
            if target_path.stat().st_size == 0:
                logger.warning("Metrics file '%s' is empty; skipping load.", target_path)
                return False

            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning(
                    "Metrics file '%s' did not contain a valid JSON object.", target_path
                )
                return False

            with self._lock:
                self.total_predictions = data.get("total_predictions", 0)
                self.successful_predictions = data.get("successful_predictions", 0)
                self.failed_predictions = data.get("failed_predictions", 0)
                self.approved_count = data.get("approved_predictions", 0)
                self.rejected_count = data.get("rejected_predictions", 0)
                self.drift_events_count = data.get("drift_events_count", 0)
                self.last_drift_score = data.get("last_drift_score", 0.0)
                self.last_drift_status = data.get("last_drift_status", "NO DRIFT DETECTED")
                self.last_drift_timestamp = data.get("last_drift_timestamp")
                self.last_drifted_features = data.get("last_drifted_features", [])
                self.retraining_attempts = data.get("retraining_attempts", 0)
                self.successful_promotions = data.get("successful_promotions", 0)
                self.current_model_version = data.get("current_model_version", "champion-v1")
                self.last_retraining_time = data.get("last_retraining_time")
                self.champion_performance = data.get("champion_performance")
                self.candidate_performance = data.get("candidate_performance")
                self.retraining_history = data.get("retraining_history", [])

            return True
        except (json.JSONDecodeError, OSError, Exception) as exc:
            logger.warning("Could not read metrics from '%s': %s", target_path, exc)
            return False

    def reset(self) -> None:
        """Reset state variables to initial demonstration defaults."""
        with self._lock:
            self.start_time = time.time()
            self.total_predictions = 0
            self.successful_predictions = 0
            self.failed_predictions = 0
            self.approved_count = 0
            self.rejected_count = 0
            self.total_latency_ms = 0.0
            self.drift_events_count = 0
            self.last_drift_score = 0.0
            self.last_drift_status = "NO DRIFT DETECTED"
            self.last_drift_timestamp = None
            self.last_drifted_features = []
            self.retraining_attempts = 0
            self.successful_promotions = 0
            self.current_model_version = "champion-v1"
            self.last_retraining_time = None
            self.champion_performance = None
            self.candidate_performance = None
            self.retraining_history = []

        if self.auto_save and self.filepath.exists():
            try:
                self.save_to_json()
            except Exception as exc:
                logger.error("Auto-save failed in reset: %s", exc)


# Backward-compatible alias
PredictionMetricsTracker = MetricsStore

# Global singleton metrics store instance
metrics_tracker = MetricsStore()
