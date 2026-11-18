"""Configuration settings and environment variable definitions."""

import os
from pathlib import Path

# Base project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REFERENCE_DATA_DIR: Path = DATA_DIR / "reference"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# MLflow tracking and model registry settings
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME: str = os.getenv("MODEL_NAME", "HomeLoanClassifier")
MODEL_ALIAS: str = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_MODEL_URI: str = os.getenv(
    "MLFLOW_MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)

# FastAPI application settings
API_TITLE: str = "Home Loan Prediction Service"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Production-grade FastAPI prediction service for Home Loan Approvals. "
    "Loads the champion model registered via MLflow, validates input payloads "
    "using Pydantic, monitors operational metrics, and provides real-time inference."
)
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
