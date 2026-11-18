"""FastAPI application entrypoint and inference endpoints."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from api.schemas import (
    AppInfoResponse,
    BatchPredictionRequest,
    HealthResponse,
    LoanApplication,
    MetricsResponse,
    PredictionResponse,
    ReloadModelResponse,
)
from src.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    MODEL_ALIAS,
    MODEL_NAME,
)
from src.predict import ModelNotLoadedError, predictor
from monitoring.metrics import metrics_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api.main")

_APP_START_TIME: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown routines."""
    logger.info("Initializing FastAPI Prediction Service...")
    try:
        predictor.load_model()
        logger.info(
            "Champion model '%s' (alias '%s') loaded on startup.",
            predictor.model_name,
            predictor.model_alias,
        )
    except Exception as exc:
        logger.warning(
            "Champion model could not be loaded on startup: %s. "
            "Service will start in degraded mode. "
            "Use POST /reload-model once the model is registered in MLflow.",
            exc,
        )
    yield
    logger.info("Shutting down FastAPI Prediction Service.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


@app.get(
    "/",
    response_model=AppInfoResponse,
    summary="Basic application information",
    tags=["System"],
)
def get_app_info() -> AppInfoResponse:
    """Return basic metadata, application status, and available service endpoints."""
    return AppInfoResponse(
        name=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        status="online",
        model_name=predictor.model_name,
        model_alias=predictor.model_alias,
        endpoints={
            "root": "GET /",
            "health": "GET /health",
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "metrics": "GET /metrics",
            "reload_model": "POST /reload-model",
            "documentation": "GET /docs",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="API health status",
    tags=["System"],
)
def get_health() -> HealthResponse:
    """Return service health status and current champion model state."""
    model_status = predictor.get_status()
    is_loaded = model_status["model_loaded"]
    uptime = max(0.0, time.time() - _APP_START_TIME)

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=predictor.model_name,
        model_alias=predictor.model_alias,
        model_version=predictor.model_version,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict loan approval",
    tags=["Inference"],
)
def predict(application: LoanApplication) -> PredictionResponse:
    """Score a single loan application and predict loan approval outcome.

    Validates applicant details using Pydantic, passes features to the active
    MLflow champion model, and returns predicted status along with confidence probability.
    """
    try:
        prediction_result = predictor.predict(application)
        return PredictionResponse(**prediction_result)
    except ModelNotLoadedError as err:
        logger.error("Predict endpoint rejected request: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is currently unavailable. Ensure the champion model is "
                "registered in MLflow and accessible, or trigger POST /reload-model."
            ),
        ) from err
    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed due to an internal error: {exc}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    summary="Batch loan approval prediction",
    tags=["Inference"],
)
def predict_batch(batch_request: BatchPredictionRequest) -> List[PredictionResponse]:
    """Score multiple loan applications in a single batch request."""
    responses: List[PredictionResponse] = []
    for app_item in batch_request.applications:
        try:
            res = predictor.predict(app_item)
            responses.append(PredictionResponse(**res))
        except ModelNotLoadedError as err:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(err),
            ) from err
        except Exception as exc:
            logger.exception("Batch item prediction error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch inference failed on item: {exc}",
            ) from exc
    return responses


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Return monitoring metrics",
    tags=["Monitoring"],
)
def get_metrics() -> MetricsResponse:
    """Return operational serving metrics, request counts, approval rates, and latency."""
    raw_metrics = metrics_tracker.get_metrics()
    return MetricsResponse(
        total_predictions=raw_metrics["total_predictions"],
        successful_predictions=raw_metrics["successful_predictions"],
        failed_predictions=raw_metrics["failed_predictions"],
        approved_predictions=raw_metrics["approved_predictions"],
        rejected_predictions=raw_metrics["rejected_predictions"],
        approval_rate=raw_metrics["approval_rate"],
        rejection_rate=raw_metrics["rejection_rate"],
        average_latency_ms=raw_metrics["average_latency_ms"],
        uptime_seconds=raw_metrics["uptime_seconds"],
        model_name=predictor.model_name,
        model_alias=predictor.model_alias,
        model_loaded=predictor.is_loaded,
    )


@app.post(
    "/reload-model",
    response_model=ReloadModelResponse,
    summary="Reload the current champion model",
    tags=["Model Registry"],
)
def reload_model() -> ReloadModelResponse:
    """Reload the latest champion model from the MLflow Model Registry."""
    try:
        predictor.reload()
        return ReloadModelResponse(
            status="success",
            message=(
                f"Champion model '{predictor.model_name}' (alias: '{predictor.model_alias}') "
                "reloaded successfully from MLflow registry."
            ),
            model_name=predictor.model_name,
            model_alias=predictor.model_alias,
            model_version=predictor.model_version,
            reloaded_at=predictor.last_loaded_at or datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error("Champion model reload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to reload champion model from MLflow: {exc}",
        ) from exc
