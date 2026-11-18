"""Pydantic request and response schemas for API endpoints."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    """Pydantic schema representing a single loan applicant profile."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        json_schema_extra={
            "example": {
                "Loan_ID": "LP001002",
                "Gender": "Male",
                "Married": "No",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5849.0,
                "CoapplicantIncome": 0.0,
                "LoanAmount": 128.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": "Urban",
            }
        },
    )

    Loan_ID: Optional[str] = Field(
        None, alias="loan_id", description="Unique loan application identifier"
    )
    Gender: Optional[str] = Field(
        None, alias="gender", description="Applicant gender ('Male', 'Female')"
    )
    Married: Optional[str] = Field(
        None, alias="married", description="Applicant marital status ('Yes', 'No')"
    )
    Dependents: Optional[Union[str, int]] = Field(
        None,
        alias="dependents",
        description="Number of dependents ('0', '1', '2', '3+')",
    )
    Education: Optional[str] = Field(
        None,
        alias="education",
        description="Applicant education level ('Graduate', 'Not Graduate')",
    )
    Self_Employed: Optional[str] = Field(
        None,
        alias="self_employed",
        description="Self-employment status ('Yes', 'No')",
    )
    ApplicantIncome: float = Field(
        ...,
        ge=0,
        alias="applicant_income",
        description="Primary applicant income (must be non-negative)",
    )
    CoapplicantIncome: Optional[float] = Field(
        0.0,
        ge=0,
        alias="coapplicant_income",
        description="Co-applicant income (must be non-negative)",
    )
    LoanAmount: Optional[float] = Field(
        None,
        ge=0,
        alias="loan_amount",
        description="Loan amount in thousands (must be non-negative if provided)",
    )
    Loan_Amount_Term: Optional[float] = Field(
        360.0,
        gt=0,
        alias="loan_amount_term",
        description="Loan term duration in months (must be greater than 0)",
    )
    Credit_History: Optional[float] = Field(
        None,
        alias="credit_history",
        description="Credit history indicator (1.0 for guidelines met, 0.0 for guidelines not met)",
    )
    Property_Area: Optional[str] = Field(
        None,
        alias="property_area",
        description="Property geographic area ('Urban', 'Semiurban', 'Rural')",
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert input data into a dictionary matching the model feature columns."""
        return {
            "Gender": self.Gender,
            "Married": self.Married,
            "Dependents": str(self.Dependents) if self.Dependents is not None else None,
            "Education": self.Education,
            "Self_Employed": self.Self_Employed,
            "ApplicantIncome": float(self.ApplicantIncome),
            "CoapplicantIncome": (
                float(self.CoapplicantIncome)
                if self.CoapplicantIncome is not None
                else 0.0
            ),
            "LoanAmount": (
                float(self.LoanAmount) if self.LoanAmount is not None else None
            ),
            "Loan_Amount_Term": (
                float(self.Loan_Amount_Term)
                if self.Loan_Amount_Term is not None
                else None
            ),
            "Credit_History": (
                float(self.Credit_History)
                if self.Credit_History is not None
                else None
            ),
            "Property_Area": self.Property_Area,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert applicant features into a single-row pandas DataFrame."""
        return pd.DataFrame([self.to_feature_dict()])


class BatchPredictionRequest(BaseModel):
    """Pydantic schema for submitting multiple loan applications for batch prediction."""

    applications: List[LoanApplication] = Field(
        ...,
        min_length=1,
        description="List of loan applications to evaluate in batch",
    )


class PredictionResponse(BaseModel):
    """Pydantic schema for loan prediction output."""

    loan_id: Optional[str] = Field(
        None, description="Loan application ID if provided in request"
    )
    loan_status: str = Field(
        ...,
        description="Predicted loan status ('Y' for Approved, 'N' for Rejected)",
    )
    prediction: Union[int, str] = Field(
        ..., description="Raw model prediction output (e.g. 1/0 or 'Y'/'N')"
    )
    probability: Optional[float] = Field(
        None,
        description="Model confidence probability for loan approval (between 0.0 and 1.0) if supported",
    )
    model_name: Optional[str] = Field(
        None, description="Registered model name that generated prediction"
    )
    model_alias: Optional[str] = Field(
        None, description="Active model alias (e.g. 'champion')"
    )
    model_version: Optional[str] = Field(
        None, description="Active model version string"
    )


class HealthResponse(BaseModel):
    """Pydantic schema for health check endpoint."""

    status: str = Field(
        ...,
        description="API health status ('healthy' if model loaded, otherwise 'degraded')",
    )
    model_loaded: bool = Field(
        ..., description="Whether the ML model is currently loaded in memory"
    )
    model_name: Optional[str] = Field(None, description="Target registered model name")
    model_alias: Optional[str] = Field(None, description="Target model alias")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    timestamp: str = Field(..., description="ISO 8601 formatted server timestamp")


class AppInfoResponse(BaseModel):
    """Pydantic schema for basic application information endpoint."""

    name: str = Field(..., description="Application service name")
    version: str = Field(..., description="Service version")
    description: str = Field(..., description="Application description")
    status: str = Field(..., description="Current application status")
    model_name: str = Field(..., description="Configured model name in MLflow")
    model_alias: str = Field(..., description="Configured model alias in MLflow")
    endpoints: Dict[str, str] = Field(
        ..., description="Map of available API endpoints"
    )


class MetricsResponse(BaseModel):
    """Pydantic schema for serving and inference metrics endpoint."""

    total_predictions: int = Field(..., description="Total prediction requests received")
    successful_predictions: int = Field(
        ..., description="Total successful predictions returned"
    )
    failed_predictions: int = Field(..., description="Total failed prediction requests")
    approved_predictions: int = Field(
        ..., description="Count of approved loan predictions ('Y')"
    )
    rejected_predictions: int = Field(
        ..., description="Count of rejected loan predictions ('N')"
    )
    approval_rate: float = Field(
        ..., description="Fraction of successful predictions that were approved"
    )
    rejection_rate: float = Field(
        ..., description="Fraction of successful predictions that were rejected"
    )
    average_latency_ms: float = Field(
        ..., description="Average inference latency in milliseconds"
    )
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    model_name: Optional[str] = Field(None, description="Active model name")
    model_alias: Optional[str] = Field(None, description="Active model alias")
    model_loaded: bool = Field(..., description="Whether the model is currently loaded")


class ReloadModelResponse(BaseModel):
    """Pydantic schema for champion model reload response."""

    status: str = Field(
        ..., description="Status of the reload operation ('success' or 'error')"
    )
    message: str = Field(
        ..., description="Human-readable message describing the reload result"
    )
    model_name: str = Field(..., description="Name of the reloaded model")
    model_alias: str = Field(..., description="Alias of the reloaded model")
    model_version: Optional[str] = Field(
        None, description="Version of the reloaded model"
    )
    reloaded_at: str = Field(
        ..., description="ISO 8601 timestamp when reload occurred"
    )
