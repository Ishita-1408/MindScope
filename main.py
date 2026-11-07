from pathlib import Path
from typing import Any, List, Literal, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.explain import MindScopeExplainer, load_explainer

# Anchor model path relative to this source file location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

# Model loading with explicit status tracking and safety
# Note: scikit-learn InconsistentVersionWarning is preserved intentionally
# to maintain scientific transparency regarding the legacy 1.9.0 serialization artifact.
model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model_load_error = str(e)

# Modern explainer pipeline initialization
explainer = None
explainer_load_error = None
try:
    explainer = load_explainer()
except Exception as e:
    explainer_load_error = str(e)

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI(
    title="MindScope API",
    description="Student Mental Health Analytics & Predictive Intelligence Platform API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas with strict observational and engineering bounds
class StudentData(BaseModel):
    age: int = Field(
        ...,
        ge=18,
        le=24,
        description="Student age strictly within the verified observational cohort (18–24)"
    )
    gender: Literal['Male', 'Female']
    country: str = Field(..., min_length=1, description="Student country of residence")
    academic_level: Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform: Literal[
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter',
        'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ]
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours: float = Field(
        ...,
        ge=0.0,
        le=16.0,
        description="Daily social media usage hours (engineering bound: 0–16)"
    )
    daily_unlocks: int = Field(
        ...,
        ge=0,
        le=500,
        description="Daily device unlocks (engineering bound: 0–500)"
    )
    study_hours: float = Field(
        ...,
        ge=0.0,
        le=16.0,
        description="Daily study hours (engineering bound: 0–16)"
    )
    physical_activity_hours: float = Field(
        ...,
        ge=0.0,
        le=8.0,
        description="Daily physical activity hours (engineering bound: 0–8; negative values rejected)"
    )
    sleep_hours_per_night: float = Field(
        ...,
        ge=2.0,
        le=16.0,
        description="Daily sleep hours (engineering bound: 2–16)"
    )
    stress_level: Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    details: str
    explainer_loaded: bool = False
    explainer_details: Optional[str] = None


class RootResponse(BaseModel):
    platform: str
    version: str
    status: str
    documentation: str
    health: str


class FeatureAttribution(BaseModel):
    feature: str
    display_name: str
    value: Any
    unit: str
    formatted_value: str
    shap_value: float
    direction: Literal["positive", "negative"]
    abs_impact: float
    rank: int
    step_start: float
    cumulative_score: float


class WaterfallStep(BaseModel):
    step_type: Literal["baseline", "attribution", "prediction"]
    label: str
    feature: Optional[str] = None
    delta: float
    cumulative_score: float


class LocalExplanationResponse(BaseModel):
    predicted_mental_health_score: float
    expected_baseline_score: float
    net_attribution: float
    feature_attributions: List[FeatureAttribution]
    waterfall_steps: List[WaterfallStep]
    additivity_verified: bool
    additivity_error: float
    disclaimer: str


@app.get('/', response_model=RootResponse)
def root():
    return RootResponse(
        platform="MindScope — Student Mental Health Analytics & Predictive Intelligence Platform",
        version="1.0.0",
        status="operational",
        documentation="/docs",
        health="/health"
    )


@app.get('/health', response_model=HealthResponse)
def health():
    is_explainer_ready = explainer is not None
    expl_details = "MindScopeExplainer active (tree_path_dependent TreeSHAP)" if is_explainer_ready else f"Explainer failed to load: {explainer_load_error}"

    if model is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_path=str(MODEL_PATH),
            details=f"Model failed to load: {model_load_error}",
            explainer_loaded=is_explainer_ready,
            explainer_details=expl_details,
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=str(MODEL_PATH),
        details="Pipeline active (Note: legacy sklearn 1.9.0 serialization artifact loaded under 1.8.0)",
        explainer_loaded=is_explainer_ready,
        explainer_details=expl_details,
    )


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive model is currently unavailable."
        )

    country_group = data.country if data.country in top_countries else "Other"

    input_row = pd.DataFrame([{
        'Age': data.age,
        'Gender': data.gender,
        'Country': data.country,
        'Academic_Level': data.academic_level,
        'Most_Used_Platform': data.most_used_platform,
        'Purpose_Of_Use': data.purpose_of_use,
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
        'Daily_Unlocks': data.daily_unlocks,
        'Study_Hours': data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
        'Stress_Level': data.stress_level,
        'Grouped_country': country_group
    }])

    try:
        prediction = model.predict(input_row)[0]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failure during model prediction."
        )

    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))


@app.post('/explain', response_model=LocalExplanationResponse)
def explain(data: StudentData):
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability engine is currently unavailable."
        )

    input_df = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Country": data.country,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
    }])

    try:
        explanation = explainer.explain_prediction(input_df)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failure during explanation generation."
        )

    return LocalExplanationResponse(
        predicted_mental_health_score=round(explanation["prediction"], 4),
        expected_baseline_score=round(explanation["base_value"], 4),
        net_attribution=round(explanation["net_attribution"], 4),
        feature_attributions=explanation["feature_attributions"],
        waterfall_steps=explanation["waterfall_steps"],
        additivity_verified=explanation["additivity_verified"],
        additivity_error=explanation["additivity_error"],
        disclaimer=explanation["disclaimer"],
    )
