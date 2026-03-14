"""
fastapi_app.py
basic model 
--------------
This is the FastAPI server that handles ML predictions.

It exposes one endpoint:
    POST /predict  →  returns top 3 career predictions

Flask calls this endpoint internally to get predictions.

To run this server (from the backend/ folder):
    uvicorn fastapi_app:app --port 8001 --reload

Then test it at: http://localhost:8001/docs
"""

import pickle
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

# ---------------------------------------------------------------
# Load the trained ML model
# ---------------------------------------------------------------
MODEL_PATH = "../models/career_model.pkl"

# We load the model once when server starts (not on every request)
model_bundle = None

def get_model():
    """Load model from disk if not already loaded."""
    global model_bundle
    if model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file not found! Run backend/ml_model.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)
        print("Model loaded successfully!")
    return model_bundle

# ---------------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------------
app = FastAPI(
    title="PathNext AI — Prediction API",
    description="ML career prediction endpoint",
    version="1.0.0"
)

# Allow Flask to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Request body model
# ---------------------------------------------------------------
class StudentInput(BaseModel):
    """
    The shape of data we receive from Flask.

    Example:
    {
        "stream": "Science PCM",
        "marks": {"maths": 90, "physics": 85, ...},
        "interests": ["interest_tech", "interest_ai"],
        "soft_skills": {"analytical_thinking": 5, "communication": 3, ...}
    }
    """
    stream: str
    marks: Dict[str, float] = {}
    interests: List[str] = []
    soft_skills: Dict[str, int] = {}

# ---------------------------------------------------------------
# Helper: convert student input to feature vector
# ---------------------------------------------------------------
def build_feature_vector(data: StudentInput, bundle: dict) -> np.ndarray:
    """
    Convert student form data into a numpy array the model can use.

    The model expects a fixed-length array of numbers.
    This function builds that array in the correct order.
    """

    # Encode stream name to a number
    try:
        stream_encoded = bundle["stream_encoder"].transform([data.stream])[0]
    except:
        stream_encoded = 0

    # Subject marks (in the same order as training)
    subject_cols = [
        "physics", "chemistry", "maths", "biology",
        "accounts", "business", "economics",
        "history", "political_science", "geography",
        "psychology", "sociology", "english"
    ]
    subject_values = [float(data.marks.get(col, 0)) for col in subject_cols]

    # Interest binary values
    all_interest_cols = [c for c in bundle["feature_cols"] if c.startswith("interest_")]
    interest_values = [1.0 if col in data.interests else 0.0 for col in all_interest_cols]

    # Soft skill values
    skill_cols = [
        "communication", "leadership", "creativity",
        "analytical_thinking", "problem_solving", "teamwork",
        "empathy", "critical_thinking", "time_management", "adaptability"
    ]
    skill_values = [float(data.soft_skills.get(col, 3)) for col in skill_cols]

    # Combine into one array
    feature_vector = np.array(
        [stream_encoded] + subject_values + interest_values + skill_values
    ).reshape(1, -1)

    return feature_vector

# ---------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "PathNext AI Prediction API is running!"}


@app.post("/predict")
def predict(student: StudentInput):
    """
    Predict top 3 careers for a student.

    Returns:
    {
        "top_careers": [
            {"career": "Software Engineer", "confidence": 0.82},
            ...
        ]
    }
    """
    try:
        bundle = get_model()
        model = bundle["model"]
        label_encoder = bundle["label_encoder"]

        # Convert input to feature vector
        X = build_feature_vector(student, bundle)

        # Get probability for each career
        probs = model.predict_proba(X)[0]

        # Get top 3 indices (highest probability first)
        top3_idx = np.argsort(probs)[::-1][:3]

        results = []
        for idx in top3_idx:
            career_name = label_encoder.inverse_transform([idx])[0]
            confidence = round(float(probs[idx]), 4)
            results.append({"career": career_name, "confidence": confidence})

        return {"top_careers": results}

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
