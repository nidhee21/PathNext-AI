"""
ml_model.py
-----------
This file:
1. Loads the career dataset
2. Prepares features
3. Trains Decision Tree and Random Forest
4. Evaluates both models
5. Saves the best model (Random Forest) as career_model.pkl

Run this file once before starting the Flask/FastAPI servers:
    python ml_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
DATASET_PATH = "../dataset/career_dataset.csv"
MODEL_PATH   = "../models/career_model.pkl"

# ---------------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------------
def load_dataset():
    """Load CSV into a pandas dataframe."""
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found! Run scripts/generate_dataset.py first.")
        exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ---------------------------------------------------------------
# Step 2: Prepare Features (X) and Labels (y)
# ---------------------------------------------------------------
def prepare_features(df):
    """
    Convert raw dataframe into ML-ready X and y.

    X = all input features (numbers only)
    y = career label (encoded as a number)
    """

    # Encode stream: "Science PCM" → 0, "Science PCB" → 1, etc.
    stream_encoder = LabelEncoder()
    df["stream_encoded"] = stream_encoder.fit_transform(df["stream"])

    # Encode career labels: "Software Engineer" → 0, etc.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["career"])

    # Subject columns
    subject_cols = [
        "physics", "chemistry", "maths", "biology",
        "accounts", "business", "economics",
        "history", "political_science", "geography",
        "psychology", "sociology", "english"
    ]

    # Interest columns
    interest_cols = [col for col in df.columns if col.startswith("interest_")]

    # Soft skill columns
    skill_cols = [
        "communication", "leadership", "creativity",
        "analytical_thinking", "problem_solving", "teamwork",
        "empathy", "critical_thinking", "time_management", "adaptability"
    ]

    # Combine all feature columns
    feature_cols = ["stream_encoded"] + subject_cols + interest_cols + skill_cols

    # Only keep columns that exist in dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of career classes: {len(label_encoder.classes_)}")

    return X, y, stream_encoder, label_encoder, feature_cols

# ---------------------------------------------------------------
# Step 3: Train and Compare Models
# ---------------------------------------------------------------
def train_and_evaluate(X, y):
    """
    Train Decision Tree and Random Forest.
    Compare accuracy and return the best model.
    """

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --- Decision Tree ---
    print("\nTraining Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")

    # --- Random Forest ---
    print("\nTraining Random Forest (100 trees)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

    # Random Forest is our winner (almost always more accurate)
    print(f"\nSelected model: Random Forest ({rf_acc*100:.2f}% accuracy)")

    return rf_model

# ---------------------------------------------------------------
# Step 4: Save Model
# ---------------------------------------------------------------
def save_model(model, stream_encoder, label_encoder, feature_cols):
    """
    Bundle and save everything we need for predictions.
    We save:
    - the trained model
    - the stream encoder (to convert stream name → number)
    - the label encoder (to convert number → career name)
    - the list of feature columns (order matters!)
    """
    os.makedirs("../models", exist_ok=True)

    bundle = {
        "model": model,
        "stream_encoder": stream_encoder,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("You can now start flask_app.py and fastapi_app.py!")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("PathNext AI — Model Training")
    print("=" * 50)

    df = load_dataset()
    X, y, stream_encoder, label_encoder, feature_cols = prepare_features(df)
    model = train_and_evaluate(X, y)
    save_model(model, stream_encoder, label_encoder, feature_cols)

    print("\nModel is ready.")
