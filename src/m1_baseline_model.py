"""
Streamline - Module 1 Baseline model
---------------------------------------------------------------
This script loads customer data, preprocesses features, trains a baseline
classification model to predict customer churn, evaluates performance,
performs cross-validation and hyperparameter tuning, and saves the final
model and preprocessing pipeline.

"""


# -----------------------------
# Imports
# -----------------------------

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/Stream_pulse_customer_data.csv"
MODEL_PATH = "models/baseline_model.pkl"
PIPELINE_PATH = "src/preprocessing_pipeline.pkl"
TARGET_COLUMN = "churned"

# Ensure output directories exist
os.makedirs("models", exist_ok=True)

# -------------------------------------------
# Load Data
# -------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()  # <<< FIX

# Separate features and targets

X = df.drop(columns=[TARGET_COLUMN]) # faetures are everything else [inputs]
y = df[TARGET_COLUMN]                # target column is the name of the variable that you want to predict [churn], this is the label [output].

# --------------------------------------------
# Identify Feature Types, data preprocessing
# --------------------------------------------

categorical_features = X.select_dtypes(include=["object", "category","string"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -----------------------------------------------
# Preprocessing Pipeline
# -----------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -----------------------------
# Train-Test Splitting 
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30, #30% of data → test set
    stratify=y, # it preserves the churn ratio, without stratification, randomness can break your split.
    random_state=42
)


# -----------------------------
# Baseline Model Pipeline
# -----------------------------
"""
This pipeline says: “Whenever I train or predict: Preprocess the data, Then run Logistic Regression”
"""
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# -----------------------------
# Train Baseline Model
# -----------------------------
model_pipeline.fit(X_train, y_train)


# -----------------------------
# Evaluation on Test Set
# -----------------------------
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print("Test Set Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")



# -----------------------------
# 5-Fold Cross-Validation
# -----------------------------
cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation Accuracy:")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Std CV Accuracy : {cv_scores.std():.4f}")

# -----------------------------
# Hyperparameter Tuning (GridSearchCV)
# -----------------------------
param_grid = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__solver": ["lbfgs"]
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# -----------------------------
# Final Evaluation (Tuned Model)
# -----------------------------
y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, zero_division=0)
recall_best = recall_score(y_test, y_pred_best, zero_division=0)

print("\nTuned Model Test Set Evaluation:")
print(f"Accuracy : {accuracy_best:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Recall   : {recall_best:.4f}")

# -----------------------------
# Save Model & Preprocessing Pipeline
# -----------------------------
joblib.dump(best_model, MODEL_PATH)
joblib.dump(preprocessor, PIPELINE_PATH)

print("\nFinal result: The baseline Logistic Regression model achieved ~81% accuracy; however, " \
"precision and recall for churn were 0. This indicates the model is biased toward the majority (non-churn) class due to class imbalance.")

