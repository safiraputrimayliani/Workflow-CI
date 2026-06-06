# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import warnings

import mlflow
import mlflow.sklearn

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

warnings.filterwarnings("ignore")


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("titanic_preprocessing.csv")

# =========================================================
# FEATURE DAN TARGET
# =========================================================

X = data.drop("Survived", axis=1)
y = data["Survived"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# START MLFLOW RUN
# =========================================================

with mlflow.start_run(nested=True):

    # ── Model ──────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ── Metrics ────────────────────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    print("Accuracy :",  accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # ── Confusion Matrix Artifact ──────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # ── Classification Report Artifact ────────────────────
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # ── Log Model ─────────────────────────────────────────
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("\nMLflow run selesai!")
    print("Artifacts: confusion_matrix.png, classification_report.txt")
