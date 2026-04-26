"""
ML Pipeline - Classification on Iris Dataset
Trains multiple classifiers, selects best, and saves the model.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & Explore Data ────────────────────────────────────────────────────
print("=" * 60)
print("  ML PIPELINE - IRIS CLASSIFICATION")
print("=" * 60)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
class_names = list(iris.target_names)

print(f"\n[DATA] Shape       : {X.shape}")
print(f"[DATA] Features    : {list(X.columns)}")
print(f"[DATA] Classes     : {class_names}")
print(f"[DATA] Distribution:\n{y.value_counts().to_string()}")
print(f"\n[DATA] Missing values: {X.isnull().sum().sum()}")
print(f"[DATA] Descriptive stats:\n{X.describe().round(2).to_string()}")

# ── 2. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Define Classifiers ─────────────────────────────────────────────────────
classifiers = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
}

# ── 4. Evaluate All Classifiers ───────────────────────────────────────────────
print("\n[TRAINING] Evaluating classifiers with 5-fold CV...")
print("-" * 60)

results = {}
for name, pipeline in classifiers.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    pipeline.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    test_f1  = f1_score(y_test, pipeline.predict(X_test), average="weighted")
    results[name] = {
        "cv_mean":  round(float(cv_scores.mean()), 4),
        "cv_std":   round(float(cv_scores.std()),  4),
        "test_acc": round(float(test_acc), 4),
        "test_f1":  round(float(test_f1),  4),
    }
    print(f"  {name:<25} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  Test={test_acc:.4f}")

# ── 5. Select Best Model ──────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_acc"])
print(f"\n[BEST] {best_name} selected (test acc={results[best_name]['test_acc']})")

# ── 6. Hyperparameter Tuning on Best Model ────────────────────────────────────
print("\n[TUNING] Running GridSearchCV on best model...")

if "Random Forest" in best_name:
    param_grid = {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 5, 10]}
elif "SVM" in best_name:
    param_grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]}
elif "Logistic" in best_name:
    param_grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__solver": ["lbfgs", "saga"]}
elif "Gradient" in best_name:
    param_grid = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.05, 0.1, 0.2]}
else:
    param_grid = {"clf__n_neighbors": [3, 5, 7, 9]}

grid_search = GridSearchCV(classifiers[best_name], param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"[TUNING] Best params: {best_params}")
print(f"[TUNING] Best CV score: {grid_search.best_score_:.4f}")

# ── 7. Final Evaluation ───────────────────────────────────────────────────────
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

final_acc = accuracy_score(y_test, y_pred)
final_f1  = f1_score(y_test, y_pred, average="weighted")
cm        = confusion_matrix(y_test, y_pred).tolist()
report    = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

print(f"\n[FINAL] Accuracy : {final_acc:.4f}")
print(f"[FINAL] F1-Score : {final_f1:.4f}")
print(f"\n[FINAL] Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 8. Save Model & Metadata ──────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

metadata = {
    "model_name":     best_name,
    "best_params":    best_params,
    "feature_names":  list(X.columns),
    "class_names":    class_names,
    "feature_ranges": {
        col: {"min": round(float(X[col].min()), 2), "max": round(float(X[col].max()), 2),
              "mean": round(float(X[col].mean()), 2)}
        for col in X.columns
    },
    "metrics": {
        "accuracy": round(float(final_acc), 4),
        "f1_score": round(float(final_f1),  4),
        "confusion_matrix": cm,
        "classification_report": report,
    },
    "all_results": results,
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
}

with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n[SAVED] model/model.pkl")
print(f"[SAVED] model/metadata.json")
print("=" * 60)
print("  Training complete!")
print("=" * 60)
