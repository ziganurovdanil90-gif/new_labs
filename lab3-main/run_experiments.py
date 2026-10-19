"""Запуск: python run_experiments.py  ->  experiment_results.json"""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (compare_classical_models, metrics,
                           train_linear_regression, train_logistic_classifier)

SEED = 42

# ---------- регрессия ----------
df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = (df["Extracurricular Activities"] == "Yes").astype(int)
y = df["Performance Index"].to_numpy(float)
X = df.drop(columns="Performance Index").to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

reg = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte))
test_mse = float(np.mean((reg.predictions - yte) ** 2))
print("MSE epoch1 / 100 / 400 / last:", reg.loss_history[0], reg.loss_history[99],
      reg.loss_history[399], reg.loss_history[-1])
print("weights (orig scale):", reg.weights, "bias:", reg.bias)
print("test MSE:", test_mse)
sk = LinearRegression().fit(Xtr, ytr)  # только для сверки, вне собственной модели
print("sklearn test MSE (check):", float(np.mean((sk.predict(Xte) - yte) ** 2)))

print("--- lr experiment (epochs=800) ---")
for lr in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5):
    r = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte, learning_rate=lr))
    print(f"lr={lr}: train MSE last={r.loss_history[-1]:.4f}, "
          f"test MSE={np.mean((r.predictions - yte) ** 2):.4f}")
print("--- epochs experiment (lr=0.05) ---")
for ep in (10, 50, 100, 300, 800, 2000):
    r = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte, epochs=ep))
    print(f"epochs={ep}: test MSE={np.mean((r.predictions - yte) ** 2):.4f}")

# ---------- классификация ----------
g = pd.read_csv("german.csv", sep=";")
target = g.columns[0]  # Creditability
y = g[target].to_numpy(int)
X = g.drop(columns=target).to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

cls_in = ClassificationInput(Xtr, ytr, Xte, yte)
own = train_logistic_classifier(cls_in)
base = compare_classical_models(cls_in)
print("\nloss epoch1 / last:", own.loss_history[0], own.loss_history[-1])

print("--- logistic lr experiment (epochs=1000) ---")
for lr in (0.001, 0.01, 0.1, 0.5, 1.0):
    o = train_logistic_classifier(ClassificationInput(Xtr, ytr, Xte, yte, learning_rate=lr))
    m = metrics(yte, o.probabilities)
    print(f"lr={lr}: final loss={o.loss_history[-1]:.4f}, auc={m['roc_auc']:.4f}, "
          f"acc={m['accuracy']:.4f}, f1={m['f1']:.4f}")

results = {
    "linear_regression": {"test_mse": test_mse},
    "classification": {
        "own_logistic": metrics(yte, own.probabilities),
        "logistic_regression": asdict(base.logistic_regression),
        "decision_tree": asdict(base.decision_tree),
        "knn": asdict(base.knn),
    },
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))