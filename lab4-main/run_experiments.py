"""Эксперименты на german.csv. Запуск: python run_experiments.py

Модель выбирается по средней ROC-AUC 5-fold CV на train. Test используется
только для итоговой отчётной таблицы и в выборе модели не участвует.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ensemble_models import CATEGORICAL_IDX, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

warnings.filterwarnings("ignore")
SEED = 42
TEST_SIZE = 0.2


def make_prep(n_features: int) -> ColumnTransformer:
    num_idx = [i for i in range(n_features) if i not in CATEGORICAL_IDX]
    return ColumnTransformer([
        ("num", StandardScaler(), num_idx),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_IDX),
    ])


def lr(c=0.1):
    return LogisticRegression(C=c, max_iter=3000, random_state=SEED)


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                  max_features="sqrt", n_jobs=-1, random_state=SEED)


def gb():
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=2, subsample=0.8, random_state=SEED)


CANDIDATES = {
    "LogReg C=0.1": lambda: lr(0.1),
    "LogReg C=0.3": lambda: lr(0.3),
    "LogReg C=1.0": lambda: lr(1.0),
    "RandomForest": rf,
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
    "GradientBoosting": gb,
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_depth=3,
        l2_regularization=1.0, random_state=SEED),
    "MLP (32,16)": lambda: MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=0.1, early_stopping=True,
        max_iter=500, random_state=SEED),
    "Voting LR+RF": lambda: VotingClassifier(
        [("lr", lr()), ("rf", rf())], voting="soft"),
    "Voting LR+RF+GB (equal weights)": lambda: VotingClassifier(
        [("lr", lr()), ("rf", rf()), ("gb", gb())], voting="soft"),
    "Voting LR+RF+GB (weights 2,1,1)": lambda: VotingClassifier(
        [("lr", lr()), ("rf", rf()), ("gb", gb())], voting="soft",
        weights=[2, 1, 1]),
}


def metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def main() -> None:
    df = pd.read_csv("german.csv", sep=";")
    y = df.pop("Creditability").to_numpy()
    X = df.to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, factory in CANDIDATES.items():
        pipe = Pipeline([("prep", make_prep(X.shape[1])), ("model", factory())])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        m = metrics(y_te, pipe.predict_proba(X_te)[:, 1])
        rows.append({"model": name, "cv_roc_auc_train": float(cv_auc), **{
            f"test_{k}": v for k, v in m.items()}})
        print(f"{name:34s} cv_auc={cv_auc:.4f} test_auc={m['roc_auc']:.4f}")

    best = max(rows, key=lambda r: r["cv_roc_auc_train"])
    print("\nВыбрано по CV на train:", best["model"])

    # Финальный запуск именно контрактной функции.
    result = run_ensemble_experiment(EnsembleInput(
        train_features=X_tr, train_target=y_tr,
        test_features=X_te, test_target=y_te, random_seed=SEED))
    final = {
        "model_name": result.model_name,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
    }
    print("Контрактная функция:", final)

    out = {
        "seed": SEED,
        "split": {"test_size": TEST_SIZE, "stratify": True,
                  "n_train": int(len(y_tr)), "n_test": int(len(y_te))},
        "selection_criterion": "mean 5-fold stratified CV ROC-AUC on train",
        "selected_by_cv": best["model"],
        "experiments": rows,
        "final_model": final,
    }
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()