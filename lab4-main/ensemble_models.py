"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

MODEL_NAME = "Soft Voting: LogisticRegression + RandomForest + GradientBoosting"

# Позиции признаков german.csv (без столбца Creditability, всего 20).
# Категориальные признаки закодированы числами, поэтому их нужно
# кодировать one-hot: порядок кодов не несёт смысла.
N_FEATURES = 20
CATEGORICAL_IDX = [
    0,   # Account_Balance
    2,   # Payment_Status_of_Previous_Credit
    3,   # Purpose
    5,   # Value_Savings_Stocks
    6,   # Length_of_current_employment
    8,   # Sex_Marital_Status
    9,   # Guarantors
    11,  # Most_valuable_available_asset
    13,  # Concurrent_Credits
    14,  # Type_of_apartment
    16,  # Occupation
    18,  # Telephone
    19,  # Foreign_Worker
]


def _build_pipeline(n_features: int, seed: int) -> Pipeline:
    """Препроцессинг и ансамбль в одном Pipeline: fit только на train."""
    if n_features == N_FEATURES:
        cat_idx = CATEGORICAL_IDX
    else:  # непредвиденный формат: не угадываем категории, масштабируем всё
        cat_idx = []
    num_idx = [i for i in range(n_features) if i not in cat_idx]

    transformers = [("num", StandardScaler(), num_idx)]
    if cat_idx:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx)
        )
    prep = ColumnTransformer(transformers)

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=0.1, max_iter=3000, random_state=seed)),
            ("rf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                n_jobs=-1, random_state=seed)),
            ("gb", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=2,
                subsample=0.8, random_state=seed)),
        ],
        voting="soft",
    )
    return Pipeline([("prep", prep), ("model", ensemble)])


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = np.asarray(pd.DataFrame(data.train_features), dtype=float)
    X_test = np.asarray(pd.DataFrame(data.test_features), dtype=float)
    y_train_raw = np.asarray(data.train_target)
    y_test_raw = np.asarray(data.test_target)  # только для подсчёта метрик

    # Положительный класс - наибольшая метка train (1 для кодировки 0/1).
    positive = np.unique(y_train_raw).max()
    y_train = (y_train_raw == positive).astype(int)
    y_test = (y_test_raw == positive).astype(int)

    model = _build_pipeline(X_train.shape[1], data.random_seed)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return EnsembleResult(
        model_name=MODEL_NAME,
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )