import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _sigmoid(z):
    # clip защищает от переполнения exp при больших |z|
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _scale(X_train, X_test):
    """Стандартизация: mean/std считаются только по train (без утечки данных)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def metrics(y_true, proba, thr=0.5):
    """Метрики бинарной классификации; положительный класс = 1, порог thr."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float)
    Xt = np.asarray(data.test_features, dtype=float)

    Xs, Xts, mu, sd = _scale(X, Xt)
    n, d = Xs.shape
    w, b, history = np.zeros(d), 0.0, []
    for _ in range(data.epochs):
        err = Xs @ w + b - y
        history.append(float(np.mean(err ** 2)))
        w = w - data.learning_rate * (2.0 / n) * (Xs.T @ err)
        b = b - data.learning_rate * (2.0 / n) * err.sum()

    predictions = Xts @ w + b
    # веса в исходном масштабе: Xt @ w_orig + b_orig == predictions
    w_orig = w / sd
    b_orig = float(b - np.sum(w * mu / sd))
    return RegressionResult(weights=w_orig, bias=b_orig,
                            loss_history=history, predictions=predictions)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float)
    Xt = np.asarray(data.test_features, dtype=float)

    Xs, Xts, mu, sd = _scale(X, Xt)
    n, d = Xs.shape
    w, b, history = np.zeros(d), 0.0, []
    eps = 1e-12
    for _ in range(data.epochs):
        p = _sigmoid(Xs @ w + b)
        history.append(float(-np.mean(y * np.log(p + eps)
                                      + (1 - y) * np.log(1 - p + eps))))
        w = w - data.learning_rate * (Xs.T @ (p - y)) / n
        b = b - data.learning_rate * (p - y).mean()

    probabilities = _sigmoid(Xts @ w + b)
    predictions = (probabilities >= 0.5).astype(int)
    w_orig = w / sd
    b_orig = float(b - np.sum(w * mu / sd))
    return ClassificationResult(weights=w_orig, bias=b_orig,
                                loss_history=history,
                                probabilities=probabilities,
                                predictions=predictions)


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target).astype(int)
    Xt = np.asarray(data.test_features, dtype=float)
    yt = np.asarray(data.test_target).astype(int)

    Xs, Xts, _, _ = _scale(X, Xt)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    out = {}
    for name, m in models.items():
        m.fit(Xs, y)
        out[name] = ClassificationMetrics(**metrics(yt, m.predict_proba(Xts)[:, 1]))
    return BaselineComparisonResult(**out)