"""
Вспомогательные функции для урока 2:
  * воспроизводимость и выбор устройства
  * генерация / загрузка данных (random, diabetes, breast_cancer)
  * Dataset-классы и DataLoader'ы
  * метрики
  * логирование и графики обучения
"""
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------- Общие утилиты ----------------

def get_logger(name="lesson2", log_file=None):
    """Логгер, который пишет в консоль и (опционально) в файл."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _standardize(X):
    """Нулевое среднее и единичный std для каждого признака.
    Без этого градиентный спуск на реальных данных сходится плохо."""
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (X - mean) / std


# ---------------- Данные ----------------

def make_regression_data(n=200, n_features=1, noise=0.1, source="random"):
    """
    Возвращает X: (n, d), y: (n, 1).
    source: 'random' — синтетика y = X @ w + b + шум
            'diabetes' — sklearn load_diabetes (442 x 10)
    """
    if source == "random":
        X = torch.rand(n, n_features) * 10 - 5
        w_true = torch.arange(1, n_features + 1, dtype=torch.float32).unsqueeze(1)
        b_true = 0.5
        y = X @ w_true + b_true + noise * torch.randn(n, 1)
        return X, y

    if source == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data.data, dtype=torch.float32)
        y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
        return _standardize(X), y

    raise ValueError(f"Неизвестный source: {source!r} (доступно: 'random', 'diabetes')")


def make_classification_data(n=200, n_features=2, source="random"):
    """
    Возвращает X: (n, d), y: (n, 1) с метками 0/1.
    source: 'random' — два линейно разделимых облака
            'breast_cancer' — sklearn load_breast_cancer (569 x 30)
    """
    if source == "random":
        X = torch.randn(n, n_features)
        w_true = torch.randn(n_features, 1)
        logits = X @ w_true + 0.3 * torch.randn(n, 1)
        y = (logits > 0).float()
        return X, y

    if source == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data.data, dtype=torch.float32)
        y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
        return _standardize(X), y

    raise ValueError(f"Неизвестный source: {source!r} (доступно: 'random', 'breast_cancer')")


# ---------------- Dataset-классы ----------------

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(dataset, batch_size=32, val_ratio=0.2, seed=42):
    """Делит датасет на train/val и возвращает два DataLoader'а."""
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------- Метрики ----------------

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()


def rmse(y_pred, y_true):
    return mse(y_pred, y_true) ** 0.5


def r2_score(y_pred, y_true):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean().item()


def precision_recall_f1(y_pred, y_true):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ---------------- Логирование / графики ----------------

def log_epoch(logger, epoch, total, **metrics):
    parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"Epoch {epoch:>3}/{total} | {parts}")


def plot_history(history, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=name)
    ax.set_xlabel("Эпоха")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)