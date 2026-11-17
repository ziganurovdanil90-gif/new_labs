"""
Логистическая регрессия «руками» (без autograd).
    p = sigmoid(X @ w + b),   BCE = -mean(y*log p + (1-y)*log(1-p))
    dL/dw = 1/n * X^T @ (p - y),   dL/db = 1/n * sum(p - y)

Запуск:
    python logreg_manual.py
    python logreg_manual.py --source breast_cancer
"""
import argparse

import torch

from utils import (ClassificationDataset, accuracy, get_logger, log_epoch,
                   make_classification_data, make_loaders, plot_history,
                   precision_recall_f1, set_seed)

EPS = 1e-7


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def bce(p, y):
    p = p.clamp(EPS, 1 - EPS)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def evaluate(loader, w, b):
    probs, targets = [], []
    for X_b, y_b in loader:
        probs.append(predict_proba(X_b, w, b))
        targets.append(y_b)
    p, y_true = torch.cat(probs), torch.cat(targets)
    y_pred = (p >= 0.5).float()
    return (bce(p, y_true).item(), accuracy(y_pred, y_true),
            *precision_recall_f1(y_pred, y_true))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="random", choices=["random", "breast_cancer"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    logger = get_logger("logreg_manual", log_file="logreg_manual.log")

    X, y = make_classification_data(n=200, n_features=2, source=args.source)
    train_loader, val_loader = make_loaders(ClassificationDataset(X, y), batch_size=args.batch_size)
    logger.info(f"Датасет: {args.source}, X={tuple(X.shape)}, доля класса 1 = {y.mean().item():.2f}")

    w = torch.zeros(X.shape[1], 1)
    b = torch.zeros(1)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        epoch_loss, n_seen = 0.0, 0
        for X_b, y_b in train_loader:
            n = X_b.shape[0]

            p = predict_proba(X_b, w, b)      # forward
            loss = bce(p, y_b)

            diff = p - y_b                    # backward
            grad_w = X_b.T @ diff / n
            grad_b = diff.sum() / n

            w -= args.lr * grad_w             # step
            b -= args.lr * grad_b

            epoch_loss += loss.item() * n
            n_seen += n

        train_loss = epoch_loss / n_seen
        val_loss, val_acc, prec, rec, f1 = evaluate(val_loader, w, b)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch == 1 or epoch % 10 == 0:
            log_epoch(logger, epoch, args.epochs, train_bce=train_loss,
                      val_bce=val_loss, val_acc=val_acc, val_f1=f1)

    _, acc, prec, rec, f1 = evaluate(val_loader, w, b)
    logger.info(f"Val: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")
    torch.save({"w": w, "b": b}, "logreg_manual.pth")
    logger.info("Веса сохранены в logreg_manual.pth")

    plot_history(history, f"Logreg (manual), {args.source}", "logreg_manual_loss.png")


if __name__ == "__main__":
    main()