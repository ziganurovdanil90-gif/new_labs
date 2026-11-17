"""
Линейная регрессия «руками» (без autograd).
    y_hat = X @ w + b,   MSE = mean((y_hat - y)^2)
    dL/dw = 2/n * X^T @ (y_hat - y),   dL/db = 2/n * sum(y_hat - y)

Запуск:
    python linreg_manual.py
    python linreg_manual.py --source diabetes
"""
import argparse

import torch

from utils import (RegressionDataset, get_logger, log_epoch, make_loaders,
                   make_regression_data, mse, plot_history, r2_score, rmse, set_seed)


def predict(X, w, b):
    return X @ w + b


def evaluate(loader, w, b):
    preds, targets = [], []
    for X_b, y_b in loader:
        preds.append(predict(X_b, w, b))
        targets.append(y_b)
    y_pred, y_true = torch.cat(preds), torch.cat(targets)
    return mse(y_pred, y_true), rmse(y_pred, y_true), r2_score(y_pred, y_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="random", choices=["random", "diabetes"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    logger = get_logger("linreg_manual", log_file="linreg_manual.log")

    X, y = make_regression_data(n=200, n_features=1, source=args.source)
    train_loader, val_loader = make_loaders(RegressionDataset(X, y), batch_size=args.batch_size)
    logger.info(f"Датасет: {args.source}, X={tuple(X.shape)}")

    w = torch.zeros(X.shape[1], 1)
    b = torch.zeros(1)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, args.epochs + 1):
        epoch_loss, n_seen = 0.0, 0
        for X_b, y_b in train_loader:
            n = X_b.shape[0]

            y_hat = predict(X_b, w, b)          # forward
            err = y_hat - y_b
            loss = (err ** 2).mean()

            grad_w = 2.0 / n * X_b.T @ err      # backward вручную
            grad_b = 2.0 / n * err.sum()

            w -= args.lr * grad_w               # шаг градиентного спуска
            b -= args.lr * grad_b

            epoch_loss += loss.item() * n
            n_seen += n

        train_loss = epoch_loss / n_seen
        val_loss, val_rmse, val_r2 = evaluate(val_loader, w, b)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            log_epoch(logger, epoch, args.epochs, train_mse=train_loss,
                      val_mse=val_loss, val_rmse=val_rmse, val_r2=val_r2)

    logger.info(f"Итог: w={w.squeeze().tolist()}, b={b.item():.4f}")
    torch.save({"w": w, "b": b}, "linreg_manual.pth")
    logger.info("Веса сохранены в linreg_manual.pth")

    ckpt = torch.load("linreg_manual.pth")
    _, _, r2_loaded = evaluate(val_loader, ckpt["w"], ckpt["b"])
    logger.info(f"R² на val после загрузки весов: {r2_loaded:.4f}")

    plot_history(history, f"Linreg (manual), {args.source}", "linreg_manual_loss.png")


if __name__ == "__main__":
    main()
    