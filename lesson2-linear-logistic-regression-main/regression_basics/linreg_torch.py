"""
Линейная регрессия на nn.Linear + nn.MSELoss + torch.optim.SGD.
Показано сохранение/загрузка state_dict.

Запуск:
    python linreg_torch.py
    python linreg_torch.py --source diabetes
"""
import argparse

import torch
from torch import nn

from utils import (RegressionDataset, get_device, get_logger, log_epoch, make_loaders,
                   make_regression_data, plot_history, r2_score, rmse, set_seed)


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, targets = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        preds.append(model(X_b))
        targets.append(y_b)
    y_pred, y_true = torch.cat(preds), torch.cat(targets)
    return criterion(y_pred, y_true).item(), rmse(y_pred, y_true), r2_score(y_pred, y_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="random", choices=["random", "diabetes"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    logger = get_logger("linreg_torch", log_file="linreg_torch.log")

    X, y = make_regression_data(n=200, n_features=1, source=args.source)
    train_loader, val_loader = make_loaders(RegressionDataset(X, y), batch_size=args.batch_size)
    logger.info(f"Датасет: {args.source}, X={tuple(X.shape)}, устройство: {device}")

    model = LinearRegression(X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_path = float("inf"), "linreg_torch_best.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()          # 1) обнулить градиенты
            y_hat = model(X_b)             # 2) forward
            loss = criterion(y_hat, y_b)   # 3) лосс
            loss.backward()                # 4) backward (autograd)
            optimizer.step()               # 5) обновить веса

            epoch_loss += loss.item() * X_b.size(0)
            n_seen += X_b.size(0)

        train_loss = epoch_loss / n_seen
        val_loss, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 10 == 0:
            log_epoch(logger, epoch, args.epochs, train_mse=train_loss,
                      val_mse=val_loss, val_rmse=val_rmse, val_r2=val_r2)

    logger.info("Содержимое state_dict:")
    for name, tensor in model.state_dict().items():
        logger.info(f"  {name}: shape={tuple(tensor.shape)} -> {tensor.flatten()[:5].tolist()}")

    torch.save(model.state_dict(), "linreg_torch.pth")
    logger.info("Финальные веса сохранены в linreg_torch.pth")

    restored = LinearRegression(X.shape[1]).to(device)
    restored.load_state_dict(torch.load(best_path, map_location=device))
    val_loss, val_rmse, val_r2 = evaluate(restored, val_loader, criterion, device)
    logger.info(f"Лучшая модель ({best_path}): val_mse={val_loss:.4f}, "
                f"val_rmse={val_rmse:.4f}, val_r2={val_r2:.4f}")

    plot_history(history, f"Linreg (torch), {args.source}", "linreg_torch_loss.png")


if __name__ == "__main__":
    main()