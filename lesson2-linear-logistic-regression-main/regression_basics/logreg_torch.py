"""
Логистическая регрессия на nn.Linear + nn.BCEWithLogitsLoss + torch.optim.SGD.

Запуск:
    python logreg_torch.py
    python logreg_torch.py --source breast_cancer
"""
import argparse

import torch
from torch import nn

from utils import (ClassificationDataset, accuracy, get_device, get_logger, log_epoch,
                   make_classification_data, make_loaders, plot_history,
                   precision_recall_f1, set_seed)


class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)  # логиты, сигмоида — только при предсказании


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    logits, targets = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits.append(model(X_b))
        targets.append(y_b)
    z, y_true = torch.cat(logits), torch.cat(targets)
    y_pred = (torch.sigmoid(z) >= 0.5).float()
    return (criterion(z, y_true).item(), accuracy(y_pred, y_true),
            *precision_recall_f1(y_pred, y_true))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="random", choices=["random", "breast_cancer"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    logger = get_logger("logreg_torch", log_file="logreg_torch.log")

    X, y = make_classification_data(n=200, n_features=2, source=args.source)
    train_loader, val_loader = make_loaders(ClassificationDataset(X, y), batch_size=args.batch_size)
    logger.info(f"Датасет: {args.source}, X={tuple(X.shape)}, устройство: {device}")

    model = LogisticRegression(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_path = -1.0, "logreg_torch_best.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_b.size(0)
            n_seen += X_b.size(0)

        train_loss = epoch_loss / n_seen
        val_loss, val_acc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 10 == 0:
            log_epoch(logger, epoch, args.epochs, train_bce=train_loss,
                      val_bce=val_loss, val_acc=val_acc, val_f1=f1)

    torch.save(model.state_dict(), "logreg_torch.pth")
    logger.info("Финальные веса сохранены в logreg_torch.pth")

    restored = LogisticRegression(X.shape[1]).to(device)
    restored.load_state_dict(torch.load(best_path, map_location=device))
    _, acc, prec, rec, f1 = evaluate(restored, val_loader, criterion, device)
    logger.info(f"Лучшая модель ({best_path}): accuracy={acc:.4f}, precision={prec:.4f}, "
                f"recall={rec:.4f}, f1={f1:.4f}")

    plot_history(history, f"Logreg (torch), {args.source}", "logreg_torch_loss.png")


if __name__ == "__main__":
    main()