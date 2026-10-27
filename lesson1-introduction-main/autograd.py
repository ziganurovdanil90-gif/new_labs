"""Задание 2: автоматическое дифференцирование."""
import torch


# ---------- 2.1 Простые вычисления с градиентами ----------
def task_2_1() -> None:
    """f(x, y, z) = x^2 + y^2 + z^2 + 2xyz, сверка с аналитикой."""
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    # Аналитические производные:
    # df/dx = 2x + 2yz, df/dy = 2y + 2xz, df/dz = 2z + 2xy
    with torch.no_grad():
        expected = (2 * x + 2 * y * z, 2 * y + 2 * x * z, 2 * z + 2 * x * y)

    print("2.1 f =", f.item())
    for name, var, exp in zip("xyz", (x, y, z), expected):
        print(f"  df/d{name}: autograd = {var.grad.item():.4f}, аналитика = {exp.item():.4f}")
        assert torch.isclose(var.grad, exp), f"Градиент по {name} не совпал"


# ---------- 2.2 Градиент функции потерь ----------
def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MSE = (1/n) * sum((y_pred - y_true)^2)."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Формы не совпадают: {tuple(y_pred.shape)} и {tuple(y_true.shape)}")
    return ((y_pred - y_true) ** 2).mean()


def task_2_2() -> None:
    """Линейная модель y_pred = w * x + b, градиенты по w и b."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])      # истинная зависимость y = 2x + 1

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    y_pred = w * x + b
    loss = mse_loss(y_pred, y_true)
    loss.backward()

    # Аналитика: dL/dw = (2/n) * sum((y_pred - y_true) * x),  dL/db = (2/n) * sum(y_pred - y_true)
    with torch.no_grad():
        n = x.numel()
        dw = 2 / n * ((y_pred - y_true) * x).sum()
        db = 2 / n * (y_pred - y_true).sum()

    print("\n2.2 MSE =", loss.item())
    print(f"  dL/dw: autograd = {w.grad.item():.4f}, аналитика = {dw.item():.4f}")
    print(f"  dL/db: autograd = {b.grad.item():.4f}, аналитика = {db.item():.4f}")
    assert torch.isclose(w.grad, dw) and torch.isclose(b.grad, db)


# ---------- 2.3 Цепное правило ----------
def task_2_3() -> None:
    """f(x) = sin(x^2 + 1); df/dx = cos(x^2 + 1) * 2x."""
    x = torch.tensor(1.5, requires_grad=True)
    f = torch.sin(x**2 + 1)

    # Способ 1: torch.autograd.grad (retain_graph, чтобы затем вызвать ещё и backward)
    (grad_autograd,) = torch.autograd.grad(f, x, retain_graph=True)

    # Способ 2: backward
    f.backward()

    with torch.no_grad():
        grad_analytic = torch.cos(x**2 + 1) * 2 * x

    print("\n2.3 f =", f.item())
    print(f"  autograd.grad = {grad_autograd.item():.4f}, backward = {x.grad.item():.4f}, "
          f"аналитика = {grad_analytic.item():.4f}")
    assert torch.isclose(grad_autograd, grad_analytic)
    assert torch.isclose(x.grad, grad_analytic)


if __name__ == "__main__":
    task_2_1()
    task_2_2()
    task_2_3()
    print("\nВсе проверки пройдены")