"""Задание 1: создание и манипуляции с тензорами."""
import torch


def check_shape(t: torch.Tensor, expected: tuple) -> None:
    """Проверяет, что форма тензора совпадает с ожидаемой."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Ожидался torch.Tensor, получено {type(t)}")
    if tuple(t.shape) != tuple(expected):
        raise ValueError(f"Форма {tuple(t.shape)} не равна ожидаемой {tuple(expected)}")


# ---------- 1.1 Создание тензоров ----------
def create_tensors():
    """Возвращает четыре тензора из пункта 1.1."""
    rand_3x4 = torch.rand(3, 4)                 # случайные числа из [0, 1)
    zeros_2x3x4 = torch.zeros(2, 3, 4)          # нули
    ones_5x5 = torch.ones(5, 5)                 # единицы
    seq_4x4 = torch.arange(16).reshape(4, 4)    # числа 0..15 в форме 4x4

    check_shape(rand_3x4, (3, 4))
    check_shape(zeros_2x3x4, (2, 3, 4))
    check_shape(ones_5x5, (5, 5))
    check_shape(seq_4x4, (4, 4))
    return rand_3x4, zeros_2x3x4, ones_5x5, seq_4x4


# ---------- 1.2 Операции с тензорами ----------
def tensor_operations(A: torch.Tensor, B: torch.Tensor):
    """Операции над A (3x4) и B (4x3). Возвращает (A^T, A@B, A*B^T, sum(A))."""
    check_shape(A, (3, 4))
    check_shape(B, (4, 3))

    a_t = A.T                    # транспонирование A -> 4x3
    matmul = A @ B               # матричное умножение -> 3x3
    elementwise = A * B.T        # B.T имеет форму 3x4, как и A
    total = A.sum()              # сумма всех элементов A
    return a_t, matmul, elementwise, total


# ---------- 1.3 Индексация и срезы ----------
def indexing_examples():
    """Срезы тензора 5x5x5 (интерпретация указана в комментариях)."""
    t = torch.arange(125).reshape(5, 5, 5)

    first_row = t[0, 0, :]            # первая строка первой матрицы, форма (5,)
    last_column = t[0, :, -1]         # последний столбец первой матрицы, форма (5,)
    center_2x2 = t[2, 1:3, 1:3]       # блок 2x2 из центральной матрицы, форма (2, 2)
    even_indices = t[::2, ::2, ::2]   # элементы с чётными индексами по всем осям, (3, 3, 3)

    check_shape(first_row, (5,))
    check_shape(last_column, (5,))
    check_shape(center_2x2, (2, 2))
    check_shape(even_indices, (3, 3, 3))
    return first_row, last_column, center_2x2, even_indices


# ---------- 1.4 Работа с формами ----------
def reshape_examples():
    """Преобразует тензор из 24 элементов в пять разных форм."""
    t = torch.arange(24)
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    result = {}
    for shape in shapes:
        reshaped = t.reshape(shape)
        check_shape(reshaped, shape)
        result[shape] = reshaped
    return result


def run_tests() -> None:
    """Простые проверки на известных значениях."""
    A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    B = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    a_t, matmul, elementwise, total = tensor_operations(A, B)
    assert torch.equal(a_t, A.T)
    assert torch.allclose(matmul, torch.mm(A, B))
    assert total.item() == 66.0                     # 0 + 1 + ... + 11
    assert elementwise.shape == (3, 4)

    first_row, last_column, center, even = indexing_examples()
    assert first_row.tolist() == [0, 1, 2, 3, 4]
    assert last_column.tolist() == [4, 9, 14, 19, 24]
    assert even.numel() == 27
    print("Все тесты пройдены")


if __name__ == "__main__":
    rand, zeros, ones, seq = create_tensors()
    print("1.1\n", rand, "\n", zeros.shape, "\n", ones, "\n", seq)

    A, B = torch.rand(3, 4), torch.rand(4, 3)
    a_t, matmul, elementwise, total = tensor_operations(A, B)
    print("\n1.2 A^T:\n", a_t, "\nA@B:\n", matmul,
          "\nA * B^T:\n", elementwise, "\nsum(A):", total.item())

    for name, value in zip(("первая строка", "последний столбец",
                            "центр 2x2", "чётные индексы"), indexing_examples()):
        print(f"\n1.3 {name}: форма {tuple(value.shape)}\n{value}")

    for shape, tensor in reshape_examples().items():
        print(f"\n1.4 форма {shape}: {tuple(tensor.shape)}")

    run_tests()