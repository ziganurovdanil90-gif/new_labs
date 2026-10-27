"""Задание 3: сравнение производительности CPU и CUDA."""
import time

import torch

SIZES = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
WARMUP = 1     # прогревочные запуски (не учитываются)
REPEATS = 3    # число измерений для усреднения


# ---------- 3.1 Подготовка данных ----------
def make_matrices(shape: tuple, device: str = "cpu"):
    """Создаёт две случайные матрицы заданной формы (batch, n, m)."""
    if len(shape) != 3:
        raise ValueError("Ожидалась форма из трёх измерений")
    return torch.rand(shape, device=device), torch.rand(shape, device=device)


# ---------- 3.2 Измерение времени ----------
def time_cpu(fn, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Среднее время выполнения fn на CPU в миллисекундах (time.time)."""
    for _ in range(warmup):
        fn()
    start = time.time()
    for _ in range(repeats):
        fn()
    return (time.time() - start) / repeats * 1000


def time_gpu(fn, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Среднее время выполнения fn на GPU в миллисекундах (torch.cuda.Event)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()               # ждём завершения всех ядер
    return start.elapsed_time(end) / repeats


# ---------- 3.3 Сравнение операций ----------
def get_operations(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Операции для сравнения. Транспонирование делаем .contiguous(), иначе это
    просто view без копирования данных и измерять там нечего."""
    return {
        "Матричное умножение": lambda: torch.matmul(a, b),
        "Сложение": lambda: a + b,
        "Умножение (поэлем.)": lambda: a * b,
        "Транспонирование": lambda: a.transpose(1, 2).contiguous(),
        "Сумма всех элементов": lambda: a.sum(),
    }


def benchmark(shape: tuple, use_cuda: bool) -> list:
    """Возвращает строки таблицы (операция, cpu_ms, gpu_ms, speedup) для заданного размера."""
    a_cpu, b_cpu = make_matrices(shape)
    cpu_times = {name: time_cpu(fn) for name, fn in get_operations(a_cpu, b_cpu).items()}

    gpu_times = {}
    if use_cuda:
        a_gpu, b_gpu = a_cpu.cuda(), b_cpu.cuda()
        gpu_times = {name: time_gpu(fn) for name, fn in get_operations(a_gpu, b_gpu).items()}
        del a_gpu, b_gpu
        torch.cuda.empty_cache()

    rows = []
    for name, cpu_ms in cpu_times.items():
        gpu_ms = gpu_times.get(name)
        speedup = cpu_ms / gpu_ms if gpu_ms else None
        rows.append((name, cpu_ms, gpu_ms, speedup))
    del a_cpu, b_cpu
    return rows


def print_table(shape: tuple, rows: list) -> None:
    """Печатает результаты в табличном виде."""
    print(f"\nРазмер: {shape}")
    print(f"{'Операция':<22}| {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
    print("-" * 60)
    for name, cpu_ms, gpu_ms, speedup in rows:
        gpu_str = f"{gpu_ms:.2f}" if gpu_ms is not None else "н/д"
        sp_str = f"{speedup:.1f}x" if speedup is not None else "н/д"
        print(f"{name:<22}| {cpu_ms:>10.2f} | {gpu_str:>10} | {sp_str:>10}")


def measure_transfer(shape: tuple) -> float:
    """Время копирования одной матрицы CPU -> GPU в мс."""
    a_cpu, _ = make_matrices(shape)
    a_cpu.cuda()                            # прогрев
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    a_cpu.cuda()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ---------- 3.4 Анализ результатов ----------
ANALYSIS = """
Анализ (сверьте с вашей таблицей):
- Наибольшее ускорение обычно даёт матричное умножение: оно вычислительно
  тяжёлое (O(n^3) на матрицу) и хорошо распараллеливается на тысячах ядер GPU.
- Поэлементные операции, сумма и транспонирование упираются в пропускную
  способность памяти, а не в вычисления, поэтому выигрыш у них скромнее.
  На малых данных GPU может быть медленнее: запуск ядра и синхронизация
  стоят времени, а нагрузки недостаточно, чтобы занять все ядра.
- Чем больше матрицы, тем лучше GPU загружен и тем выше ускорение. На самой
  маленькой форме (256x256x256) накладные расходы заметнее.
- Копирование данных CPU <-> GPU идёт по шине PCIe и обычно намного медленнее
  вычислений на самом GPU. Если переносить данные ради одной простой операции,
  выигрыш можно полностью потерять, поэтому данные держат на GPU как можно дольше.
"""


def main() -> None:
    use_cuda = torch.cuda.is_available()
    print("CUDA доступна:", use_cuda)
    if use_cuda:
        print("Устройство:", torch.cuda.get_device_name(0))
    else:
        print("GPU не найден: будут измерены только CPU-времена.")

    for shape in SIZES:
        print_table(shape, benchmark(shape, use_cuda))
        if use_cuda:
            print(f"Копирование одной матрицы CPU -> GPU: {measure_transfer(shape):.2f} мс")
    print(ANALYSIS)


if __name__ == "__main__":
    main()