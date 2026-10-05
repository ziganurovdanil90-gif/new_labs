"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.matmul(matrices, vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(col).tolist() for col in np.asarray(matrix).T]


def matrix_statistics(data: MatrixStatisticsInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(loc=data.mean, scale=data.std, size=(data.m, data.n))

    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )
import matplotlib.pyplot as plt


def plot_histograms(stats: MatrixStatistics) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(stats.matrix.flatten(), bins=20)
    axes[0].set_title("Значения всей матрицы")

    axes[1].hist(stats.row_means, bins=10, alpha=0.6, label="Строки")
    axes[1].hist(stats.column_means, bins=10, alpha=0.6, label="Столбцы")
    axes[1].set_title("Средние по строкам и столбцам")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    top = (image_height - height) // 2
    left = (image_width - width) // 2
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    y, x = np.indices((image_height, image_width))
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2

    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[mask] = shape_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)

    mean = values.mean()
    variance = values.var()
    std = values.std()

    prev, curr, nxt = values[:-2], values[1:-1], values[2:]
    local_maxima = np.where((curr > prev) & (curr > nxt))[0] + 1
    local_minima = np.where((curr < prev) & (curr < nxt))[0] + 1

    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima=local_maxima,
        local_minima=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)

    if class_count is None:
        class_count = labels.max() + 1

    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result