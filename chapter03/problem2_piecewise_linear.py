"""Piecewise linear interpolation of f(x)=1/(1+25x^2) on [-1, 1]."""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Sequence


def f(x: float) -> float:
    return 1.0 / (1.0 + 25.0 * x * x)


def uniform_nodes(a: float, b: float, n: int) -> list[float]:
    step = (b - a) / n
    return [a + step * j for j in range(n + 1)]


def sample_grid(a: float, b: float, count: int) -> Iterable[float]:
    if count == 1:
        yield a
        return
    step = (b - a) / (count - 1)
    for i in range(count):
        yield a + i * step


def piecewise_linear_interpolate(nodes: Sequence[float], values: Sequence[float], x: float) -> float:
    if x <= nodes[0]:
        return values[0]
    if x >= nodes[-1]:
        return values[-1]
    index = bisect_right(nodes, x) - 1
    x0, x1 = nodes[index], nodes[index + 1]
    y0, y1 = values[index], values[index + 1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def compute_error_metrics(n: int, grid_size: int = 2001) -> tuple[float, float]:
    a, b = -1.0, 1.0
    nodes = uniform_nodes(a, b, n)
    values = [f(x) for x in nodes]
    errors = []
    for x in sample_grid(a, b, grid_size):
        approx = piecewise_linear_interpolate(nodes, values, x)
        exact = f(x)
        errors.append(approx - exact)
    max_abs = max(abs(e) for e in errors)
    rms = (sum(e * e for e in errors) / len(errors)) ** 0.5
    return max_abs, rms


def main() -> None:
    n_values = [10, 20, 40]
    print("Piecewise linear interpolation errors for f(x) = 1/(1+25x^2)")
    print("N   max|error|     RMS error")
    for n in n_values:
        max_abs, rms = compute_error_metrics(n)
        print(f"{n:2d}  {max_abs:11.4e}  {rms:11.4e}")


if __name__ == "__main__":
    main()
