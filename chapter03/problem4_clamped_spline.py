"""Clamped cubic spline interpolation for f(x)=1/(1+25x^2)."""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Sequence


def f(x: float) -> float:
    return 1.0 / (1.0 + 25.0 * x * x)


def f_prime(x: float) -> float:
    return -50.0 * x / (1.0 + 25.0 * x * x) ** 2


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


def solve_tridiagonal(
    lower: Sequence[float], diag: Sequence[float], upper: Sequence[float], rhs: Sequence[float]
) -> list[float]:
    n = len(diag)
    if len(lower) != n - 1 or len(upper) != n - 1 or len(rhs) != n:
        raise ValueError("Incompatible dimensions for tridiagonal system.")
    if n == 1:
        if abs(diag[0]) < 1e-14:
            raise ZeroDivisionError("Singular tridiagonal system.")
        return [rhs[0] / diag[0]]

    c_prime = [0.0] * (n - 1)
    d_prime = [0.0] * n

    denom = diag[0]
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("Singular tridiagonal system.")
    c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < 1e-14:
            raise ZeroDivisionError("Singular tridiagonal system.")
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    x = [0.0] * n
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def clamped_cubic_second_derivatives(nodes: Sequence[float], values: Sequence[float]) -> list[float]:
    n = len(nodes) - 1
    if n < 1:
        raise ValueError("Need at least two nodes for spline interpolation.")
    if n == 1:
        h0 = nodes[1] - nodes[0]
        slope = (values[1] - values[0]) / h0
        m0 = 6.0 * (slope - f_prime(nodes[0])) / h0
        m1 = 6.0 * (f_prime(nodes[1]) - slope) / h0
        return [m0 / 2.0, m1 / 2.0]

    h = [nodes[i + 1] - nodes[i] for i in range(n)]
    size = n + 1
    lower = [0.0] * (size - 1)
    diag = [0.0] * size
    upper = [0.0] * (size - 1)
    rhs = [0.0] * size

    diag[0] = 2.0 * h[0]
    upper[0] = h[0]
    rhs[0] = 6.0 * ((values[1] - values[0]) / h[0] - f_prime(nodes[0]))

    for i in range(1, n):
        lower[i - 1] = h[i - 1]
        diag[i] = 2.0 * (h[i - 1] + h[i])
        upper[i] = h[i]
        rhs[i] = 6.0 * (
            (values[i + 1] - values[i]) / h[i]
            - (values[i] - values[i - 1]) / h[i - 1]
        )

    lower[n - 1] = h[n - 1]
    diag[n] = 2.0 * h[n - 1]
    rhs[n] = 6.0 * (f_prime(nodes[n]) - (values[n] - values[n - 1]) / h[n - 1])

    return solve_tridiagonal(lower, diag, upper, rhs)


def cubic_spline_interpolate(
    nodes: Sequence[float], values: Sequence[float], second_derivatives: Sequence[float], x: float
) -> float:
    if x <= nodes[0]:
        interval = 0
    elif x >= nodes[-1]:
        interval = len(nodes) - 2
    else:
        interval = bisect_right(nodes, x) - 1

    x0 = nodes[interval]
    x1 = nodes[interval + 1]
    y0 = values[interval]
    y1 = values[interval + 1]
    m0 = second_derivatives[interval]
    m1 = second_derivatives[interval + 1]
    h = x1 - x0
    if h == 0.0:
        raise ZeroDivisionError("Repeated nodes encountered.")
    t = x1 - x
    u = x - x0
    return (
        m0 * (t ** 3) / (6.0 * h)
        + m1 * (u ** 3) / (6.0 * h)
        + (y0 - m0 * h * h / 6.0) * (t / h)
        + (y1 - m1 * h * h / 6.0) * (u / h)
    )


def compute_error_metrics(n: int, grid_size: int = 2001) -> tuple[float, float]:
    a, b = -1.0, 1.0
    nodes = uniform_nodes(a, b, n)
    values = [f(x) for x in nodes]
    second_derivatives = clamped_cubic_second_derivatives(nodes, values)
    errors = []
    for x in sample_grid(a, b, grid_size):
        approx = cubic_spline_interpolate(nodes, values, second_derivatives, x)
        exact = f(x)
        errors.append(approx - exact)
    max_abs = max(abs(e) for e in errors)
    rms = (sum(e * e for e in errors) / len(errors)) ** 0.5
    return max_abs, rms


def main() -> None:
    n_values = [10, 20, 40]
    print("Clamped cubic spline interpolation errors for f(x) = 1/(1+25x^2)")
    print("N   max|error|     RMS error")
    for n in n_values:
        max_abs, rms = compute_error_metrics(n)
        print(f"{n:2d}  {max_abs:11.4e}  {rms:11.4e}")


if __name__ == "__main__":
    main()
