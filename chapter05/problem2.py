"""Compute the integral of sin(x) / x over [0, 1] using Romberg integration."""
from __future__ import annotations

from math import sin
from typing import Callable, List, Tuple


def romberg_integration(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_level: int = 8,
    tol: float = 1e-10,
) -> Tuple[float, List[List[float]]]:
    """Approximate the integral of ``f`` from ``a`` to ``b`` using Romberg integration."""
    if a == b:
        return 0.0, [[0.0]]

    table: List[List[float]] = []
    initial = 0.5 * (b - a) * (f(a) + f(b))
    table.append([initial])

    for level in range(1, max_level + 1):
        num_new_points = 2 ** (level - 1)
        h = (b - a) / (2**level)
        new_points_sum = sum(f(a + (2 * k - 1) * h) for k in range(1, num_new_points + 1))
        row: List[float] = [0.5 * table[level - 1][0] + h * new_points_sum]

        for k in range(1, level + 1):
            factor = 4**k
            refined = row[k - 1] + (row[k - 1] - table[level - 1][k - 1]) / (factor - 1)
            row.append(refined)

        table.append(row)

        if abs(row[-1] - table[level - 1][-1]) < tol:
            return row[-1], table

    return table[-1][-1], table


def integrand(x: float) -> float:
    """Integrand sin(x) / x with the removable singularity handled at x=0."""
    if x == 0.0:
        return 1.0
    return sin(x) / x


def main() -> None:
    a, b = 0.0, 1.0
    approximate, table = romberg_integration(integrand, a, b)

    print("Romberg integration table (problem 2):")
    for i, row in enumerate(table):
        print(f"Level {i}: " + ", ".join(f"{value:.10f}" for value in row))

    print(f"\nApproximate integral: {approximate:.10f}")


if __name__ == "__main__":
    main()
