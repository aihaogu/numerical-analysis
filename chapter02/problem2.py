"""Solve the linear system for problem (2) using Gaussian elimination variants."""
from __future__ import annotations

from typing import List


def gaussian_elimination(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax=b using plain Gaussian elimination without pivoting."""
    n = len(a)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for k in range(n - 1):
        pivot = aug[k][k]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError(
                f"Zero pivot encountered at row {k}. Try a pivoting strategy."
            )
        for i in range(k + 1, n):
            factor = aug[i][k] / pivot
            for j in range(k, n + 1):
                aug[i][j] -= factor * aug[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        rhs = aug[i][n]
        for j in range(i + 1, n):
            rhs -= aug[i][j] * x[j]
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError(f"Zero pivot encountered at row {i} during back substitution.")
        x[i] = rhs / pivot
    return x


def gaussian_elimination_partial_pivot(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax=b using Gaussian elimination with column (partial) pivoting."""
    n = len(a)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for k in range(n - 1):
        pivot_row = max(range(k, n), key=lambda i: abs(aug[i][k]))
        if abs(aug[pivot_row][k]) < 1e-12:
            raise ZeroDivisionError(f"Column {k} has all zeros; system may be singular.")
        if pivot_row != k:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
        pivot = aug[k][k]
        for i in range(k + 1, n):
            factor = aug[i][k] / pivot
            for j in range(k, n + 1):
                aug[i][j] -= factor * aug[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        rhs = aug[i][n]
        for j in range(i + 1, n):
            rhs -= aug[i][j] * x[j]
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError(f"Zero pivot encountered at row {i} during back substitution.")
        x[i] = rhs / pivot
    return x


def main() -> None:
    a = [
        [4.0, -2.0, 0.0],
        [4.0, 4.0, -3.0],
        [2.0, -2.0, 1.0],
    ]
    b = [2.0, 11.0, 1.0]

    x_plain = gaussian_elimination(a, b)
    x_pivot = gaussian_elimination_partial_pivot(a, b)

    print("Problem (2)")
    print("Plain Gaussian elimination solution:")
    print([f"{value:.10f}" for value in x_plain])
    print("Gaussian elimination with column pivoting solution:")
    print([f"{value:.10f}" for value in x_pivot])
    diff = [abs(u - v) for u, v in zip(x_plain, x_pivot)]
    print("Absolute differences between the two solutions:")
    print([f"{value:.3e}" for value in diff])


if __name__ == "__main__":
    main()
