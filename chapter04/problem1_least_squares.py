"""Least squares polynomial fitting for the Table 6 data set."""
from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple


def build_normal_equations(
    x_values: Sequence[float], y_values: Sequence[float], degree: int
) -> Tuple[List[List[float]], List[float]]:
    """Construct the normal equations for polynomial least squares fitting."""
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if degree >= len(x_values):
        raise ValueError(
            "Polynomial degree must be less than the number of data points to avoid a singular system."
        )

    m = degree
    size = m + 1
    # Pre-compute sums of powers of x to avoid redundant work.
    power_sums = [0.0 for _ in range(2 * m + 1)]
    for power in range(2 * m + 1):
        power_sums[power] = sum((x ** power) for x in x_values)

    # Construct the symmetric matrix for the normal equations.
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            matrix[row][col] = power_sums[row + col]

    # Construct the right-hand side vector.
    rhs = [0.0 for _ in range(size)]
    for row in range(size):
        rhs[row] = sum((x ** row) * y for x, y in zip(x_values, y_values))

    return matrix, rhs


def gaussian_elimination_partial_pivot(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> List[float]:
    """Solve a linear system using Gaussian elimination with partial pivoting."""
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Coefficient matrix must be square.")
    if len(rhs) != n:
        raise ValueError("Right-hand side vector length must match matrix dimension.")

    # Create working copies of the inputs.
    a = [list(row) for row in matrix]
    b = list(rhs)

    for k in range(n - 1):
        # Select the pivot row based on the largest absolute value in column k.
        pivot_row = max(range(k, n), key=lambda i: abs(a[i][k]))
        pivot_value = a[pivot_row][k]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned; pivot is zero.")
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]

        pivot = a[k][k]
        for i in range(k + 1, n):
            factor = a[i][k] / pivot
            if factor == 0:
                continue
            for j in range(k, n):
                a[i][j] -= factor * a[k][j]
            b[i] -= factor * b[k]

    # Back substitution to solve the upper triangular system.
    solution = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned during back substitution.")
        accumulated = sum(a[i][j] * solution[j] for j in range(i + 1, n))
        solution[i] = (b[i] - accumulated) / pivot

    return solution


def least_squares_polynomial(
    x_values: Sequence[float], y_values: Sequence[float], degree: int
) -> List[float]:
    """Compute the least squares polynomial coefficients of the specified degree."""
    matrix, rhs = build_normal_equations(x_values, y_values, degree)
    return gaussian_elimination_partial_pivot(matrix, rhs)


def evaluate_polynomial(coefficients: Sequence[float], x: float) -> float:
    """Evaluate a polynomial with coefficients in increasing power order at point x."""
    result = 0.0
    power = 1.0
    for coefficient in coefficients:
        result += coefficient * power
        power *= x
    return result


def format_polynomial(coefficients: Sequence[float]) -> str:
    """Create a human-readable string representation of a polynomial."""
    terms: List[Tuple[str, str]] = []
    for power, coeff in enumerate(coefficients):
        if abs(coeff) < 1e-12:
            continue
        magnitude = abs(coeff)
        if power == 0:
            base = f"{magnitude:.6f}"
        elif power == 1:
            base = f"{magnitude:.6f} x"
        else:
            base = f"{magnitude:.6f} x^{power}"
        sign = "+" if coeff >= 0 else "-"
        terms.append((sign, base))

    if not terms:
        return "p(x) = 0"

    sign, base = terms[0]
    expression = base if sign == "+" else f"- {base}"
    for sign, base in terms[1:]:
        expression += f" {sign} {base}"

    return f"p(x) = {expression}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Least squares polynomial fitting for the Table 6 experimental data."
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Degree of the polynomial to fit (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    degree = args.degree
    x_values = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0]
    y_values = [2.01, 2.98, 3.50, 5.05, 5.47, 6.02, 7.05]

    coefficients = least_squares_polynomial(x_values, y_values, degree)
    polynomial = format_polynomial(coefficients)

    print("Fitted polynomial coefficients (constant term first):")
    print([f"{value:.6f}" for value in coefficients])
    print(polynomial)
    print()
    print("Data fitting details:")
    print("    x_i      f_i     p(x_i)   residual")
    residuals = []
    for x, y in zip(x_values, y_values):
        approximation = evaluate_polynomial(coefficients, x)
        residual = y - approximation
        residuals.append(residual)
        print(f"{x:8.2f} {y:9.3f} {approximation:9.4f} {residual:9.4f}")

    sse = sum(residual ** 2 for residual in residuals)
    rmse = math.sqrt(sse / len(x_values))
    print()
    print(f"Sum of squared errors (SSE): {sse:.6f}")
    print(f"Root mean square error (RMSE): {rmse:.6f}")


if __name__ == "__main__":
    main()
