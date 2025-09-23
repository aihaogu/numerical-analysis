"""
Broyden's (quasi-Newton) method for solving a 3x3 nonlinear system.
The initial Jacobian uses the analytical derivatives and a backtracking line
search improves global convergence.
"""
from __future__ import annotations
from math import exp, sqrt
from typing import Callable, List, Tuple

Vector = List[float]
Matrix = List[List[float]]


def F(x: Vector) -> Vector:
    a, b, c = x
    return [
        a * b - c**2 - 1.0,
        a * b * c + b**2 - a**2 - 2.0,
        exp(a) + c - exp(b) - 3.0,
    ]


def jacobian(x: Vector) -> Matrix:
    a, b, c = x
    return [
        [b, a, -2.0 * c],
        [b * c - 2.0 * a, a * c + 2.0 * b, a * b],
        [exp(a), -exp(b), 1.0],
    ]


def vector_norm(vec: Vector) -> float:
    return sqrt(sum(component * component for component in vec))


def mat_vec(mat: Matrix, vec: Vector) -> Vector:
    return [sum(mat_row[j] * vec[j] for j in range(len(vec))) for mat_row in mat]


def outer(u: Vector, v: Vector) -> Matrix:
    return [[ui * vj for vj in v] for ui in u]


def add_matrices(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale_matrix(mat: Matrix, scalar: float) -> Matrix:
    return [[scalar * mat[i][j] for j in range(len(mat[0]))] for i in range(len(mat))]


def solve_linear_system(matrix: Matrix, rhs: Vector) -> Vector:
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-14:
            raise ZeroDivisionError("Matrix is singular during Gaussian elimination.")
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
        pivot_value = a[i][i]
        for j in range(i, n):
            a[i][j] /= pivot_value
        b[i] /= pivot_value
        for k in range(i + 1, n):
            factor = a[k][i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]
    x = [0.0 for _ in range(n)]
    for i in reversed(range(n)):
        x[i] = b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))
    return x


def broyden(
    func: Callable[[Vector], Vector],
    x0: Vector,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> Tuple[Vector, int]:
    x = x0[:]
    fx = func(x)
    if vector_norm(fx) <= tolerance:
        return x, 0
    b = jacobian(x)
    for iteration in range(1, max_iterations + 1):
        step_direction = solve_linear_system(b, [-value for value in fx])
        alpha = 1.0
        descent_found = False
        while alpha >= 1e-4:
            s = [alpha * comp for comp in step_direction]
            x_candidate = [x[i] + s[i] for i in range(len(x))]
            try:
                fx_candidate = func(x_candidate)
            except OverflowError:
                alpha *= 0.5
                continue
            if vector_norm(fx_candidate) < vector_norm(fx):
                x_new = x_candidate
                fx_new = fx_candidate
                descent_found = True
                break
            alpha *= 0.5
        if not descent_found:
            raise RuntimeError("Line search failed to find a descent step in Broyden method.")

        if vector_norm(fx_new) <= tolerance:
            return x_new, iteration

        s = [x_new[i] - x[i] for i in range(len(x))]
        y = [fx_new[i] - fx[i] for i in range(len(x))]
        denominator = sum(s_i * s_i for s_i in s)
        if denominator < 1e-14:
            raise ZeroDivisionError("Denominator too small in Broyden update.")
        bs = mat_vec(b, s)
        u = [y_i - bs_i for y_i, bs_i in zip(y, bs)]
        b = add_matrices(b, scale_matrix(outer(u, s), 1.0 / denominator))
        x, fx = x_new, fx_new
    raise RuntimeError("Broyden method did not converge within the maximum iterations.")


if __name__ == "__main__":
    root, iterations = broyden(F, [1.0, 0.1, 0.1])
    print("Approximate solution:")
    print(f"  x = {root[0]:.10f}")
    print(f"  y = {root[1]:.10f}")
    print(f"  z = {root[2]:.10f}")
    print(f"Residual norm: {vector_norm(F(root)):.3e}")
    print(f"Iterations: {iterations}")