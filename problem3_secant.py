"""
Secant method for solving x * e^x - 1 = 0 with initial guesses 0.4 and 0.6.
"""
from math import exp


def f(x: float) -> float:
    return x * exp(x) - 1.0


def secant(x0: float, x1: float, tolerance: float = 1e-8, max_iterations: int = 50):
    f0 = f(x0)
    f1 = f(x1)
    for iteration in range(1, max_iterations + 1):
        if abs(f1 - f0) < 1e-14:
            raise ZeroDivisionError("Secant method denominator is too small.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) <= tolerance:
            return x2, iteration
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    raise RuntimeError("Secant method did not converge within the maximum iterations.")


if __name__ == "__main__":
    root, iterations = secant(0.4, 0.6)
    print(f"Approximate root: {root:.10f}")
    print(f"f(root): {f(root):.3e}")
    print(f"Iterations: {iterations}")