"""
Bisection method for solving sin(x) - x**2 / 2 = 0 on (1, 2).
"""
from math import sin


def f(x: float) -> float:
    return sin(x) - (x ** 2) / 2


def bisection(a: float, b: float, tolerance: float, max_iterations: int = 100) -> tuple[float, int]:
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        raise ValueError("Function must have opposite signs at interval endpoints.")

    iteration = 0
    while (b - a) / 2 > tolerance and iteration < max_iterations:
        iteration += 1
        midpoint = (a + b) / 2
        fm = f(midpoint)
        if fm == 0:
            return midpoint, iteration
        if fa * fm < 0:
            b = midpoint
            fb = fm
        else:
            a = midpoint
            fa = fm
    return (a + b) / 2, iteration


if __name__ == "__main__":
    root, iterations = bisection(1.0, 2.0, tolerance=5e-6)
    print(f"Approximate root: {root:.10f}")
    print(f"f(root): {f(root):.3e}")
    print(f"Iterations: {iterations}")