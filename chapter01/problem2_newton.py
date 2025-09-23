"""
Newton's method for three nonlinear equations with specified initial guesses.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp
from typing import Callable, List


@dataclass
class NewtonResult:
    root: float
    iterations: int


def newton_method(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> NewtonResult:
    x = x0
    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Derivative is too close to zero; Newton's method fails.")
        step = fx / dfx
        x -= step
        if abs(step) <= tolerance:
            return NewtonResult(root=x, iterations=iteration)
    raise RuntimeError("Newton's method did not converge within the maximum iterations.")


def main() -> None:
    tasks: List[tuple[str, Callable[[float], float], Callable[[float], float], List[float]]] = [
        (
            "x * e^x - 1 = 0",
            lambda x: x * exp(x) - 1.0,
            lambda x: exp(x) * (x + 1.0),
            [0.5],
        ),
        (
            "x^3 - x - 1 = 0",
            lambda x: x**3 - x - 1.0,
            lambda x: 3.0 * x**2 - 1.0,
            [1.0],
        ),
        (
            "(x - 1)^2 * (2x - 1) = 0",
            lambda x: (x - 1.0) ** 2 * (2.0 * x - 1.0),
            lambda x: 2.0 * (x - 1.0) * (2.0 * x - 1.0) + 2.0 * (x - 1.0) ** 2,
            [0.45, 0.65],
        ),
    ]

    for description, f, df, initial_guesses in tasks:
        print(f"Equation: {description}")
        for guess in initial_guesses:
            try:
                result = newton_method(f, df, guess)
                print(
                    f"  Initial guess {guess:.2f} -> root {result.root:.10f} "
                    f"after {result.iterations} iterations"
                )
            except (ZeroDivisionError, RuntimeError) as exc:
                print(f"  Initial guess {guess:.2f} -> method failed: {exc}")
        print()


if __name__ == "__main__":
    main()