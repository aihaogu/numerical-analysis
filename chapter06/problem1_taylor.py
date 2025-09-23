"""Fourth-order Taylor method for a scalar initial value problem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class StepResult:
    """Container for a single integration step."""

    x: float
    y: float
    exact: float


# Problem-specific definitions -------------------------------------------------

def f(x: float, y: float) -> float:
    """Right-hand side f(x, y) = y - x^2 + 1."""
    return y - x**2 + 1.0


def second_derivative(x: float, y: float) -> float:
    """Return y'' evaluated using the differential equation."""
    return -2.0 * x + f(x, y)


def third_derivative(x: float, y: float) -> float:
    """Return y''' based on the analytical differentiation."""
    return second_derivative(x, y) - 2.0


def fourth_derivative(x: float, y: float) -> float:
    """Return y'''' evaluated analytically."""
    return third_derivative(x, y)


def exact_solution(x: float) -> float:
    """Exact solution (x + 1)^2 - 0.5 * e^x."""
    from math import exp

    return (x + 1) ** 2 - 0.5 * exp(x)


# Numerical method -------------------------------------------------------------

def taylor4_step(x: float, y: float, h: float) -> float:
    """Advance one step of size ``h`` using the fourth-order Taylor method."""
    y1 = f(x, y)
    y2 = second_derivative(x, y)
    y3 = third_derivative(x, y)
    y4 = fourth_derivative(x, y)

    return y + h * y1 + (h**2 / 2.0) * y2 + (h**3 / 6.0) * y3 + (h**4 / 24.0) * y4


def solve_taylor4(
    x0: float,
    y0: float,
    h: float,
    steps: int,
    exact: Callable[[float], float],
) -> List[StepResult]:
    """Compute a table of approximations using the Taylor method."""
    results: List[StepResult] = []
    x, y = x0, y0
    results.append(StepResult(x=x, y=y, exact=exact(x)))
    for _ in range(steps):
        y = taylor4_step(x, y, h)
        x = round(x + h, 10)  # mitigate floating point drift
        results.append(StepResult(x=x, y=y, exact=exact(x)))
    return results


def format_table(rows: List[StepResult]) -> str:
    """Create a formatted table for console output."""
    header = "{:<8s} {:>18s} {:>18s} {:>18s}".format(
        "x", "y (Taylor)", "y (Exact)", "Absolute Error"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        error = abs(row.y - row.exact)
        lines.append(
            f"{row.x:<8.3f} {row.y:>18.10f} {row.exact:>18.10f} {error:>18.3e}"
        )
    return "\n".join(lines)


def main() -> None:
    x0, y0 = 0.0, 0.5
    h = 0.1
    steps = 5  # compute up to x = 0.5
    results = solve_taylor4(x0, y0, h, steps, exact_solution)

    print("Fourth-order Taylor method for y' = y - x^2 + 1, y(0) = 0.5")
    print(format_table(results))


if __name__ == "__main__":
    main()
