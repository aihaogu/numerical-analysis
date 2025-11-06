"""Classical fourth-order Runge-Kutta solver for several IVPs."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable, Dict, List, Tuple


@dataclass
class RKResult:
    x: float
    y: float
    exact: float


def rk4_step(f: Callable[[float, float], float], x: float, y: float, h: float) -> float:
    """Perform one classical RK4 step."""
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ivp_rk4(
    f: Callable[[float, float], float],
    exact: Callable[[float], float],
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int,
) -> List[RKResult]:
    """Integrate the ODE and record intermediate states."""
    h = (x_end - x0) / n_steps
    x, y = x0, y0
    results: List[RKResult] = [RKResult(x=x, y=y, exact=exact(x))]
    for _ in range(n_steps):
        y = rk4_step(f, x, y, h)
        x = round(x + h, 12)
        results.append(RKResult(x=x, y=y, exact=exact(x)))
    return results


def format_table(results: List[RKResult]) -> str:
    header = "{:<8s} {:>18s} {:>18s} {:>18s}".format(
        "x", "y (RK4)", "y (Exact)", "Absolute Error"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        error = abs(row.y - row.exact)
        lines.append(
            f"{row.x:<8.3f} {row.y:>18.10f} {row.exact:>18.10f} {error:>18.3e}"
        )
    return "\n".join(lines)


# Problem definitions ----------------------------------------------------------

PROBLEMS: Dict[str, Dict[str, object]] = {
    "Problem (1)": {
        "f": lambda x, y: x + y,
        "exact": lambda x: -x - 1 + 2 * exp(x),
        "initial": (0.0, 1.0),
        "interval": 1.0,
    },
    "Problem (2)": {
        "f": lambda x, y: x - y,
        "exact": lambda x: x - 1 + 2 * exp(-x),
        "initial": (0.0, 1.0),
        "interval": 1.0,
    },
    "Problem (3)": {
        "f": lambda x, y: y - x**2 + 1,
        "exact": lambda x: (x + 1) ** 2 - 0.5 * exp(x),
        "initial": (0.0, 0.5),
        "interval": 1.0,
    },
}


STEP_OPTIONS = [5, 10, 20]


def run_problem(name: str, config: Dict[str, object]) -> None:
    f = config["f"]  # type: ignore[assignment]
    exact = config["exact"]  # type: ignore[assignment]
    x0, y0 = config["initial"]  # type: ignore[assignment]
    x_end = x0 + config["interval"]  # type: ignore[assignment]

    print(f"\n{name}")
    for n in STEP_OPTIONS:
        print(f"\n  N = {n}")
        results = solve_ivp_rk4(f, exact, x0, y0, x_end, n)
        print(format_table(results))


def main() -> None:
    for name, config in PROBLEMS.items():
        run_problem(name, config)


if __name__ == "__main__":
    main()
