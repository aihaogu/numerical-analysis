"""Fourth-order Adams-Bashforth-Moulton PECE solver for IVPs."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable, Dict, List


@dataclass
class Step:
    x: float
    y: float
    exact: float


def rk4_step(f: Callable[[float, float], float], x: float, y: float, h: float) -> float:
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def adams_pece(
    f: Callable[[float, float], float],
    exact: Callable[[float], float],
    x0: float,
    y0: float,
    x_end: float,
    step_count: int,
) -> List[Step]:
    """Solve the IVP using the fourth-order Adams PECE scheme."""
    h = (x_end - x0) / step_count
    results: List[Step] = []

    # Bootstrap using RK4 to obtain the first three points.
    x_values = [x0]
    y_values = [y0]
    x, y = x0, y0
    for _ in range(3):
        y = rk4_step(f, x, y, h)
        x = round(x + h, 12)
        x_values.append(x)
        y_values.append(y)

    # Record bootstrap results
    for xv, yv in zip(x_values, y_values):
        results.append(Step(x=xv, y=yv, exact=exact(xv)))

    # Predictor-corrector iterations
    for n in range(3, step_count):
        x_n = x_values[n]
        y_n = y_values[n]
        # Adams-Bashforth predictor (four-step)
        f_n = f(x_values[n], y_values[n])
        f_n1 = f(x_values[n - 1], y_values[n - 1])
        f_n2 = f(x_values[n - 2], y_values[n - 2])
        f_n3 = f(x_values[n - 3], y_values[n - 3])
        y_predict = y_n + (h / 24.0) * (
            55 * f_n - 59 * f_n1 + 37 * f_n2 - 9 * f_n3
        )

        x_next = round(x_n + h, 12)
        # Adams-Moulton corrector
        f_predict = f(x_next, y_predict)
        y_correct = y_n + (h / 24.0) * (
            9 * f_predict + 19 * f_n - 5 * f_n1 + f_n2
        )

        x_values.append(x_next)
        y_values.append(y_correct)
        results.append(Step(x=x_next, y=y_correct, exact=exact(x_next)))

    return results


def format_table(results: List[Step]) -> str:
    header = "{:<8s} {:>18s} {:>18s} {:>18s}".format(
        "x", "y (Adams)", "y (Exact)", "Absolute Error"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        error = abs(row.y - row.exact)
        lines.append(
            f"{row.x:<8.3f} {row.y:>18.10f} {row.exact:>18.10f} {error:>18.3e}"
        )
    return "\n".join(lines)


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

STEP_COUNT = 10  # use h = 0.1 on [0, 1]


def run_problem(name: str, config: Dict[str, object]) -> None:
    f = config["f"]  # type: ignore[assignment]
    exact = config["exact"]  # type: ignore[assignment]
    x0, y0 = config["initial"]  # type: ignore[assignment]
    x_end = x0 + config["interval"]  # type: ignore[assignment]

    print(f"\n{name}")
    results = adams_pece(f, exact, x0, y0, x_end, STEP_COUNT)
    print(format_table(results))


def main() -> None:
    for name, config in PROBLEMS.items():
        run_problem(name, config)


if __name__ == "__main__":
    main()
