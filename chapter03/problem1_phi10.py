"""Evaluate the degree-10 interpolating polynomial for f(x)=1/(1+25x^2)."""
from __future__ import annotations

from typing import Iterable, Sequence


def f(x: float) -> float:
    """Runge function used in the experiment."""
    return 1.0 / (1.0 + 25.0 * x * x)


def uniform_nodes(a: float, b: float, n: int) -> list[float]:
    """Return n+1 uniformly spaced nodes on [a, b]."""
    step = (b - a) / n
    return [a + step * j for j in range(n + 1)]


def barycentric_weights(nodes: Sequence[float]) -> list[float]:
    """Compute first-form barycentric weights for the given nodes."""
    w: list[float] = []
    for j, xj in enumerate(nodes):
        product = 1.0
        for k, xk in enumerate(nodes):
            if k != j:
                product *= xj - xk
        w.append(1.0 / product)
    return w


def barycentric_interpolate(
    nodes: Sequence[float], values: Sequence[float], weights: Sequence[float], x: float
) -> float:
    """Evaluate the interpolating polynomial at x using the barycentric formula."""
    numerator = 0.0
    denominator = 0.0
    for xj, yj, wj in zip(nodes, values, weights):
        diff = x - xj
        if abs(diff) < 1e-12:
            return yj
        term = wj / diff
        numerator += term * yj
        denominator += term
    return numerator / denominator


def sample_grid(a: float, b: float, count: int) -> Iterable[float]:
    """Generate evenly spaced sample points on [a, b]."""
    if count == 1:
        yield a
        return
    step = (b - a) / (count - 1)
    for i in range(count):
        yield a + i * step


def main() -> None:
    a, b = -1.0, 1.0
    n = 10
    nodes = uniform_nodes(a, b, n)
    values = [f(x) for x in nodes]
    weights = barycentric_weights(nodes)

    grid = list(sample_grid(a, b, 1001))
    max_error = -1.0
    x_at_max = grid[0]
    phi_at_max = 0.0

    for x in grid:
        phi_x = barycentric_interpolate(nodes, values, weights, x)
        error = abs(phi_x - f(x))
        if error > max_error:
            max_error = error
            x_at_max = x
            phi_at_max = phi_x

    print("Degree-10 polynomial interpolation for f(x) = 1/(1+25x^2)")
    print(f"Nodes: {n + 1} equally spaced points on [{a}, {b}]")
    print(f"Maximum absolute error on {len(grid)}-point grid: {max_error:.6e}")
    print(f"Location of maximum error: x = {x_at_max:.5f}")
    print(f"Interpolant at x: {phi_at_max:.6e}, f(x): {f(x_at_max):.6e}")

    probe_points = [-0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.95]
    print("\nComparison at selected points:")
    print("    x        phi_10(x)      f(x)        |error|")
    for x in probe_points:
        phi_x = barycentric_interpolate(nodes, values, weights, x)
        exact = f(x)
        error = abs(phi_x - exact)
        print(f"{x:7.2f}  {phi_x:12.8f}  {exact:12.8f}  {error:9.2e}")


if __name__ == "__main__":
    main()
