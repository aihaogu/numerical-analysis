"""
Improved Newton method for (x - 1)^2 * (2x - 1) = 0 with initial guess 0.55.
The update formula accelerates convergence near multiple roots:
    x_{k+1} = x_k - f(x_k) f'(x_k) / ([f'(x_k)]^2 - f(x_k) f''(x_k)).
"""

def f(x: float) -> float:
    return (x - 1.0) ** 2 * (2.0 * x - 1.0)


def df(x: float) -> float:
    return 2.0 * (x - 1.0) * (3.0 * x - 2.0)


def ddf(x: float) -> float:
    return 12.0 * x - 10.0


def improved_newton(
    x0: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
):
    x = x0
    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        if abs(fx) <= tolerance:
            return x, iteration - 1
        dfx = df(x)
        ddfx = ddf(x)
        denominator = dfx * dfx - fx * ddfx
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError("Denominator too small in improved Newton update.")
        step = fx * dfx / denominator
        x -= step
        if abs(step) <= tolerance:
            return x, iteration
    raise RuntimeError("Improved Newton method did not converge within the maximum iterations.")


if __name__ == "__main__":
    root, iterations = improved_newton(0.55)
    print(f"Approximate root: {root:.10f}")
    print(f"f(root): {f(root):.3e}")
    print(f"Iterations: {iterations}")