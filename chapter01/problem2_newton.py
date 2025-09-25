"""
Newton's method for three nonlinear equations with specified initial guesses.

此文件实现了对若干一元非线性方程使用牛顿法（Newton-Raphson）求根。
我在源码中添加了中文注释用于教学与阅读，不改变程序的行为。
"""

from __future__ import annotations  # 启用延后类型注解（使类型注解在运行时作为字符串），改善前向引用兼容性
from dataclasses import dataclass  # 方便定义简单的数据容器类
from math import exp  # 指数函数 e^x
from typing import Callable, List  # 类型注解：函数类型与列表类型


@dataclass
class NewtonResult:
    """封装 Newton 方法结果的简单数据类。

    属性：
    - root: 求得的根（float）
    - iterations: 使用的迭代次数（int）
    """

    root: float
    iterations: int


def newton_method(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> NewtonResult:
    """
    使用牛顿法求解单变量方程 f(x)=0。

    参数：
    - f: 目标函数 f(x)
    - df: 目标函数的导数 f'(x)
    - x0: 初始猜测值
    - tolerance: 收敛容差，当 |step| <= tolerance 时认为收敛
    - max_iterations: 最大迭代次数，超过则报错

    返回：NewtonResult，包含根与迭代次数。

    重要实现细节（注释）：
    - 每次迭代计算 fx=f(x) 与 dfx=f'(x)，并用 step=fx/dfx 更新 x。
    - 若导数绝对值过小（<1e-14）则抛出 ZeroDivisionError，避免除以接近 0。
    - 收敛判据使用步长的绝对值而不是直接使用 |f(x)|。
    """

    x = x0  # 当前近似值
    for iteration in range(1, max_iterations + 1):  # 从 1 开始计数，便于返回使用的迭代次数
        fx = f(x)  # 计算函数值 f(x)
        dfx = df(x)  # 计算导数 f'(x)
        # 如果导数过小，Newton 更新会数值不稳定（或除以零），因此直接报错
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Derivative is too close to zero; Newton's method fails.")
        step = fx / dfx  # Newton 步长（更新量）
        x -= step  # x_{n+1} = x_n - f(x)/f'(x)
        # 使用步长作为收敛判据：当更新量很小时，认为已经到达根附近
        if abs(step) <= tolerance:
            return NewtonResult(root=x, iterations=iteration)
    # 超过最大迭代次数仍未收敛，抛出异常以便上层处理
    raise RuntimeError("Newton's method did not converge within the maximum iterations.")


def main() -> None:
    """
    定义要求解的方程以及对应的导数和初始猜测，并逐个运行 Newton 方法。
    """

    # tasks 列表每个元素包含：说明字符串、函数 f、导数 df、初始猜测列表
    tasks: List[tuple[str, Callable[[float], float], Callable[[float], float], List[float]]] = [
        (
            "x * e^x - 1 = 0",
            lambda x: x * exp(x) - 1.0,  # f(x) = x e^x - 1
            lambda x: exp(x) * (x + 1.0),  # f'(x) = e^x (x + 1)
            [0.5],  # 初始猜测
        ),
        (
            "x^3 - x - 1 = 0",
            lambda x: x**3 - x - 1.0,  # f(x) = x^3 - x - 1
            lambda x: 3.0 * x**2 - 1.0,  # f'(x) = 3x^2 - 1
            [1.0],
        ),
        (
            "(x - 1)^2 * (2x - 1) = 0",
            lambda x: (x - 1.0) ** 2 * (2.0 * x - 1.0),
            # 对 f(x) 求导：使用乘积与链式法则展开（见下式）
            lambda x: 2.0 * (x - 1.0) * (2.0 * x - 1.0) + 2.0 * (x - 1.0) ** 2,
            # 此方程有一个重根 x=1（二阶）和另一根 x=1/2，因此用不同初始猜测观察收敛行为
            [0.45, 0.65],
        ),
    ]

    for description, f, df, initial_guesses in tasks:
        print(f"Equation: {description}")
        for guess in initial_guesses:
            try:
                # 调用 newton_method 并获取结果
                result = newton_method(f, df, guess)
                print(
                    f"  Initial guess {guess:.2f} -> root {result.root:.10f} "
                    f"after {result.iterations} iterations"
                )
            except (ZeroDivisionError, RuntimeError) as exc:
                # 捕获导数过小或未收敛的异常并打印信息
                print(f"  Initial guess {guess:.2f} -> method failed: {exc}")
        print()  # 方程之间空行分隔输出


if __name__ == "__main__":
    main()