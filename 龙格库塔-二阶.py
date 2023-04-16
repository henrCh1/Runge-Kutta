# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:17:08 2023

@author: 86319
"""

import numpy as np

def runge_kutta_second_order(f, y0, y1, x0, x1, h):
    """
    使用龙格库塔方法求解二阶微分方程

    参数：
        f: 二阶微分方程的函数，例如 y'' = f(x, y, y')
        y0, y1: 初始条件，即 y(x0) 和 y'(x0)
        x0, x1: 自变量的起始值和终止值
        h: 步长

    返回值：
        x: 自变量数组
        y: 因变量数组，即 y(x) 的值
    """
    x = np.arange(x0, x1 + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    y[1] = y1

    for i in range(1, len(x) - 1):
        k1 = h * y[i]
        l1 = h * f(x[i], y[i], y[i-1])
        k2 = h * (y[i] + l1/2)
        l2 = h * f(x[i] + h/2, y[i] + k1/2, y[i-1] + l1/2)
        k3 = h * (y[i] + l2/2)
        l3 = h * f(x[i] + h/2, y[i] + k2/2, y[i-1] + l2/2)
        k4 = h * (y[i] + l3)
        l4 = h * f(x[i+1], y[i] + k3, y[i-1] + l3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x, y

# 示例：求解 y'' + 2y' + 2y = 0, y(0) = 1, y'(0) = 0 在区间 [0, 10] 上的近似解
def f(x, y, y_prime):
    return np.exp(x)-2*x*y_prime-x**2*y

x, y = runge_kutta_second_order(f, y0=1, y1=-1, x0=0, x1=1, h=0.01)

# 输出结果
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
