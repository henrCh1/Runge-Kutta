# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def rk4(f, t, y, h):
    """
    四阶龙格-库塔方法
    """
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    y1 = y + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y1

def ode_solver(f, y0, t0, tn, h):
    # 计算时间步数
    N = int((tn - t0) / h)

    # 初始化时间和解向量
    t = np.linspace(t0, tn, N+1)
    y = np.zeros(N+1)
    y[0] = y0

    # 迭代计算解向量
    for i in range(N):
        # 使用指定的数值方法
        y[i+1] = rk4(f, t[i], y[i], h)

    return t, y

# 定义微分方程dy/dt=f(t,y)
def f(t, y):
    return np.exp(t)-2*t*y2-t**2*y1

# 求解微分方程y' =1+exp(-t)*sin(y)，y(0) = 0的数值解
t, y = ode_solver(f, 0, 0, 1, 0.1)

# 输出数值解
print("数值解：", y)

# 绘制数值解
plt.plot(t, y, '-o')
plt.xlabel('t')
plt.ylabel('y')
plt.title("y' =1+exp(-t)*sin(y), y(0) = 0")
plt.show()