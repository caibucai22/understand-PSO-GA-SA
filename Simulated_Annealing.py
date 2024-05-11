# -*- coding: UTF-8 -*-
"""
@File    ：Simulated_Annealing.py
@Author  ：Csy
@Date    ：2024/5/10 15:02 
@Bref    : 模拟退火算法
@Ref     : https://blog.csdn.net/weixin_48241292/article/details/109468947
TODO     :
         :
"""

import math
import numpy as np
from random import random
from matplotlib import pyplot as plt


def func(x, y):  # 函数优化问题
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


class SA:
    def __init__(self,
                 func,
                 iterations=100,
                 t0=100,
                 tf=0.01,
                 alpha=0.99,
                 ):
        self.func = func
        self.iterations = iterations
        self.T0 = t0
        self.Tf = tf
        self.T = t0
        self.alpha = alpha

        self.x = [np.random.rand() * 11 - 5 for i in range(iterations)]
        self.y = [np.random.rand() * 11 - 5 for i in range(iterations)]
        self.most_best = []
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break
        return x_new, y_new

    def metrospolis(self, f, f_new):
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):
        f_list = []
        for i in range(self.iterations):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx  # f_best idx 在该温度下 迭代 l 次后 目标函数的最优解 和 最优解下标

    def run(self):
        count = 0
        while self.T > self.Tf:

            for i in range(self.iterations):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])  # 产生新解
                f_new = self.func(x_new, y_new)  # 产生新值
                if self.metrospolis(f, f_new):  # 判断是否接收新值
                    self.x[i] = x_new
                    self.y[i] = y_new

            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)

            self.T *= self.alpha
            count += 1

        # 最优解
        f_best, idx = self.best()
        print(f'Best function, F={f_best}, x={self.x[idx]},y={self.y[idx]}')


if __name__ == '__main__':
    sa = SA(func)
    sa.run()

    plt.plot(sa.history['T'], sa.history['f'])
    plt.title('SA')
    plt.xlabel('T')
    plt.ylabel('f')
    plt.gca().invert_xaxis()
    plt.show()
