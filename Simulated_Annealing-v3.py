# -*- coding: UTF-8 -*-
"""
@File    ：Simulated_Annealing.py
@Author  ：Csy
@Date    ：2024/5/10 15:02 
@Bref    : 模拟退火算法
@Ref     : https://bhistory.csdn.net/weixin_48241292/article/details/109468947
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


# def func2(solution):
#     x, y = solution
#     energy_ = func(x, y)
#     return energy_


def energy(solution, func_=func):
    '''
    提供的方案 和 要求解的问题 进一步抽象 封装 【也是核心设计的地方】
    :param func_:
    :param solution:
    :return:
    '''
    x, y = solution
    energy_ = func_(x, y)
    return energy_


class SA:
    def __init__(self,
                 func,
                 low,
                 high,
                 n_dim,
                 iterations=100,
                 t0=100,
                 tf=0.01,
                 alpha=0.99,
                 ):
        self.func = func
        self.low = low
        self.high = high
        self.n_dim = n_dim
        self.iterations = iterations
        self.T0 = t0
        self.Tf = tf
        self.T = t0
        self.alpha = alpha

        self.x = np.random.uniform(low, high, size=n_dim)
        self.y = self.func(self.x)

        self.best_x = self.x.copy()
        self.best_y = self.y.copy()

        self.history = {'T': [self.T0],
                        'best_x': [self.x],
                        'best_y': [self.y]}

    def generate_new(self, x):
        '''
        模拟 分子/原子 不稳定的过程 随机到下一个位置
        可以改进的点：使用不同的策略
        :param x:
        :param y:
        :return:
        '''

        delta = self.T * (np.random.rand(self.n_dim) - np.random.rand(self.n_dim))
        x_new = self.x + delta

        return x_new

    def metrospolis(self, f, f_new):
        '''
        Metropolis 准则实现 帮助跳出局部最优
        :param f:
        :param f_new:
        :return:
        '''
        if f_new <= f:
            return 1
        else:
            p = math.exp(-1 * (f_new - f) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def run(self):
        '''
        主降温过程 嵌套粒子的随机过程
        :return:
        '''
        count = 0
        current_x, current_y = self.best_x, self.best_y
        while self.T > self.Tf:

            for i in range(self.iterations):
                x_new = self.generate_new(current_x)  # 粒子随机 跳动
                y_new = self.func(x_new)  # 计算新值
                if self.metrospolis(self.best_y, y_new):  # 判断是否接收新值
                    current_x, current_y = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x = current_x
                        self.best_y = current_y

            # cool down
            self.T *= self.alpha

            # log
            self.history['T'].append(self.T)
            self.history['best_x'].append(self.best_x)
            self.history['best_y'].append(self.best_y)

            count += 1

        print(f'Best function, F={self.best_x}')

    def plot(self):
        plt.title('SA Iterations vs. Energy')
        plt.plot(self.history['T'], self.history['best_y'], color='green', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.gca().invert_xaxis()
        plt.show()


if __name__ == '__main__':
    sa = SA(func=energy, low=-5, high=5, n_dim=2)
    sa.run()
    sa.plot()
