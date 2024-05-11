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
                 low,
                 high,
                 atom_dim,
                 iterations=100,
                 t0=100,
                 tf=0.01,
                 alpha=0.99,
                 ):
        self.func = func
        self.low = low
        self.high = high
        self.iterations = iterations
        self.T0 = t0
        self.Tf = tf
        self.T = t0
        self.alpha = alpha

        self.x = np.random.uniform(low, high, size=iterations)
        self.y = np.random.uniform(low, high, size=iterations)
        
        self.atoms = np.random.uniform(low, high, size=(iterations, atom_dim))

        self.log = {'f': [], 'T': []}

    def generate_new(self, x, y):
        '''
        模拟 分子/原子 不稳定的过程 随机到下一个位置
        :param x:
        :param y:
        :return:
        '''
        while True:
            x_new = x + self.T * (np.random.rand() - np.random.rand())
            y_new = y + self.T * (np.random.rand() - np.random.rand())
            if (self.low <= x_new <= self.high) & (self.low <= y_new <= self.high):
                break
        return x_new, y_new

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

    def best(self):
        '''
        在该温度下 迭代 l 次后
        :return: 目标函数的最优解 和 最优解下标
        '''
        f_list = []
        for i in range(self.iterations):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx

    def run(self):
        '''
        主降温过程 嵌套粒子的随机过程
        :return:
        '''
        count = 0
        while self.T > self.Tf:

            for i in range(self.iterations):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])  # 粒子随机 跳动
                f_new = self.func(x_new, y_new)  # 计算新值
                if self.metrospolis(f, f_new):  # 判断是否接收新值
                    self.x[i] = x_new
                    self.y[i] = y_new

            ft, _ = self.best()
            self.log['f'].append(ft)
            self.log['T'].append(self.T)

            self.T *= self.alpha
            count += 1

        # 最优解
        f_best, idx = self.best()
        print(f'Best function, F={f_best}, x={self.x[idx]},y={self.y[idx]}')

    def plot(self):
        plt.title('SA Iterations vs. Energy')
        plt.plot(self.log['T'], self.log['f'], color='green', linewidth=2)
        plt.xlabel('T')
        plt.ylabel('f')
        plt.gca().invert_xaxis()
        plt.show()


if __name__ == '__main__':
    sa = SA(func, low=-5, high=5)
    sa.run()
    sa.plot()
