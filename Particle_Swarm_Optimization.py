# -*- coding: UTF-8 -*-
"""
@File    ：Particle_Swarm_Optimization.py
@Author  ：Csy
@Date    ：2024/5/10 14:59 
@Bref    : 粒子群算法  越小越好
@Ref     :
TODO     :
         :
"""

import numpy as np
from matplotlib import pyplot as plt


class Particles:

    def __init__(self,
                 n_iterations,
                 n_particles,
                 n_dims,
                 dim_min,
                 dim_max,
                 fitness_func,
                 c1,
                 c2,
                 w, ):
        self.n_iterations = n_iterations
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.w_max = 0.9
        self.w_min = 0.4
        self.dim_min = dim_min
        self.dim_max = dim_max
        self.fitness_func = fitness_func

        self.particles = np.random.uniform(size=(n_particles, n_dims),
                                           low=dim_min,
                                           high=dim_max)
        self.velocities = np.zeros((n_particles, n_dims))

        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.apply_along_axis(fitness_func, 1, self.particles)

        self.gbest_position = self.particles[np.argmin(self.pbest_fitness)]
        self.gbest_fitness = np.min(self.pbest_fitness)

        self.log_fitness = []

    def update_velocity(self):
        # 粒子速度
        r1 = np.random.rand(self.n_particles, self.n_dims)
        r2 = np.random.rand(self.n_particles, self.n_dims)
        '''
        首先，计算新的速度。计算时考虑了本身的速度（乘以一个惯性权重w）、
        粒子本身的最优经验 （pbest，引导粒子向自己的历史最优位置搜索）和
        群体的最优经验（gbest，引导粒子向群体的历史最优位置搜索）。
        '''
        # self.w = self.w_max-cur_iter*((self.w_max-self.w_min)/self.n_iterations);

        self.velocities = self.w * self.velocities \
                          + self.c1 * r1 * (self.pbest_positions - self.particles) \
                          + self.c2 * r2 * (self.gbest_position - self.particles)

    def update_position(self,adjust=False):
        # update positions
        self.particles += self.velocities

        # adjust 根据 dim_low dim_high 来调整
        # self.particles.clip(min=self.dim_min, max=self.dim_max)

        #
        if adjust:
            self.particles = np.clip(self.particles, a_min=self.dim_min,
                                     a_max=self.dim_max)

    def evaluate(self):

        cur_fitness = np.apply_along_axis(self.fitness_func, 1, self.particles)
        cur_position = self.particles

        # 更新每一个粒子的最优位置
        # update_index = cur_fitness < self.pbest_fitness
        update_index = cur_fitness > self.pbest_fitness
        self.pbest_fitness = np.where(update_index, cur_fitness, self.pbest_fitness)
        # self.pbest_positions = np.where(cur_fitness < self.pbest_fitness, self.particles, self.pbest_positions)
        self.pbest_positions[update_index,:] = self.particles[update_index,:]

        # 更新全局的最优位置
        cur_min_fitness = np.min(cur_fitness)
        # if self.gbest_fitness > cur_min_fitness:
        if self.gbest_fitness < cur_min_fitness:
            self.gbest_fitness = cur_min_fitness
            self.gbest_position = self.particles[np.argmin(self.pbest_fitness)]
        return cur_min_fitness

    def run(self):
        for i in range(self.n_iterations):
            self.update_velocity()
            self.update_position()
            cur_min_fitness = self.evaluate()
            print(f"Iter-{i:<4}", "--> fitness: ", cur_min_fitness)
            self.log_fitness.append(cur_min_fitness)

        print("best solution: ", self.gbest_position, "best fitness: ", self.gbest_fitness)

    def plot(self):
        plt.figure(figsize=(10,8))

        ax = plt.gca()  # 得到图像的Axes对象
        ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
        ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明

        plt.title('PSO Iteration vs. Fitness/Loss')
        plt.plot(range(len(self.log_fitness)),np.array(self.log_fitness),color='green',linewidth=2.0)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.show()


if __name__ == '__main__':
    function_inputs = [4, -2, 3.5, 5, -11, -4.7]
    desired_output = 44


    # 越大越好
    def fitness_func(solution):
        output = np.sum(solution * function_inputs)
        fitness = 1.0 / (np.abs(output - desired_output))
        return fitness


    PSO = Particles(n_iterations=1000,
                    n_particles=100,
                    n_dims=len(function_inputs),
                    dim_max=4, dim_min=-4,
                    fitness_func=fitness_func,
                    c1=2,
                    c2=2,
                    w=0.7
                    )
    PSO.run()
    PSO.plot()