# -*- coding: UTF-8 -*-
"""
@File    ：gene_algorithm.py
@Author  ：Csy
@Date    ：2024/5/10 13:38 
@Bref    : 遗传算法
@Ref     : P
TODO     :
         :
"""

import numpy as np
import pandas as pd
import pygad

np.random.seed(42)

num, l_new, w_new, h_new = 30, 2500, 150, 140
order_ = (num, l_new, w_new, h_new)

data = pd.read_excel(r'E:/01-LabProjects/华龙匹配算法/data.xlsx')
l_w_h_cols = data[['长', '宽', '高']]
v = pd.DataFrame({'v': l_w_h_cols.apply(lambda row: row[0] * row[1] * row[2], axis=1)})
v.to_csv('./v.csv')

def filter_data(data, order):
    l_w_h_cols = data[['长', '宽', '高']]
    v = pd.DataFrame({'v': l_w_h_cols.apply(lambda row: row[0] * row[1] * row[2], axis=1)})

    def filter_by_v(order):
        _, l_new, w_new, h_new = order
        v_new = l_new * w_new * h_new
        return v[v['v'] >= v_new].index, v[v['v'] >= v_new]

    def sort_df(df):
        df_sorted = df.apply(lambda row: sorted(row, reverse=True), axis=1)
        df_new = pd.DataFrame(df_sorted.values.tolist(), index=df.index, columns=df.columns)
        return df_new

    def filter_by_l_w_h(order, sorted_lwh):
        _, l_new, w_new, h_new = order
        l_new, w_new, h_new = sorted([l_new, w_new, h_new], reverse=True)
        ret = sorted_lwh[(sorted_lwh['长'] >= l_new) & (sorted_lwh['宽'] >= w_new) & (sorted_lwh['高'] >= h_new)]
        return ret.index, ret

    index1, v_match = filter_by_v(order)
    sorted_lwh = sort_df(l_w_h_cols)
    index2, l_w_h = filter_by_l_w_h(order, sorted_lwh=sorted_lwh)

    base_choices = set(index2).intersection(set(index1))
    return list(base_choices)


choice_idx = filter_data(data, order_)
print(len(choice_idx))
print(choice_idx)
if (len(choice_idx) == 0):
    exit()
choices = v['v'][choice_idx]
choices.to_csv('./base_choices.csv')

function_inputs = choices


def on_start(ga_instance):
    print("on_start()")


def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")


def on_parents(ga_instance, selected_parents):
    print("on_parents()")


def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")


def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")
    # offspring_mutation = ga_instance.solutions
    print("begin detect ...")
    for i, offspring in enumerate(offspring_mutation):
        if np.count_nonzero(offspring) == 0:
            print('detect all 0')
            # 保证至少有一个1
            j = np.random.randint(0, len(offspring))
            offspring_mutation[i][j] = 1
    print("end detect ...")


last_fitness = 0


def on_generation(ga_instance):
    # print("on_generation()")
    global last_fitness

    # print(f"Generation = {ga_instance.generations_completed}")
    # print(f"Fitness    = {ga_instance.best_solution()[1]}")
    # print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}")
    last_fitness = ga_instance.best_solution()[1]
    # print(f"Curr Solution = {ga_instance.best_solution()[0]}")


def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


# 分数 越大越好
def fitness(ga_instance, solution, solution_idx):
    v_new = l_new * w_new * h_new
    # return 1.0 / abs(choice[choice_idx] - v_new) + 0.0001
    fitness_value = 1.0 / (np.array(solution).dot(choices.values))
    # return abs(choice[choice_idx] - v_new)
    return fitness_value

function_inputs2 = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

def fitness_func(ga_instance, solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    output = np.sum(solution*function_inputs2)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

ga_instance = pygad.GA(num_generations=5000,
                       num_parents_mating=5,
                       fitness_func=fitness,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       # on_start=on_start,
                       # on_fitness=on_fitness,
                       # on_parents=on_parents,
                       # on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       # on_generation=on_generation,
                       # on_stop=on_stop,
                       gene_type=int,
                       gene_space=[0, 1]
                       )

# ga_instance.run()
# ga_instance.plot_fitness()
# print(ga_instance.best_solution())


ga_instance2 = pygad.GA(num_generations=100,
                       num_parents_mating=7,
                       fitness_func=fitness_func,
                       sol_per_pop=50,
                       num_genes=len(function_inputs2),
                       # on_start=on_start,
                       # on_fitness=on_fitness,
                       # on_parents=on_parents,
                       # on_crossover=on_crossover,
                       # on_generation=on_generation,
                       # on_stop=on_stop,
                       )
ga_instance2.run()
ga_instance2.plot_fitness()
print(ga_instance2.best_solution())
print(ga_instance2.best_solutions_fitness)

