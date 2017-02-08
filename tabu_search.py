# -*- coding: utf-8 -*-
import random
import itertools
from copy import deepcopy
import time
import math
import json
import numpy as np
from scipy.special import bdtr
import matplotlib.pyplot as plt

class TabuSearchProblem(object):
    def __init__(self):
        pass

    def initial_solution(self):
        raise NotImplementedError

    def feasible_solution(self, node):
        raise NotImplementedError

    def cost(self, node, amplification_parameter):
        raise NotImplementedError

    def neighbours(self, node):
        return map(lambda m: self.apply(node, m), self.movements(node))

    def movements(self, node):
        raise NotImplementedError

    def apply(self, node, movement):
        raise NotImplementedError

    def address(self, node):
        raise NotImplementedError

    def increment_address(self, node):
        raise NotImplementedError

class TabuSearchSolver(object):
    def __init__(self, amplification_parameter=1, tabu_shrink_period=10,
                 tabu_shrink_factor=0.96, infeasible_search=True):
        self.amplification_parameter = amplification_parameter
        self.tabu_shrink_period = tabu_shrink_period
        self.tabu_shrink_factor = tabu_shrink_factor
        self.infeasible_search = infeasible_search
        self.solution = None

    def solve(self, problem, max_local_iters, max_global_iters=None,
              partitioned_search=False, figure_output=False):
        raise NotImplementedError

class PartitionedSpaceSolver(TabuSearchSolver):
    def solve(self, problem, max_local_iters, max_global_iters=None,
              partitioned_search=False, figure_output=None):
        if max_global_iters is None:
            max_global_iters = max_local_iters

        # keep track of how many moves we've made without improving best_solution
        true_iters = 0
        global_iters = 0
        local_iters = 0

        best_solution = problem.initial_solution()
        best_local_solution = best_solution
        q = self.amplification_parameter

        figure_iters = []
        figure_costs = []

        current_node = best_solution
        tabu_list = []
        max_tabu_len = max_local_iters

#        same_kind_iters = 0
#        iters_feasible = self.feasible_solution(current_node)

        while True:
            if (local_iters + 1) % self.tabu_shrink_period == 0:
                max_tabu_len = math.floor(max_tabu_len*self.tabu_shrink_factor)

#            if self.feasible_solution(current_node) == iters_feasible:
#                same_kind_iters += 1
#                if same_kind_iters >= amplification_period:
#                    if iters_feasible:
#                        q = max(q/2, self.cost(best_solution, 0))
#                    else:
#                        q = min(q*2, math.factorial(100))
#            else:
#                same_kind_iters = 0
#                iters_feasible = not iters_feasible

            possible_moves = list(
                filter(
                    lambda n: n not in tabu_list and
                    (self.infeasible_search or problem.feasible_solution(n)),
                    problem.neighbours(current_node)))
            if not possible_moves:
                if global_iters > max_global_iters:
                    break
                tabu_list = []
                local_iters = 0
                best_local_solution = problem.increment_address(current_node)
                current_node = best_local_solution
                #q = amplification_parameter
#                same_kind_iters = 0
#                iters_feasible = self.feasible_solution(current_node)
                max_tabu_len = max_local_iters
                continue

            best_move = min(possible_moves, key=lambda m: problem.cost(m, q))

            if problem.cost(best_move, q) < problem.cost(best_solution, q):
                best_solution = best_move
                global_iters = 0
                figure_iters.append(true_iters)
                figure_costs.append(problem.cost(best_solution, q))

            if problem.cost(best_move, q) < problem.cost(best_local_solution, q):
                best_local_solution = best_move
                local_iters = 0

            current_node = best_move
            tabu_list.append(best_move)
            local_iters += 1
            global_iters += 1
            true_iters += 1
            if len(tabu_list) > max_tabu_len:
                del tabu_list[0:len(tabu_list) - max_tabu_len]
            if local_iters > max_local_iters:
                if global_iters > max_global_iters:
                    break
                else:
                    tabu_list = []
                    local_iters = 0
                    best_local_solution = problem.increment_address(current_node)
                    current_node = best_local_solution
                    #q = amplification_parameter
#                    same_kind_iters = 0
#                    iters_feasible = self.feasible_solution(current_node)
                    max_tabu_len = max_local_iters
                    continue
        figure_iters.append(true_iters)
        figure_costs.append(problem.cost(best_solution, q))
        if figure_output is not None:
            plt.plot(figure_iters, figure_costs)
            plt.ylabel("Cp")
            plt.xlabel("Numero de iteraciones")
            plt.ylim(4, 24)
            plt.savefig(figure_output)
            plt.close()
        self.solution = best_solution
        return self.solution

class RedundancyAllocationProblem(TabuSearchProblem):
    def __init__(self, n_components, max_components, max_versions,
                 components_characteristics, min_availability,
                 min_performances, T, discounts=None):
        '''
        n_components: number of parallel subsystems connected in series
        max_components: maximum number of components in each parallel subsystem (list)
        component_characteristics: tuple of 3 lists of lists:
        - reliability per unit for each combination of component and version
        - cost per unit for each combination of component and version
        - performance per unit for each combination of component and version
        min_availability: minimum availability required of the system
        min_performances: minimum performance of the whole system for each time
            interval (list)
        T: length of each time interval (list)
        discounts(optional): list of lists where each element of the inner
            lists is a pair of pairs, the first pair is the first units bought
            threshold for a given component along with the discount applied
            and the second pair is analogous to the first one but for a higher
            units bought threshold
        infeasible_search(optional): if True the search will admit as a valid
            movement a infeasible solution
        '''
        super(RedundancyAllocationProblem, self).__init__()
        self.solution = None
        self.n_components = n_components
        self.max_components = max_components
        self.max_versions = max_versions
        self.components_characteristics = components_characteristics
        self.min_availability = min_availability
        self.K = len(T)
        self.T = T
        self.min_performances = min_performances
        self.discounts = discounts
        self.N = sum(self.max_components) + sum(self.max_versions) - self.n_components

    def initial_solution(self):
        res = [[], []]
        for _ in range(self.n_components):
            res[0].append(1)
            res[1].append(0)
        return tuple(res)

    def random_solution(self):
        res = [[], []]
        for i in range(self.n_components):
            res[0].append(random.randint(1, self.max_components[i]))
            res[1].append(random.randint(0, self.max_versions[i] - 1))
        return tuple(res)

    def success_probability(self, node, min_performance):
        parallel_probabilities = [] # per-subsystem success

        if min_performance <= 0:
            return 1

        for i in range(self.n_components):
            # k out of n components needed
            characteristics = self.components_characteristics[i, node[1][i]*3:(node[1][i]+1)*3]
            k = math.ceil(min_performance / characteristics[1])

            if k == 1:
                parallel_probabilities.append(1 - (1-characteristics[0])**node[0][i])
            elif k == node[0][i]:
                parallel_probabilities.append(characteristics[0]**node[0][i])
            elif k > node[0][i]:
                parallel_probabilities.append(0)
            else:
                parallel_probabilities.append(1 - bdtr(k - 1, node[0][i], characteristics[0]))

        return np.product(parallel_probabilities)

    def availability(self, node):
        res = 0.0

        for p, T_k in zip(self.min_performances, self.T):
            res += T_k * self.success_probability(node, p)

        return res / sum(self.T)

    def feasible_solution(self, node):
        return self.availability(node) >= self.min_availability

    def discounted_cost_of_component(self, component_i, version, m):
        if m <= self.discounts[component_i][0][0]:
            return self.components_characteristics[component_i, version*3+2] * m
        elif m <= self.discounts[component_i][1][0]:
            return self.components_characteristics[component_i, version*3+2] * self.discounts[component_i][0][1] * m
        else:
            return self.components_characteristics[component_i, version*3+2] * self.discounts[component_i][1][1] * m

    def cost(self, node, amplification_parameter):
        res = 0
        for i in range(self.n_components):
            if self.discounts is not None:
                res += self.discounted_cost_of_component(i, node[1][i], node[0][i])
            else:
                res += self.components_characteristics[i, node[1][i]*3+2] * node[0][i]

        return res + amplification_parameter*self.infeasibility_penalty(node)

    def infeasibility_penalty(self, node):
        return max(0, self.min_availability - self.availability(node))

    def half_movements_negative(self, node):
        res = []
        for i in range(self.n_components):
            if node[0][i] > 1:
                res.append((i, 0, -1))
            if node[1][i] > 0:
                res.append((i, 1, -1))
        return res

    def half_movements_positive(self, node):
        res = []
        for i in range(self.n_components):
            if node[0][i] < self.max_components[i]:
                res.append((i, 0, 1))
            if node[1][i] < self.max_versions[i] - 1:
                res.append((i, 1, 1))
        return res

    def movements(self, node):
        return itertools.product(self.half_movements_positive(node),
                                 self.half_movements_negative(node))

    def apply(self, node, movement):
        res = list(deepcopy(node))
        for m in movement:
            # apply each partial movement
            res[m[1]][m[0]] += m[2]
        return res

    def address(self, node):
        return sum(node[0]) + sum(node[1])

    def increment_address(self, node):
        # increment address randomly
        res = self.random_solution()
        while self.address(res) == self.address(node):
            res = self.random_solution()
        return res
        # discarded code
#        final_address = random.randint(self.n_components, self.N)
#        while final_address == self.address(node):
#            final_address = random.randint(self.n_components, self.N)
#        increment = final_address - self.address(node)
#        if increment > 0:
#            movement_pool = self.half_movements_positive
#        elif increment < 0:
#            movement_pool = self.half_movements_negative
#        res = deepcopy(node)
#        while self.address(res) != final_address:
#            if movement_pool(res):
#                res = self.half_apply(res, random.choice(movement_pool(res)))
#            else:
#                final_address = random.randint(self.n_components, self.N)
#                while final_address == self.address(node):
#                    final_address = random.randint(self.n_components, self.N)
#                increment = final_address - self.address(node)
#
#                if increment > 0:
#                    movement_pool = self.half_movements_positive
#                elif increment < 0:
#                    movement_pool = self.half_movements_negative
#                res = deepcopy(node)
#        return res

def RAP_from_json(path):
    with open(path) as file:
        parsed = json.load(file)
        max_n_versions = max(map(lambda c: c['n_versions'], parsed['components_characteristics']))
        components_characteristics = np.zeros((parsed['n_components'],
                                               max_n_versions*3),
                                              dtype=np.float64)

        for i, comp in enumerate(parsed['components_characteristics']):
            for j, version in enumerate(zip(comp['component_reliabilities'],
                                            comp['component_performances'],
                                            comp['component_costs'])):

                components_characteristics[i, j*3:(j+1)*3] = version

        return RedundancyAllocationProblem(parsed['n_components'],
                                           parsed['max_components'],
                                           list(map(lambda c: c['n_versions'],
                                                    parsed['components_characteristics'])),
                                           components_characteristics, parsed['min_availability'],
                                           parsed['min_performances'], parsed['T'])

def test_solver(solver, problem_json, min_availability, max_local_iters=200,
                max_global_iters=500, figure_output=None):
    problem = RAP_from_json(problem_json)
    problem.min_availability = min_availability

    running_time = time.clock()
    sol = solver.solve(problem, max_local_iters=max_local_iters,
                       max_global_iters=max_global_iters,
                       partitioned_search=True,
                       figure_output=figure_output)

    running_time = time.clock() - running_time
    print("Min availability:", problem.min_availability)
    print(running_time, "seconds")
    print("Solution:", (sol[0], list(map(lambda x: x+1, sol[1]))))
    print("Availability:", problem.availability(sol))
    print("Cost:", problem.cost(sol, 0))
    print("Feasible:", problem.feasible_solution(sol))

if __name__ == '__main__':

    DEFAULT_SOLVER = PartitionedSpaceSolver(amplification_parameter=100,
                                            tabu_shrink_period=10,
                                            tabu_shrink_factor=0.96,
                                            infeasible_search=True)

    test_solver(DEFAULT_SOLVER, 'lev4-(4-6)-3.json', min_availability=0.9,
                max_local_iters=200, max_global_iters=500,
                figure_output='convergence-lev4-(4-6)-3-0900.png')
    test_solver(DEFAULT_SOLVER, 'lev4-(4-6)-3.json', min_availability=0.96,
                max_local_iters=200, max_global_iters=500,
                figure_output='convergence-lev4-(4-6)-3-0900.png')
    test_solver(DEFAULT_SOLVER, 'lev4-(4-6)-3.json', min_availability=0.99,
                max_local_iters=200, max_global_iters=500,
                figure_output='convergence-lev4-(4-6)-3-0900.png')
