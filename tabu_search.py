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
    '''
    Extend this class and implement the methods required by the particular
    solver in order to solve an instance of a problem expressed in terms
    that can be used by a Tabu Search algorithm.
    '''
    def __init__(self):
        pass

    def initial_solution(self):
        '''
        Return a (not necessarily) random solution to the problem at hand.
        '''
        raise NotImplementedError

    def feasible_solution(self, node):
        '''
        Returns True if the solution(node) is feasible in the context of the
        problem, you only need to implement this method if you plan on limiting
        your search to the feasible search space.
        '''
        raise NotImplementedError

    def cost(self, node, amplification_parameter=1):
        '''
        Returns the cost of solution(node) plus a infeasibility penalty times
        an amplification_parameter, while the infeasibility penalty part is
        optional, if you want to use infeasible solutions as part of your search
        it becomes necessary.
        '''
        raise NotImplementedError

    def neighbours(self, node):
        '''
        Given a solution(node) it returns its neighbourhood as a generator.
        '''
        return map(lambda m: self.apply(node, m), self.movements(node))

    def movements(self, node):
        '''
        Given a solution(node) returns all the possible movements from it as
        a generator.
        '''
        raise NotImplementedError

    def apply(self, node, movement):
        '''
        Apply the movement to the solution(node)/move from one solution to
        another in its neighbourhood.
        '''
        raise NotImplementedError

    def address(self, node):
        '''
        This function is a mapping from the solution(node) to a number
        representing a partition of the search space.

        This method is only necessary if the solver implements a partitioned
        search.
        '''
        raise NotImplementedError

    def increment_address(self, node):
        '''
        Returns a new random solution that is in a different random partition
        to the given one(node).

        This method is only necessary if the solver implements a partitioned
        search.
        '''
        raise NotImplementedError

class TabuSearchSolver(object):
    '''
    Extend this class and implement its methods in order to provide users with
    a solver based on the Tabu Search algorithm.

    Attributes:
    - amplification_parameter: multiplies the penalty in the cost of infeasible
        solutions. (default: 1)
    - tabu_shrink_period, tabu_shrink_factor: every period iterations the tabu
        list will shrink by factor(its maximum length is multiplied by it).
        (default: 10, 0.96)
    - infeasible_search: allow infeasible solutions to appear in our search.
        (default: True)
    '''
    def __init__(self, amplification_parameter=1, tabu_shrink_period=10,
                 tabu_shrink_factor=0.96, infeasible_search=True):
        self.amplification_parameter = amplification_parameter
        self.tabu_shrink_period = tabu_shrink_period
        self.tabu_shrink_factor = tabu_shrink_factor
        self.infeasible_search = infeasible_search
        self.solution = None

    def solve(self, problem, max_local_iters, max_global_iters=None,
              partitioned_search=False, figure_output=None):
        '''
        Solve the problem given with the algorithm implemented by this solver,
        arguments passed:
        problem: the problem given (extends TabuSearchProblem)
        max_local_iters: maximum number of iterations in a local search without
            improving the best local solution found so far
        max_global_iters: maximum number of global iterations without improving
            the best solution found so far, if None max_global_iters = max_local_iters.
            (default: None)
        partitioned_search: search using the information provided about the
            partitioning of the search space by problem. (default: False)
        figure_output: if given a path to a file it will output a plot of the
            convergence curve to it. (default: None)
        '''
        raise NotImplementedError

    @staticmethod
    def generate_figure(x, y, x_name, y_name, figure_output):
        '''
        Outputs a plot of the curve given by the points x, y to the path
        figure_output.
        '''
        plt.plot(x, y)
        plt.ylabel(y_name)
        plt.xlabel(x_name)
        plt.ylim(4, 24)
        plt.savefig(figure_output)
        plt.close()

class PartitionedSpaceSolver(TabuSearchSolver):
    '''
    Implements Tabu Search taking advantage of a partitioning of the search
    space.
    '''
    def solve(self, problem, max_local_iters, max_global_iters=None,
              partitioned_search=False, figure_output=None):
        if max_global_iters is None:
            max_global_iters = max_local_iters

        # keep track of how many moves we've made without improving best_solution
        true_iters = 0
        global_iters = 0
        local_iters = 0

        # initialize solutions
        best_solution = problem.initial_solution()
        best_solution_cost = problem.cost(best_solution, self.amplification_parameter)
        best_local_solution = best_solution
        best_local_solution_cost = problem.cost(best_local_solution, self.amplification_parameter)

        if figure_output is not None:
            figure_iters = []
            figure_costs = []

        # initialize search
        current_node = best_solution
        tabu_list = []
        max_tabu_len = max_local_iters

        while True:
            # shrink tabu list
            if (local_iters + 1) % self.tabu_shrink_period == 0:
                max_tabu_len = math.floor(max_tabu_len*self.tabu_shrink_factor)

            possible_moves = list(
                filter(
                    lambda n: n not in tabu_list and
                    (self.infeasible_search or problem.feasible_solution(n)),
                    problem.neighbours(current_node)))

            if not possible_moves or local_iters > max_local_iters:
                if global_iters > max_global_iters:
                    break
                
                # reset local search
                tabu_list = []
                local_iters = 0
                best_local_solution = problem.increment_address(current_node)
                best_local_solution_cost = problem.cost(best_local_solution,
                                                        self.amplification_parameter)
                current_node = best_local_solution
                max_tabu_len = max_local_iters
                continue

            best_move = min(possible_moves,
                            key=lambda m: problem.cost(m, self.amplification_parameter))

            if (problem.cost(best_move, self.amplification_parameter) <
                    best_solution_cost):

                best_solution = best_move
                best_solution_cost = problem.cost(best_move,
                                                  self.amplification_parameter)
                global_iters = 0
                figure_iters.append(true_iters)
                figure_costs.append(best_solution_cost)

            if problem.cost(best_move, self.amplification_parameter) < best_local_solution_cost:

                best_local_solution = best_move
                best_local_solution_cost = problem.cost(best_move,
                                                        self.amplification_parameter)
                local_iters = 0

            current_node = best_move
            tabu_list.append(best_move)

            local_iters += 1
            global_iters += 1
            true_iters += 1
            
            if len(tabu_list) > max_tabu_len:
                del tabu_list[0:len(tabu_list) - max_tabu_len]

        figure_iters.append(true_iters)
        figure_costs.append(best_solution_cost)
        self.generate_figure(figure_iters, figure_costs, 'Cp', 'iteration', figure_output)

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
            return self.components_characteristics[component_i, version*3+2]*m
        elif m <= self.discounts[component_i][1][0]:
            return (self.components_characteristics[component_i, version*3+2] *
                    self.discounts[component_i][0][1] * m)
        else:
            return (self.components_characteristics[component_i, version*3+2] *
                    self.discounts[component_i][1][1] * m)

    def cost(self, node, amplification_parameter=1):
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

def rap_from_json(path):
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
    problem = rap_from_json(problem_json)
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
                figure_output='convergence-lev4-(4-6)-3-0960.png')
    test_solver(DEFAULT_SOLVER, 'lev4-(4-6)-3.json', min_availability=0.99,
                max_local_iters=200, max_global_iters=500,
                figure_output='convergence-lev4-(4-6)-3-0990.png')
