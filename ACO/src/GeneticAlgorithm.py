import os, sys
import random
import numpy as np
from src.TSPData import TSPData

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size

    # Knuth-Yates shuffle, reordering a array randomly
    # @param chromosome array to shuffle.
    def shuffle(self, chromosome):
        n = len(chromosome)
        for i in range(n):
            r = i + int(random.uniform(0, 1) * (n - i))
            swap = chromosome[r]
            chromosome[r] = chromosome[i]
            chromosome[i] = swap
        return chromosome

    def get_cost(self, product_order, s_to_p, p_to_p, e_to_p):
        length = 0
        length += s_to_p[product_order[0]]
        for i in range(len(product_order)-1):
            length += p_to_p[i][i+1]
        length += e_to_p[product_order[len(product_order) - 1]]
        return length

    def get_fitness_ratio(self, decoded_inputs):
        fitness = []
        fitness_ratio = []
        for i in decoded_inputs:
            fitness.append(100000/i)
            # fitness_ratio.append((fitness/sum(decoded_inputs))*100)
        for i in fitness:
            fitness_ratio.append((i/sum(fitness))*100)
        return fitness_ratio

    def sumRange(self, list, a, b):
        sum = 0
        for i in range(a, b+1, 1):
            sum += list[i]
        return sum

    def get_cumulative_fitness_ratio(self, fitness_ratio):
        cft = []
        for i in range(len(fitness_ratio)):
            cft.append(self.sumRange(fitness_ratio, 0, i))
        return cft

    def pick_two_chromosomes(self, cfr):
        chosen = []
        for i in range(2):
            r = random.random()
            for j in range(len(cfr)-1):
                if r > cfr[j]/100 and r <= cfr[j+1]/100:
                    chosen.append(j+1)
                    break
        if len(chosen) == 0:
            chosen.append(0)
        if len(chosen) == 1:
            chosen.append(0)
        return chosen

    def cross_over(self, c1, c2):
        r1 = random.randrange(18)
        r2 = random.randrange(r1, 18)
        middle_blob = c2[slice(r1, r2)]
        return middle_blob + [item for item in c1 if item not in middle_blob]

    def swap(self, chromosomes, pos_1, pos_2):
        chromosomes[pos_1], chromosomes[pos_2] = chromosomes[pos_2], chromosomes[pos_1]
        return chromosomes

    # This method should solve the TSP.
    # What is the order of products with the shortest possible route
    # that visits each product and go to the end?
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        cross_over_probability = 0.7
        mutation_probability = 0.005

        list_of_products = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        parent_chromosomes = []
        parent_costs = []
        p_to_p = tsp_data.get_distances()
        s_to_p = tsp_data.get_start_distances()
        e_to_p = tsp_data.get_end_distances()


        # TODO loop here
        for i in range(self.pop_size):
            parent_chromosomes.append(self.shuffle(list_of_products)[:])

        for i in range(self.pop_size):
            parent_costs.append(self.get_cost(parent_chromosomes[i], s_to_p, p_to_p, e_to_p))
            cumulative_fitness_ratio = self.get_cumulative_fitness_ratio(self.get_fitness_ratio(parent_costs))

        # print(chromosome)
        # print(lengths)
        # print(cumulative_fitness_ratio)

        for k in range(self.generations):

            child_chromosomes = []

            # does crossover
            for i in range(int(self.pop_size/2)):
                chosen_two = self.pick_two_chromosomes(cumulative_fitness_ratio)
                if random.random() < cross_over_probability:
                    cross_over_chromosome_1 = self.cross_over(parent_chromosomes[chosen_two[0]], parent_chromosomes[chosen_two[1]])
                    cross_over_chromosome_2 = self.cross_over(parent_chromosomes[chosen_two[1]], parent_chromosomes[chosen_two[0]])
                    child_chromosomes.append(cross_over_chromosome_1)
                    child_chromosomes.append(cross_over_chromosome_2)
                else:
                    child_chromosomes.append(parent_chromosomes[chosen_two[0]])
                    child_chromosomes.append(parent_chromosomes[chosen_two[1]])

            # does mutation
            for i in range(self.pop_size):
                if random.random() < mutation_probability:
                    r1 = random.randrange(18)
                    r2 = random.randrange(18)
                    self.swap(child_chromosomes[i], r1, r2)

            # get good parents
            children_costs = []
            for i in range(self.pop_size):
                children_costs.append(self.get_cost(child_chromosomes[i], s_to_p, p_to_p, e_to_p))
            max_kid = max(children_costs)
            for i in range(5):
                removed_index = -1
                min_index = children_costs.index(min(children_costs))
                for j in range(self.pop_size):
                    if(parent_costs[j] < max_kid):
                        removed_index = j
                        del child_chromosomes[min_index]
                        break
                if removed_index != -1:
                    child_chromosomes.append(parent_chromosomes[removed_index])

            parent_chromosomes = child_chromosomes

            # print(child_chromosomes)
        #
        final_list = []
        #
        # for i in child_chromosomes:
        #     final_list.append(self.get_fitness_ratio(i))

        for i in range(self.pop_size):
            final_list.append(self.get_cost(parent_chromosomes[i], s_to_p, p_to_p, e_to_p))

        final_list_index = final_list.index(min(final_list))

        return parent_chromosomes[final_list_index]



# Assignment 2.b
if __name__ == "__main__":
    # parameters
    population_size = 20
    generations = 20
    persistFile = "./../tmp/productMatrixDist"

    # setup optimization
    tsp_data = TSPData.read_from_file(persistFile)
    ga = GeneticAlgorithm(generations, population_size)

    # run optimzation and write to file
    solution = ga.solve_tsp(tsp_data)
    tsp_data.write_action_file(solution, "./../data/TSP solution 2.txt")
    print("Wrote to file TSP solution")
