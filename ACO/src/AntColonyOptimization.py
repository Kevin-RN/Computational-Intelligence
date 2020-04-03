import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import time
from src.Maze import Maze
from src.Ant import Ant
from src.PathSpecification import PathSpecification


# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

    # Loop that starts the shortest path process
    # @param spec Specification of the route we wish to optimize
    # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()
        short_route = None
        for generation in range(self.generations):
            gen_pheromones = []
            for ant in range(self.ants_per_gen):
                route_i = Ant(self.maze, path_specification).find_route()
                if short_route is None or route_i.shorter_than(short_route):
                    short_route = route_i
                gen_pheromones.append(route_i)
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(gen_pheromones, self.q)
        return short_route


# Driver function for Assignment 1
if __name__ == "__main__":
    # parameters
    gen = 40
    no_gen = 15
    q = 1500
    evap = 0.3

    # construct the optimization objects
    maze = Maze.create_maze("./../data/hard maze.txt")
    spec = PathSpecification.read_coordinates("./../data/hard coordinates.txt")
    aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

    # save starting time
    start_time = int(round(time.time() * 1000))

    # run optimization
    shortest_route = aco.find_shortest_route(spec)
    # print time taken
    print("Time taken: " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))

    # save solution
    shortest_route.write_to_file("./../data/hard_solution.txt")

    # print route size
    print("Route size: " + str(shortest_route.size()))
