import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.Route import Route
from src.Direction import Direction
import numpy as np


# Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = np.random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        route = Route(self.start)
        self.maze.start = self.start
        self.maze.end = self.end

        # create 2d array to check which coordinates the ant has visited to prevent loops and other tricky features
        visited_maze = np.zeros((self.maze.get_width(), self.maze.get_length()), dtype=bool)
        visited_maze[self.current_position.get_x()][self.current_position.get_y()] = True

        # loop until ant has reached the end coordinate
        while self.current_position != self.end:
            surrounding_pheromone = self.maze.get_surrounding_pheromone(self.current_position)
            possible_directions = []
            pheromones_directions = []

            # loop through all directions (N/E/S/W)
            for direction in Direction:
                pos = self.current_position.add_direction(direction)
                pheromone_direction = surrounding_pheromone.get(direction)

                #  Check if the next possible position is valid if not go to next
                if pheromone_direction > 0 and not visited_maze[pos.get_x()][pos.get_y()]:
                    possible_directions.append(direction)
                    pheromones_directions.append(pheromone_direction)

            # If there are no directions to go (dead end), go 1 step back
            if len(possible_directions) < 1:
                # to avoid removing from empty list
                if route.size() > 0:
                    last = route.remove_last()
                    self.current_position = self.current_position.subtract_direction(last)
                continue

            # Calculate all probabilities for all possible directions
            total_surrounding = sum(pheromones_directions)
            probability_directions = []
            for pheromone in pheromones_directions:
                probability_directions.append(pheromone/total_surrounding)

            # add direction to the route of the ant
            direction = self.rand.choice(possible_directions, p=probability_directions)
            route.add(direction)
            # go to next position and update visited array
            self.current_position = self.current_position.add_direction(direction)
            visited_maze[self.current_position.get_x()][self.current_position.get_y()] = True

        return route
