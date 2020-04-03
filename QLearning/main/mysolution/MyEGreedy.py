import random
import numpy as np


class MyEGreedy:

    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        # Choose random action from all valid actions
        valid_actions = maze.get_valid_actions(agent)
        return random.choice(valid_actions)


    def get_best_action(self, agent, maze, q_learning):
        # Using all valid actions to get all action values
        valid_actions = maze.get_valid_actions(agent)
        values = q_learning.get_action_values(agent.get_state(maze), valid_actions)
        # Get the max action value index
        max_val = max(values)
        choices = []
        for idx, maxvalue in enumerate(values):
            if max_val == maxvalue:
                choices.append(idx)
        # In case of multiple max values choose random one and return the action belonging to that index
        max_index = random.choice(choices)
        return valid_actions[max_index]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        # using probability (1/epsilon) to choose either random or best action
        if random.random() <= epsilon:
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)
