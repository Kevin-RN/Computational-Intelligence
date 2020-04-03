from main.Maze import Maze
from main.Agent import Agent
from main.mysolution.MyQLearning import MyQLearning
from main.mysolution.MyEGreedy import MyEGreedy

if __name__ == "__main__":
    # load the maze
    file = "..\\..\\data\\toy_maze.txt"
    maze = Maze(file)

    # Set the reward at the bottom right to 10
    maze.set_reward(maze.get_state(len(maze.states[0])-1, len(maze.states)-1), 10)
    # Set the second reward at the top right to 5
    maze.set_reward(maze.get_state(len(maze.states[0])-1, 0), 5)

    # create a robot at starting and reset location (0,0) (top left)
    robot = Agent(0, 0)

    # make a selection object (you need to implement the methods in this class)
    selection = MyEGreedy()

    # make a QLearning object (you need to implement the methods in this class)
    learn = MyQLearning()

    # Initialise some constants for updating the QLearning object
    alpha = 0.7
    gamma = 0.9
    epsilon = 1

    # keep learning until you decide to stop
    for i in range(30000):

        # Get the next action and update the QLearning
        state = robot.get_state(maze)
        action = selection.get_egreedy_action(robot, maze, learn, epsilon)
        state_next = robot.do_action(action, maze)
        r = maze.get_reward(state_next)
        possible_actions = maze.get_valid_actions(robot)
        learn.update_q(state, action,  r, state_next, possible_actions, alpha, gamma)

        # Reset the robot once it reaches the end state so it can run though the maze again
        if state.__eq__(maze.get_state(len(maze.states[0])-1, len(maze.states)-1)) or \
                state.__eq__(maze.get_state(len(maze.states[0])-1, 0)):
            epsilon *= 0.995
            robot.reset()
