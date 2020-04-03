from main.QLearning import QLearning


class MyQLearning(QLearning):

    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        q_old = self.get_q(state, action)
        q_max = 0
        for pa in possible_actions:
            q_temp = self.get_q(state_next, pa)
            q_max = max(q_max, q_temp)
        value = q_old + alfa * (r + gamma * q_max - q_old)
        self.set_q(state, action, value)
        return
