import numpy as np
import random
import math


class Jammer:
    def __init__(self):
        super(Jammer, self).__init__()
        self.sinr_set = [20, 30, 40, 50, 60, 70, 80, 90]
        self.power_set = [45, 50, 55]
        self.jammer_cost_weight = 0.2
        self.Q_table = np.zeros((len(self.sinr_set) + 1, len(self.power_set)))
        self.learning_rate = 0.2
        self.discount_rate = 0.3
        self.jam_gain_set = [0.12, 0.14]
        self.last_jam_gain = 220
        self.Markov_jam = [[0.9, 0.1],
                           [0.1, 0.9]]

    def obtain_state_index(self, state):
        state_index = 0
        for value in self.sinr_set:
            if state > value:
                state_index += 1
            else:
                break
        return state_index

    def jamming_power(self, state, epsilon, car_to_hacking_gain, noise, jam_gain):
        state_index = self.obtain_state_index(state)
        #print(state_index)
        if random.random() > epsilon-0.05:
            action_index = self.Q_table[state_index].argmax()
        else:
            action_index = random.randint(0, len(self.power_set) - 1)

        jamming_power = (((car_to_hacking_gain * self.power_set[action_index] * 10) / self.sinr_set[
            state_index]) - noise) / self.last_jam_gain
        self.last_jam_gain = jam_gain

        return action_index, jamming_power#

    def calculate_jammer_utility(self, vehicle_utility, jammer_power):
        jammer_utility = -1 * vehicle_utility - self.jammer_cost_weight * jammer_power
        return jammer_utility

    def update_Qtable(self, state, action, utility, next_state):
        state_index = self.obtain_state_index(state)
        next_state_index = self.obtain_state_index(next_state)
        action_index = action
        Q = self.Q_table[state_index][action_index]
        Q_ = self.Q_table[next_state_index].max()
        self.Q_table[state_index][action_index] = (1 - self.learning_rate) * Q + \
                                                   self.learning_rate * (utility + self.discount_rate * Q)

if __name__=="__main__":
    jam = Jammer()
    e = 0.02 + 0.9987 * 0.98
    a, b = jam.jamming_power(7, e)