import numpy as np

import random


#  https://www.kaggle.com/a763337092/pull-vegas-slot-machines-add-weaken-rate-continue5
class PullVegasAgent:

    def __init__(self, num_bandits):
        self.total_reward = 0
        self.actions = []
        self.opp_actions = []

        self.wins = np.ones(num_bandits)
        self.losses = np.zeros(num_bandits)
        self.opp = np.zeros(num_bandits)
        self.my_continue = np.zeros(num_bandits)
        self.opp_continue = np.zeros(num_bandits)

    def description(self):
        return "Pull Vegas"

    def step(self, observation, configuration):
        return self.multi_armed_probabilities(observation, configuration)

    def get_next_bandit(self):
        total_pulls = self.wins + self.losses + self.opp
        probs = (self.wins - self.losses + self.opp - (self.opp > 0) * 1.5 + self.opp_continue) / (total_pulls) \
                * np.power(0.97, total_pulls)
        best_bandit = int(np.argmax(probs))
        return best_bandit

    def multi_armed_probabilities(self, observation, configuration):

        if observation.step == 0:
            return random.randrange(configuration.banditCount)

        last_reward = observation.reward - self.total_reward
        self.total_reward = observation.reward

        my_idx = observation.agentIndex
        my_last_action = observation.lastActions[my_idx]
        opp_last_action = observation.lastActions[1 - my_idx]

        self.actions.append(my_last_action)
        self.opp_actions.append(opp_last_action)

        if last_reward:
            self.wins[my_last_action] += 1
        else:
            self.losses[my_last_action] += 1

        self.opp[opp_last_action] += 1

        if observation.step >= 3:
            if self.actions[-1] == self.actions[-2]:
                self.my_continue[my_last_action] += 1
            else:
                self.my_continue[my_last_action] = 0
            if self.opp_actions[-1] == self.opp_actions[-2]:
                self.opp_continue[opp_last_action] += 1
            else:
                self.opp_continue[opp_last_action] = 0

        if last_reward:
            return my_last_action

        if observation.step >= 4:
            if (self.actions[-1] == self.actions[-2]) and (self.actions[-1] == self.actions[-3]):
                if random.random() < 0.5:
                    return self.actions[-1]

        return self.get_next_bandit()
