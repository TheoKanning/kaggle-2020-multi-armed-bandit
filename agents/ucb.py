import numpy as np
import scipy.stats


class UcbAgent:

    def __init__(self, c=3, opp_bonus=0.0, random=False):
        self.wins = None
        self.losses = None
        self.num_bandits = None
        self.opp_actions = []
        self.total_reward = 0

        # parameters
        self.c = c  # number of standard deviations in confidence interval
        self.opp_bonus = opp_bonus
        self.random = random

    def description(self):
        return f"UCB Agent c={self.c} opp_bonus={self.opp_bonus} random={self.random}"

    def step(self, observation, configuration) -> int:
        if observation.step == 0:
            self.num_bandits = configuration.banditCount
            self.wins = np.ones(self.num_bandits)
            if self.random:
                self.wins += np.random.rand(self.num_bandits) / 100
            self.losses = np.ones(self.num_bandits)
        else:
            player = observation.agentIndex
            opponent = 1 - player

            reward = observation.reward - self.total_reward
            self.total_reward = observation.reward

            # adjust win or loss counts
            if reward:
                self.wins[observation.lastActions[player]] += 1
            else:
                self.losses[observation.lastActions[player]] += 1

            self.opp_bonus_adjustment(observation.lastActions[opponent])

        ucbs = self.wins / (self.wins + self.losses) + self.c * scipy.stats.beta.std(self.wins, self.losses)
        return int(np.argmax(ucbs))

    def opp_bonus_adjustment(self, opp_action):
        # give a bonus to a bandit if our opponent tried it twice in a row

        if self.opp_actions and self.opp_actions[-1] == opp_action:
            # could add decay here but it probably won't matter because opp_bonus is arbitrary anyway
            self.wins[opp_action] += self.opp_bonus

        self.opp_actions.append(opp_action)


agent = UcbAgent(
    opp_bonus=0.2,
    c=3
)


def ucb_agent(observation, configuration):
    return agent.step(observation, configuration)
