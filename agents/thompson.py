import numpy as np


class ThompsonAgent:

    def __init__(self):
        self.bandit_states = None
        self.last_action = None
        self.total_reward = 0

    def description(self):
        return "Thompson Agent"

    def step(self, observation, configuration):
        if observation.step == 0:
            self.bandit_states = np.ones((configuration.banditCount, 2))
        else:

            reward = observation.reward - self.total_reward
            self.total_reward = observation.reward

            if reward:
                self.bandit_states[self.last_action][0] += 1
            else:
                self.bandit_states[self.last_action][1] += 1

            for bandit in observation.lastActions:
                self.bandit_states[bandit][0] *= 0.97

        probs = np.random.beta(self.bandit_states[:, 0], self.bandit_states[:, 1])
        best_bandit = int(np.argmax(probs))

        self.last_action = best_bandit

        return best_bandit


agent = ThompsonAgent()


def thompson_agent(observation, configuration):
    return agent.step(observation, configuration)
