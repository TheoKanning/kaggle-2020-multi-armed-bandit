import numpy as np
import pandas as pd
from tensorflow import keras


class KerasAgent:

    def __init__(self, num_bandits, filename, margin=0.99):
        self.total_reward = 0
        self.filename = filename
        self.margin = margin

        self.model = keras.models.load_model(self.filename)
        self.machine_states = pd.DataFrame(
            index=range(num_bandits),
            columns=['step', 'n_pulls', 'n_success', 'n_opp_pulls', 'streak', 'win_streak', 'opp_streak']
        ).fillna(0)

    def description(self):
        return f"Keras - {self.filename}, margin:{self.margin:.2f}"

    def step(self, observation, configuration):
        if observation.step == 0:
            return np.random.randint(configuration.banditCount)

        reward = observation.reward - self.total_reward
        self.total_reward = observation.reward
        last_action = observation.lastActions[observation.agentIndex]
        opp_action = observation.lastActions[1 - observation.agentIndex]

        self.machine_states['step'] = observation.step
        self.machine_states.at[last_action, 'n_pulls'] += 1
        self.machine_states.at[last_action, 'n_success'] += reward
        self.machine_states.at[opp_action, 'n_opp_pulls'] += 1

        self.machine_states.at[last_action, 'streak'] += 1
        self.machine_states.loc[self.machine_states.index != last_action, 'streak'] = 0
        self.machine_states.at[opp_action, 'opp_streak'] += 1
        self.machine_states.loc[self.machine_states.index != opp_action, 'opp_streak'] = 0

        if reward:
            self.machine_states.at[last_action, 'win_streak'] += 1
        else:
            self.machine_states.at[last_action, 'win_streak'] = 0

        probs = self.model(self.machine_states.to_numpy())

        max_return = np.max(probs)
        result = np.random.choice(np.where(probs >= self.margin * max_return)[0])
        return int(result)


agent = None


def keras_agent(observation, configuration):
    global agent
    if observation.step == 0:
        print("Creating keras agent")
        agent = KerasAgent(configuration.banditCount, "/kaggle_simulations/agent/model.h5")

    return agent.step(observation, configuration)
