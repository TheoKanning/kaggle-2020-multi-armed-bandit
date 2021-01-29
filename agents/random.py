
import random

class RandomAgent:
    
    def step(self, observation, configuration):
        return random.randrange(configuration.banditCount)
    
    def description(self):
        return "Random Agent"
