import gym
import gym_everglades
from keras.models import Model, load_model
import numpy as np
import random
from agents.dqn_agent import DQNAgent


class DQNTest(DQNAgent):
    def __init__(self, action_space, player_num, map_name):
        super().__init__(action_space, player_num, map_name)
        self.model_name = "everglades-dqn-101-0.6633663366336634.h5"
        self.model = load_model(self.model_name)

    def get_action(self, obs):
        obs = np.reshape(obs, [1, len(obs)])
        action = super().get_action(obs)
        # print(action)
        return action[1]
