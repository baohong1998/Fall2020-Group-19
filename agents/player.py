import os
import numpy as np
import time
import json
import gym
import gym_everglades


class Player:
    def __init__(self, action_space, player_num, map_name):
        self.action_space = action_space
        self.num_groups = 12
        self.player_num = player_num
        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)

        self.nodes_array = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.nodes_array.append(in_node['ID'])

        self.num_nodes = len(self.map_dat['nodes'])
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)
        self.action_choices = self.get_action_choices(
            (self.num_groups * len(self.nodes_array), 2))

        self.unit_config = {
            0: [('controller', 1), ('striker', 5)],  # 6
            1: [('controller', 3), ('striker', 3), ('tank', 3)],  # 15
            2: [('tank', 5)],  # 20
            3: [('controller', 2), ('tank', 4)],  # 26
            4: [('striker', 10)],  # 36
            5: [('controller', 4), ('striker', 2)],  # 42
            6: [('striker', 4)],  # 46
            7: [('controller', 1), ('striker', 2), ('tank', 3)],  # 52
            8: [('controller', 3)],  # 55
            9: [('controller', 2), ('striker', 4)],  # 61
            10: [('striker', 9)],  # 70
            11: [('controller', 20), ('striker', 8), ('tank', 2)]  # 100
        }

    def get_action_choices(self, shape):
        action_choices = np.zeros(shape)
        group_id = 0
        node_id = 1
        for i in range(0, action_choices.shape[0]):
            if i > 0 and i % 11 == 0:
                group_id += 1
                node_id = 1
            action_choices[i] = [group_id, node_id]
            node_id += 1
        return action_choices

    def get_action(self, obs):
        action = np.zeros(self.shape)
        action[:, 0] = np.random.choice(
            self.num_groups, self.num_actions, replace=False)
        action[:, 1] = np.random.choice(
            self.nodes_array, self.num_actions, replace=False)
        return action
