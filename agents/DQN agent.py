import os
import importlib
import gym
import gym_everglades
import pdb
import sys
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.models import Model, load_model
from player import Player
from collections import deque
import numpy as np
import random
from everglades_server import server

# create model


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu",
              kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions
    X = Dense(action_space, activation="softmax",
              kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='Everglades-DQN-model')
    model.compile(loss="mse", optimizer=Adam(
        learning_rate=0.01), metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent(Player):
    def __init__(self, action_space, player_num, map_name):
        super().__init__(action_space, player_num, map_name)

        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

    def set_init_state(self, env, players,
                       config_dir, map_file, unit_file, output_dir, names, debug):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.action_choices.shape[0]
        self.config_dir = config_dir
        self.map_file = map_file
        self.unit_file = unit_file
        self.output_dir = output_dir
        self.names = names
        self.debug = debug
        self.players = players
        # create main model
        self.model = OurModel(input_shape=(self.state_size,),
                              action_space=self.action_size)

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            states[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_states[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(states)
        target_next = self.model.predict(next_states)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                for a in action[i][0]:
                    target[i][a] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                for a in action[i][0]:
                    target[i][a] = reward[i] + \
                        self.gamma * (np.amax(target_next[i]))

        self.model.fit(states, target, batch_size=self.batch_size, verbose=2)

    def run(self):
        i = 0
        for e in range(self.EPISODES):
            state = env.reset(
                players=self.players,
                config_dir=self.config_dir,
                map_file=self.map_file,
                unit_file=self.unit_file,
                output_dir=self.output_dir,
                pnames=self.names,
                debug=self.debug
            )

            done = False
            while not done:
                agent_move = None
                actions = {}
                for pid in self.players:
                    if pid == self.player_num:
                        state[pid] = np.reshape(
                            state[pid], [1, self.state_size])
                        agent_move = self.act(state[pid])
                        actions[pid] = agent_move[1]
                    else:
                        actions[pid] = self.players[pid].get_action(state[pid])

                next_state, reward, done, _ = self.env.step(actions)
                next_state[self.player_num] = np.reshape(
                    next_state[self.player_num], [1, self.state_size])
                self.remember(state[self.player_num], agent_move,
                              reward[self.player_num], next_state[self.player_num], done)
                state = next_state

                if done:
                    print("reward", reward)
                    if reward[self.player_num] == 1:
                        i += 1
                    print("episode: {}/{}, win: {}, e: {:.2}".format(e+1,
                                                                     self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as everglades-dqn.h5")
                        self.save("everglades-dqn.h5")
                        return
                self.replay()

    def act(self, obs):
        if np.random.random() <= self.epsilon:
            return ([], Player.get_action(self, obs))
        else:
            return self.get_action(obs)

    def get_action(self, obs):
        action = np.zeros(self.shape)
        pred = self.model.predict(obs)
        maxIndices = (-pred[0]).argsort()[:self.num_actions]
        for i in range(0, self.num_actions):
            action[i] = self.action_choices[maxIndices[i]]
        return (maxIndices, action)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    map_name = "DemoMap.json"

    config_dir = './config/'
    map_file = config_dir + map_name
    setup_file = config_dir + 'GameSetup.json'
    unit_file = config_dir + 'UnitDefinitions.json'
    output_dir = './game_telemetry/'

    debug = 1

    env = gym.make('everglades-v0')
    players = {}
    names = {}

    rand_player = Player(env.num_actions_per_turn, 0, map_name)
    dqn_player = DQNAgent(env.num_actions_per_turn, 1, map_name)
    players[0] = rand_player
    names[0] = rand_player.__class__.__name__
    players[1] = dqn_player
    names[1] = dqn_player.__class__.__name__

    dqn_player.set_init_state(env, players,
                              config_dir, map_file, unit_file, output_dir, names, debug)
    dqn_player.run()
