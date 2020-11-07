import os
import importlib
import gym
import gym_everglades
import pdb
import sys
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.models import Model, load_model
from agents.player import Player
from collections import deque
import numpy as np
import random
from everglades_server import server
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# create model


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu",
              kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # # Hidden layer with 64 nodes
    #X = Dense(128, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions
    X = Dense(action_space, activation="linear",
              kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='Everglades-DQN-model')
    model.compile(loss="mse", optimizer=Adam(
        learning_rate=0.01), metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent(Player):

    # init the dqn agent with hyperparams
    def __init__(self, action_space, player_num, map_name):
        super().__init__(action_space, player_num, map_name)

        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 500
        self.winRate = 1.0

    # additional init variables, mostly env related
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

    # function to store states into the memory and apply epsilon decay (exploration rate decay)
    def remember(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # replay the memory
    def replay(self):
        if len(self.memory) < self.train_start:
            return

        # pull random batch_size (e.g 64) states from the memory
        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))

        # empty objects to store collected different attr of states in the current batch
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # collect and store states in the current batch before Q value processing
        for i in range(self.batch_size):
            states[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_states[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # make a prediction for the states in the current batch
        target = self.model.predict(states)
        # print(target[0])

        # same thing for next state
        target_next = self.model.predict(next_states)

        # update the q values for the best actions of a state in the current batch
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
                # print(target_next[i])
                max_next = (-target_next[i]).argsort()[:self.num_actions]
                for a in range(0, len(action[i][0])):
                    target[i][action[i][0][a]] = reward[i] + \
                        self.gamma * target_next[i][max_next[a]]

        # train the model with the new q values
        self.model.fit(states, target, batch_size=self.batch_size, verbose=2)

    # helper functions for calculate the winrate
    def get_winrate(self, curWin, totalGame):
        return curWin/totalGame

    # the function does 2 things:
    # - explore a new state
    # - or make an action based on the model prediction
    def act(self, obs):
        if np.random.random_sample() <= self.epsilon:
            return Player.get_action(self, obs)
        else:
            return self.get_action(obs)

    # get prediction (q values) of all actions and get the 7 best ones
    def get_action(self, obs):
        action = np.zeros(self.shape)
        pred = self.model.predict(obs)
        maxIndices = (-pred[0]).argsort()[:self.num_actions]
        for i in range(0, self.num_actions):
            action[i] = self.action_choices[maxIndices[i]]
        return (maxIndices, action)

    # save model
    def save(self, name):
        self.model.save(name)

    # load model
    def load(self, name):
        self.model = load_model(name)

    # train the AI
    def run(self):

        # variable for track number of win games by dqn agent
        i = 0
        for e in range(self.EPISODES):
            # reset env for every new game
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

                # actions of both dqn and rand agent since the env need both actions before step() to the next state
                actions = {}
                for pid in self.players:
                    # dqn agent makes a move
                    if pid == self.player_num:
                        state[pid] = np.reshape(
                            state[pid], [1, self.state_size])
                        agent_move = self.act(state[pid])
                        actions[pid] = agent_move[1]
                    # the other agent makes a move
                    else:
                        actions[pid] = self.players[pid].get_action(state[pid])[
                            1]

                # get the next state
                next_state, reward, done, _ = self.env.step(actions)

                # reshapte before feed to the model
                next_state[self.player_num] = np.reshape(
                    next_state[self.player_num], [1, self.state_size])
                # print(agent_move)

                # store the current state into the memory
                self.remember(state[self.player_num], agent_move,
                              reward[self.player_num], next_state[self.player_num], done)

                # move to the next state
                state = next_state

                # post game processing
                if done:
                    print("reward", reward)

                    # if dqn agent win, increment i
                    if reward[self.player_num] == 1:
                        i += 1

                    # get winrate of dqn agent
                    next_winrate = self.get_winrate(i, e+1)
                    self.winRate = next_winrate
                    print("Current winrate: {:2}".format(self.winRate))

                    # save the model every 100 games
                    if e % 100 == 0:
                        print(
                            "Saving trained model as everglades-dqn-{}-{:2}.h5 with win rate at: {:2}".format(e+1, next_winrate, next_winrate))
                        self.save(
                            "everglades-dqn-{}-{:2}.h5".format(e+1, next_winrate))

                    print("episode: {}/{}, win: {}, e: {:.2}".format(e+1,
                                                                     self.EPISODES, i, self.epsilon))

                    # save and quit after win 500 games
                    if i == 500:
                        print("Saving trained model as everglades-dqn.h5")
                        self.save("everglades-dqn.h5")
                        return

                # replay buffer
                self.replay()


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
