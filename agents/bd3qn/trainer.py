import numpy as np
from random import sample
from torch.utils import tensorboard

from utils import save_checkpoint, save_best, build_action_table, state_pre_processing
from datetime import datetime
import os
import time as delay

class Trainer:
    def __init__(self, model, bdqn_independent_models, bd3qn_type_models, swarm_method,
                 env,
                 memory,
                 max_steps,
                 max_episodes,
                 epsilon_start,
                 epsilon_final,
                 epsilon_decay,
                 start_learning,
                 batch_size,
                 save_update_freq,
                 exploration_method,
                 output_dir,
                 players,
                 player_num,
                 config_dir,
                 map_file,
                 unit_file,
                 env_output_dir,
                 pnames,
                 debug,
                 renderer):
        self.model = model
        self.bdqn_independent_models = bdqn_independent_models
        self.bd3qn_type_models = bd3qn_type_models
        self.swarm_method = swarm_method
        self.env = env
        self.memory = memory
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.start_learning = start_learning
        self.batch_size = batch_size
        self.save_update_freq = save_update_freq
        self.output_dir = output_dir
        self.action_table = build_action_table(env.num_groups, env.num_nodes)
        self.players = players
        self.player_num = player_num
        self.config_dir = config_dir
        self.map_file = map_file
        self.unit_file = unit_file
        self.env_output_dir = env_output_dir
        self.pnames = pnames
        self.debug = debug
        self.exploration_method = exploration_method
        self.renderer = renderer
        self.nodes_array = []
        for i in range(1, self.env.num_nodes + 1):
            self.nodes_array.append(i)

    def _exploration(self, step):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)

    def loop(self):
        #state = self.env.reset()
        state = self.env.reset(
            players=self.players,
            config_dir=self.config_dir,
            map_file=self.map_file,
            unit_file=self.unit_file,
            output_dir=self.env_output_dir,
            pnames=self.pnames,
            debug=self.debug
        )
        num_of_wins = 0
        episode_winrate = 0
        total_games_played = 0
        all_winrate = []
        highest_winrate = 0
        w = tensorboard.SummaryWriter()
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = './runs/{}/'.format(self.output_dir)
        try:
            os.makedirs(path)
        except:
            pass

        for step in range(self.max_steps):
            self.renderer.render(state)
            epsilon = self._exploration(step)
            # print(epsilon)
            action_idx = []
            action = {}
            for pid in self.players:
                if pid == self.player_num:
                    per_swarm_states = state_pre_processing(state[pid])
                    if self.exploration_method == "Noisy" or np.random.random_sample() > epsilon:
                        acts = []
                        if self.swarm_method == "Single network":
                            for idx, s in enumerate(per_swarm_states):
                                chose_next_location = self.model.get_action(s)
                                acts.append(([idx, chose_next_location[0]], chose_next_location[1]))
                        elif self.swarm_method == "Group by type":
                            for idx, s in enumerate(per_swarm_states):
                                types = per_swarm_states[11:14]
                                t = types.index(1)
                                chose_next_location = self.bd3qn_type_models[t].get_action(s)
                                acts.append(([idx, chose_next_location[0]], chose_next_location[1]))
                        else:
                            for idx, s in enumerate(per_swarm_states):
                                chose_next_location = self.bdqn_independent_models[idx].get_action(s)
                                acts.append(([idx, chose_next_location[0]], chose_next_location[1]))
                        print("acts", acts)
                        acts.sort(key=lambda tup: tup[1], reverse=True)
                        acts = sample(acts, 7)
                        action[pid] = []
                        for i in range(0, 7):
                            action[pid].append(acts[i][0])
                        action[pid] = np.array(action[pid])
                        print("actions", action[pid])
                        
                    else:
                        #print("not here")
                        action_idx = np.random.choice(
                            len(self.action_table), size=7)
                        action[pid] = np.zeros(
                            (self.env.num_actions_per_turn, 2))
                        for n in range(0, len(action_idx)):
                            action[pid][n][0] = self.action_table[action_idx[n]][0]
                            action[pid][n][1] = self.action_table[action_idx[n]][1]
                else:
                    action[pid] = self.players[pid].get_action(state[pid])

            next_state, reward, done, infos = self.env.step(action)

            if done:
                next_state = self.env.reset(
                    players=self.players,
                    config_dir=self.config_dir,
                    map_file=self.map_file,
                    unit_file=self.unit_file,
                    output_dir=self.env_output_dir,
                    pnames=self.pnames,
                    debug=self.debug
                )
                if reward[self.player_num] == 1:
                    num_of_wins += 1
                total_games_played += 1
                print("Result on game {}: {}".format(
                    len(all_winrate), reward))
                episode_winrate = (num_of_wins/total_games_played) * 100
                all_winrate.append(episode_winrate)
                with open(os.path.join(path, "rewards-{}.txt".format(time)), 'a') as fout:
                    fout.write("{}\n".format(episode_winrate))
                print("Current winrate: {}%".format(episode_winrate))
                w.add_scalar("winrate",
                             episode_winrate, global_step=len(all_winrate))
                if episode_winrate > highest_winrate:
                    highest_winrate = episode_winrate
                    save_best(self.model, all_winrate,
                              "Evergaldes", self.output_dir)

            self.memory.push((
                state[self.player_num],
                action_idx,
                reward[self.player_num],
                next_state[self.player_num],
                done)
            )
            
            state = next_state
            

            if step > self.start_learning:
                if self.swarm_method == "Single network":
                    for i in range(0, len(self.bdqn_independent_models)):
                        loss = self.model.update_policy(
                            self.memory.sample(self.batch_size), self.memory, i)
                elif self.swarm_method == "Group by type":
                    for g in self.bd3qn_type_models:
                        loss = g.update_policy(
                            self.memory.sample(self.batch_size), self.memory, idx)
                else:
                    for idx, g in enumerate(self.bdqn_independent_models):
                        loss = g.update_policy(
                            self.memory.sample(self.batch_size), self.memory, idx)

                # with open(os.path.join(path, "loss-{}.txt".format(time)), 'a') as fout:
                #     fout.write("{}\n".format(loss))
                # w.add_scalar("loss/loss", loss, global_step=step)

            if step % self.save_update_freq == 0:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)

            if len(all_winrate) == self.max_episodes:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)
                break

        w.close()
