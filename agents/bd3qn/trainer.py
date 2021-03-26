import numpy as np

from torch.utils import tensorboard
import random
from utils import save_checkpoint, save_best, build_action_table
from datetime import datetime
import os
import importlib

class Trainer:
    def __init__(self, model,
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
        self.nodes_array = []
        self.renderer = renderer
        for i in range(1, self.env.num_nodes + 1):
            self.nodes_array.append(i)

    def _exploration(self, step):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)

    def _changePlayer(self, player, pid):
        rand_agent_file = "./{}".format(player)
        rand_agent_name, rand_agent_extension = os.path.splitext(rand_agent_file)
        rand_agent_mod = importlib.import_module(
            rand_agent_name.replace('./', 'agents.'))
        rand_agent_class = getattr(
            rand_agent_mod, os.path.basename(rand_agent_name))
        rand_player = rand_agent_class(self.env.num_actions_per_turn, 0)
        self.players[pid] = rand_player
        self.pnames[pid] = rand_player.__class__.__name__

    def loop(self):
        #state = self.env.reset()
        player_list = {
            'random_actions': 15, 
            'base_rushV1': 1, 
            'Cycle_BRush_Turn25': 1, 
            'Cycle_BRush_Turn50': 1,
            'Cycle_Target_Node': 1,
            'cycle_targetedNode1': 1,
            'cycle_targetedNode11': 1,
            'cycle_targetedNode11P2': 1,
            'same_commands': 1,
            'SwarmAgent': 1
            }
        plist = []

        for p in list(player_list.keys()):
            for i in range(0, player_list[p]):
                plist.append(p)
        
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
        
        total_turn_played = 0
        turn_played_by_network = 0

        for step in range(self.max_steps):
            epsilon = self._exploration(step)
            self.renderer.render(state)
            # print(epsilon)
            
            action_idx = []
            action = {}
            total_turn_played += 1
            for pid in self.players:
                if pid == self.player_num:
                    # print(self.exploration_method)
                    if self.exploration_method == "Noisy" or np.random.random_sample() > epsilon:
                        action_idx = self.model.get_action(state[pid])
                        action[pid] = np.zeros(
                            (self.env.num_actions_per_turn, 2))
                        for n in range(0, len(action_idx)):
                            action[pid][n][0] = self.action_table[action_idx[n]][0]
                            action[pid][n][1] = self.action_table[action_idx[n]][1]
                        # print(action[pid])
                        turn_played_by_network += 1
                    else:
                        #print("not here")
                        # action_idx = np.random.choice(
                        #     len(self.action_table), size=7)
                        # action[pid] = np.zeros(
                        #     (self.env.num_actions_per_turn, 2))
                        # for n in range(0, len(action_idx)):
                        #     action[pid][n][0] = self.action_table[action_idx[n]][0]
                        #     action[pid][n][1] = self.action_table[action_idx[n]][1]
                        agent_id = 0
                        if pid == 0:
                            agent_id = 1
                        
                        action[pid] = self.players[agent_id].get_action(state[pid])
                else:
                    action[pid] = self.players[pid].get_action(state[pid])

            next_state, reward, done, infos = self.env.step(action)

            if done:
                for pid in self.players:
                    if pid != self.player_num:
                        #print(plist)
                        self.player_name = random.choice(plist)
                        counter = player_list[self.player_name]
                        self._changePlayer(self.player_name, pid)
                        print("Training with {}".format(self.player_name))

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
                print("Result on game {}: {}. Number of moves made by the network: {}/{}. Agents: {}".format(
                    len(all_winrate), reward, turn_played_by_network, total_turn_played, self.player_name))
                episode_winrate = (num_of_wins/total_games_played) * 100
                all_winrate.append(episode_winrate)
                with open(os.path.join(path, "rewards-{}.txt".format(time)), 'a') as fout:
                    fout.write("Winrate: {}. Number of moves made by the network: {}/{}. Agents: {}\n".format(episode_winrate, turn_played_by_network, total_turn_played, self.player_name))
                print("Current winrate: {}%".format(episode_winrate))
                w.add_scalar("winrate",
                             episode_winrate, global_step=len(all_winrate))
                turn_played_by_network = 0
                total_turn_played = 0
                if episode_winrate > highest_winrate:
                    highest_winrate = episode_winrate
                    save_best(self.model, all_winrate,
                              "Evergaldes", self.output_dir)

            self.memory.add(
                state[self.player_num],
                action_idx,
                reward[self.player_num],
                next_state[self.player_num],
                done
            )
            state = next_state

            if step > self.start_learning:
                loss = self.model.update_policy(
                    self.memory.miniBatch(self.batch_size), self.memory)
                with open(os.path.join(path, "loss-{}.txt".format(time)), 'a') as fout:
                    fout.write("{}\n".format(loss))
                w.add_scalar("loss/loss", loss, global_step=step)

            if step % self.save_update_freq == 0:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)

            if len(all_winrate) == self.max_episodes:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)
                break

        w.close()
