import numpy as np

from torch.utils import tensorboard

from utils import save_checkpoint, save_best, build_action_table


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
                 output_dir,
                 players,
                 player_num,
                 config_dir,
                 map_file,
                 unit_file,
                 env_output_dir,
                 pnames,
                 debug):
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

        for step in range(self.max_steps):
            epsilon = self._exploration(step)
            # print(epsilon)
            action_idx = []
            action = {}
            for pid in self.players:
                if pid == self.player_num:
                    if np.random.random_sample() > epsilon:
                        action_idx = self.model.get_action(state[pid])
                        action[pid] = np.zeros(
                            (self.env.num_actions_per_turn, 2))
                        for n in range(0, len(action_idx)):
                            action[pid][n][0] = self.action_table[action_idx[n]][0]
                            action[pid][n][1] = self.action_table[action_idx[n]][1]

                    else:
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
                print("Current winrate: {}%".format(episode_winrate))
                w.add_scalar("winrate",
                             episode_winrate, global_step=len(all_winrate))
                if episode_winrate > highest_winrate:
                    highest_winrate = episode_winrate
                    save_best(self.model, all_winrate,
                              "Evergaldes", self.output_dir)

            self.memory.store(
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
                w.add_scalar("loss/loss", loss, global_step=step)

            if step % self.save_update_freq == 0:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)

            if len(all_winrate) == self.max_episodes:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)
                break

        w.close()
