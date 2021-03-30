import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from noisy_net import NoisyLinear, NoisyFactorizedLinear
from player import PlayerHelper
from utils import build_action_table
class BranchingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture = "DQN"):
        super().__init__()
        self.exploration_method = exploration_method
        # if self.exploration_method == "Noisy":
        #     self.model = nn.Sequential(
        #         NoisyLinear(observation_space, hidden_dim*4),
        #         nn.ReLU(),
        #         NoisyLinear(hidden_dim*4, hidden_dim*2),
        #         nn.ReLU(),
        #         NoisyLinear(hidden_dim*2, hidden_dim),
        #         nn.ReLU()
        #     )
        #     self.value_head = NoisyLinear(hidden_dim, 1)
        #     self.adv_heads = nn.ModuleList(
        #         [NoisyLinear(hidden_dim, action_bins) for i in range(action_space)])
        # else:
        self.architecture = architecture
        self.model = nn.Sequential(
            nn.Linear(observation_space, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        if architecture == "Dueling":
            self.value_head = nn.Linear(hidden_dim, 1)
            self.adv_heads = nn.Linear(hidden_dim, action_bins)
        else:
            self.out = nn.Linear(hidden_dim, action_bins)

    def forward(self, x):
        first_layer = self.model(x)
        out = self.out(first_layer)
        q_val = out
        if self.architecture == "Dueling":
            value = self.value_head(out)
            advs = self.advs_heads(out)
            q_val = value + advs - advs.mean()
        # if value.shape[0] == 1:
        #     advs = torch.stack([l(out) for l in self.adv_heads], dim=0)
        #     q_val = value + advs - advs.mean(1, keepdim=True)
        # else:
        #     advs = torch.stack([l(out) for l in self.adv_heads], dim=1)
        #     q_val = value.unsqueeze(1) + advs - advs.mean(2, keepdim=True)
        return q_val

    def sample_noise(self):
        self.model[0].sample_noise()
        self.model[2].sample_noise()
        self.model[4].sample_noise()
        self.value_head.sample_noise()
        for l in self.adv_heads:
            l.sample_noise()


class BranchingDQN(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, target_update_freq, learning_rate, gamma, hidden_dim, td_target, device, exploration_method):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.gamma = gamma
        self.exploration_method = exploration_method

        self.player_helper = PlayerHelper(7,1, "../config/DemoMap.json")
        self.policy_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method)
        self.target_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optim = optim.Adam(
            self.policy_network.parameters(), lr=learning_rate)

        self.action_choices = build_action_table()

        self.policy_network.to(device)
        self.target_network.to(device)
        self.device = device

        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.td_target = td_target

    def get_action(self, x):
        #turn = x[0]
        legal_obs = self.player_helper.legal_moves(x)
        legal_idx = [i for i, x in enumerate(legal_obs) if x]
        legal_idx = torch.LongTensor(legal_idx).to(self.device)
        #print("legal idx" , legal_idx)
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            if self.exploration_method == "Noisy":
                self.policy_network.sample_noise()
            preds = self.policy_network(x)
            #print("pred", preds)
            legal_preds = preds.gather(0, legal_idx)
            #print(legal_preds)
            legal_preds = legal_preds.sort(descending=True)
            legal_preds_idx = legal_preds.indices[:7]
            action_idx = legal_idx.gather(0, legal_preds_idx)
            action_idx = action_idx.detach().cpu().numpy()
            actions = np.take(self.action_choices, action_idx, 0)
        #print("action", actions)
        #print("Turn {}:".format(turn), out.max(1, keepdim = True))
        return actions # action.numpy()

    def update_policy(self, batch, memory):
        sample, batch_indxs, batch_weights = batch
        
        batch_states = sample[0]
        batch_actions = sample[1]
        batch_rewards = sample[2]
        batch_next_states = sample[3]
        batch_done = sample[4]

        states = torch.tensor(batch_states).float().to(self.device)
        actions = []
        temp = self.action_choices.tolist()
        #print("temp", temp)
        for batch in batch_actions:
            act = []
            for a in batch:
                a = a.tolist()
                act.append(temp.index(a))
            if len(act) < 7:
                remain = 7 - len(act)
                for i in range(remain):
                    act.append(-1)
            actions.append(act)
        #print("actions", actions)
        #actions = torch.stack(actions)
        actions = torch.tensor(actions).long().to(self.device)
        #print("actions", actions)
        rewards = torch.tensor(batch_rewards).float().to(self.device).unsqueeze(1)
        batch_action_choices = []
        for nState in batch_next_states:
            legal_nState = self.player_helper.legal_moves(nState)
            legal_idx = [i for i, x in enumerate(legal_nState) if x]
            batch_action_choices.append(torch.LongTensor(legal_idx).to(self.device))
        #print("batch action choices", batch_action_choices)
        next_states = torch.tensor(batch_next_states).float().to(self.device)

        current_Q = self.policy_network(states)

        new_current_Q = []
        for idx, index_group in enumerate(actions):
            x = []
            for i in index_group:
                if i > -1:
                    x.append(current_Q[idx][i])
                else:
                    x.append(torch.tensor(0).to(self.device))
            new_current_Q.append(torch.stack(x))
        new_current_Q = torch.stack(new_current_Q)

        #print("Current Q", current_Q)
        with torch.no_grad():
            next_Q = self.target_network(next_states)

            next_Q_final = []
            for i, n in enumerate(next_Q):
                legal_next = n.gather(0, batch_action_choices[idx])
                legal_next = legal_next.sort(descending=True)
                legal_next = legal_next.values[:7]
                if len(legal_next) < 7:
                    remain = 7 - len(legal_next)
                    for k in range(remain):
                        legal_next = torch.cat((legal_next, torch.tensor([0]).to(self.device)))
                    #print("legal next", legal_next)
                next_Q_final.append(legal_next)
            next_Q_final = torch.stack(next_Q_final)
            #print("next q", next_Q_final)
            
        # next_Q_final = []
        # for q in next_Q.values:
        #     next_Q_final.append(q[:7])
        # next_Q_final = torch.stack(next_Q_final)
            #print("next q", next_Q_final)
        # states = torch.tensor(batch_states).float().to(self.device)
        # actions = torch.tensor(batch_actions).long().reshape(
        #     states.shape[0], -1, 1).to(self.device)
        # rewards = torch.tensor(batch_rewards).float(
        # ).reshape(-1, 1).to(self.device)
        # next_states = torch.tensor(batch_next_states).float().to(self.device)
        # if self.exploration_method == "Noisy":
        #     self.policy_network.sample_noise()
        # current_Q = self.policy_network(states).gather(2, actions).squeeze(-1)
        # if self.td_target == "mean":
        #     current_Q = current_Q.mean(1, keepdim=True)
        # elif self.td_target == "max":
        #     current_Q, _ = current_Q.max(1, keepdim=True)
        # with torch.no_grad():
        #     argmax = torch.argmax(self.policy_network(next_states), dim=2)
        #     if self.exploration_method == "Noisy":
        #         self.target_network.sample_noise()
        #     max_next_Q = self.target_network(next_states).gather(
        #         2, argmax.unsqueeze(2)).squeeze(-1)
        #     if self.td_target == "mean":
        #         max_next_Q = max_next_Q.mean(1, keepdim=True)
        #     elif self.td_target == "max":
        #         max_next_Q, _ = max_next_Q.max(1, keepdim=True)

        #print("Current Q", current_Q)
        expected_Q = rewards + next_Q_final * self.gamma
        #print("Expect:", expected_Q, "Current:", current_Q, "Error:", errors)
        #print("Expect Q", expected_Q)
        #print("weights", batch_weights)
        batch_weights = torch.tensor(batch_weights).float().to(self.device)
        loss = batch_weights * F.mse_loss(new_current_Q, expected_Q)
        prios = loss + 1e-5
        loss = loss.mean()
        # print(self.policy_network)
        self.optim.zero_grad()
        loss.backward()
        # print(self.policy_network.parameters())
        # for p in self.policy_network.parameters():
        #     p.grad.data.clamp_(-1., 1.)
        self.optim.step()

        
        memory.update_priorities(batch_indxs, prios.data.cpu().numpy())

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            #print("Update target net")
            self.update_counter = 0
            self.target_network.load_state_dict(
                self.policy_network.state_dict())

        return loss.detach().cpu()
