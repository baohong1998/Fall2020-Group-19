import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from noisy_net import NoisyLinear, NoisyFactorizedLinear
from OneHotEncode import OneHotEncode

class BranchingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture="DQN"):
        super().__init__()
        self.exploration_method = exploration_method
        self.architecture = architecture
        if self.exploration_method == "Noisy":
            self.model = nn.ModuleList([nn.Sequential(
                NoisyLinear(62, hidden_dim*4),
                nn.ReLU(),
                NoisyLinear(hidden_dim*4, hidden_dim*2),
                nn.ReLU(),
                NoisyLinear(hidden_dim*2, hidden_dim),
                nn.ReLU()
            ) for i in range(12)])
            if self.architecture == "Dueling":
                self.value_head = nn.ModuleList([NoisyLinear(hidden_dim, 1) for i in range(12)])
                self.adv_heads = nn.ModuleList([NoisyLinear(hidden_dim, 11) for i in range(12)])
            else:
                self.out = nn.ModuleList([NoisyLinear(hidden_dim, 11) for i in range(12)])
        else:
            self.model = nn.ModuleList([nn.Sequential(
                nn.Linear(62, hidden_dim*4),
                nn.ReLU(),
                nn.Linear(hidden_dim*4, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU()
            ) for i in range(12)])
            if self.architecture == "Dueling":
                self.value_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for i in range(12)])
                self.adv_heads = nn.ModuleList([nn.Linear(hidden_dim, 11) for i in range(12)])
            else:
                self.out = nn.ModuleList([nn.Linear(hidden_dim, 11) for i in range(12)])

    def forward(self, x):
        if(len(x.shape) > 1):
            #print(x.shape)
            processed_x = self.batch_state_processing(x)
            processed_x = processed_x.transpose(0,1)
            layer1 = torch.stack([self.model[i](processed_x[i]) for i, _ in enumerate(processed_x)])
            if self.architecture == "Dueling":
                value = torch.stack([self.value_head[i](layer1[i]) for i, _ in enumerate(layer1)])
                advs = torch.stack([self.adv_heads[i](layer1[i]) for i, _ in enumerate(layer1)])
                mean = advs.mean(2, keepdim=True)
                q_val = value + advs - mean
                # print(q_val.device)
            else:
                q_val = torch.stack([self.out[i](layer1[i]) for i, _ in enumerate(layer1)])

        else:
            processed_x = self.state_processing(x)
            layer1 = torch.stack([self.model[i](processed_x[i]) for i, _ in enumerate(processed_x)])
            if self.architecture == "Dueling":
                value = torch.stack([self.value_head[i](layer1[i]) for i, _ in enumerate(layer1)])
                advs = torch.stack([self.adv_heads[i](layer1[i]) for i, _ in enumerate(layer1)])
                q_val = value + advs - advs.mean()
            else:
                q_val = torch.stack([self.out[i](layer1[i]) for i, _ in enumerate(layer1)])
            
        return q_val

    def sample_noise(self):
        for m in self.model:
            m[0].sample_noise()
            m[2].sample_noise()
            m[4].sample_noise()
        for v in self.value_head:
            v.sample_noise()
        for l in self.adv_heads:
            l.sample_noise()
    
    def state_processing(self, obs):
        node_info = obs[:45]
        groups_info = obs[45:]
        partitions = [17 for i in range(12)]
        groups = torch.split(groups_info, partitions)
        groups_final = torch.stack([torch.cat((node_info, groups[i])) for i in range(len(groups))])
        return groups_final
    
    def batch_state_processing(self, batch_obs):
        groups_final = []
        for i in range(batch_obs.shape[0]):
            node_info = batch_obs[i][:45]
            groups_info = batch_obs[i][45:]
            partitions = [17 for i in range(12)]
            groups = torch.split(groups_info, partitions)
            groups_final.append(torch.stack([torch.cat((node_info, groups[i])) for i in range(len(groups))]))
        groups_final = torch.stack(groups_final)
        return groups_final

class BranchingDQN(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, target_update_freq, learning_rate, gamma, hidden_dim, td_target, device, exploration_method):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.gamma = gamma
        self.exploration_method = exploration_method
        self.policy_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method)
        self.target_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optim = optim.Adam(
            self.policy_network.parameters(), lr=learning_rate)

        self.policy_network.to(device)
        self.target_network.to(device)
        self.device = device

        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.td_target = td_target

    def get_action(self, x):
        turn = x[0]
        new_obs = OneHotEncode(x)
        new_obs = torch.tensor(new_obs).float()
        new_obs = new_obs.to(self.device)
        with torch.no_grad():
            if self.exploration_method == "Noisy":
                self.policy_network.sample_noise()
            out = self.policy_network(new_obs)
            out_max = out.max(1)
            out_max_sorted = out_max.values.sort(descending=True)
            chosen_group = out_max_sorted.indices[:7]
            chosen_location = torch.stack([out_max.indices[i] for i in chosen_group]) + 1
            action = torch.stack([chosen_group, chosen_location], dim = 1)
            
        #print("Turn {}:".format(turn), out.max(1, keepdim = True))
        return action.detach().cpu().numpy()  # action.numpy()

    def update_policy(self, batch, memory):
        sample, batch_indxs, batch_weights = batch
        
        batch_states = sample[0]
        batch_action = sample[1]
        batch_rewards = sample[2]
        batch_next_states = sample[3]
        batch_done = sample[4]

        states = torch.tensor(batch_states).float().to(self.device)
        actions = torch.tensor(batch_action).long().to(self.device)
        rewards = torch.tensor(batch_rewards).float().to(self.device)
        next_states = torch.tensor(batch_next_states).float().to(self.device)
        done = torch.tensor(batch_done).float().to(self.device)

        
        next_Q = self.target_network(next_states)
    
        actions = actions.transpose(1,2)
        actions[:,1] = actions[:,1] - 1
        actions = actions.transpose(1,2)

        current_Q = self.policy_network(states)
        current_Q = current_Q.transpose(0,1)
       
        nCurQt = []
        for idx, group in enumerate(current_Q):
            new_curQT = []
            for sub_act in actions[idx]:
                new_curQT.append(group[sub_act[0]][sub_act[1]])
            new_curQT = torch.stack(new_curQT)
            nCurQt.append(new_curQT)
        nCurQt = torch.stack(nCurQt)
        #print(nCurQt[0][0])
        with torch.no_grad():
            next_Q = self.target_network(next_states)
            next_Q = next_Q.transpose(0,1)
            nNextQt = []
            for i, x in enumerate(next_Q):
                nNextQt.append(x.max(1).values.sort(descending=True).values[:7])
            nNextQt = torch.stack(nNextQt)
        # print(actions)
        # print("pre Current Q", current_Q[0][0][0])
        # print("pre next Q", next_Q.shape)

        # current_Q = current_Q.max(1, keepdim=True).values
        # next_Q = next_Q.max(1, keepdim=True).values
        # if self.td_target == "mean":
        #     current_Q = current_Q.mean()
        # elif self.td_target == "max":
        #     current_Q, _ = current_Q.max(1, keepdim=True)
        # with torch.no_grad():
        #     max_next_Q = self.target_network(next_states).gather(
        #         2, argmax.unsqueeze(2)).squeeze(-1)
        #     if self.td_target == "mean":
        #         max_next_Q = max_next_Q.mean(1, keepdim=True)
        #     elif self.td_target == "max":
        #         max_next_Q, _ = max_next_Q.max(1, keepdim=True)

        # print("Current Q", current_Q.shape)
        #print("reward", rewards.shape, rewards)
        #print("next Q", nNextQt.shape, nNextQt[0][0])
        #print("done", done.shape, done)
        # rewards = rewards.reshape(states.shape[0],1,1)
        # done = done.reshape(states.shape[0],1,1)
        expected_Q = rewards.unsqueeze(1) +  nNextQt*self.gamma 
        # errors = torch.abs(expected_Q - current_Q).cpu().data.numpy()
        #print("Expect:", expected_Q, "Current:", current_Q, "Error:", errors)
        # print("Expect Q", expected_Q.shape, expected_Q[0][0])
        batch_weights = torch.from_numpy(batch_weights).float()
        batch_weights = batch_weights.to(self.device)
        loss = batch_weights * F.mse_loss(nCurQt, expected_Q)
        #print("curQ",nCurQt[0][0])
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
