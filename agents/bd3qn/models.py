import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import state_pre_processing
from noisy_net import NoisyLinear, NoisyFactorizedLinear


class BranchingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture):
        super().__init__()
        self.exploration_method = exploration_method
        self.architecture = architecture
        if self.exploration_method == "Noisy":
            self.model = nn.Sequential(
                NoisyLinear(observation_space, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            if self.architecture == "Dueling":
                self.value_head =  nn.Sequential(
                    NoisyLinear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    NoisyLinear(hidden_dim, 1)
                )
                self.adv_heads = nn.Sequential(
                    NoisyLinear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    NoisyLinear(hidden_dim, 11)
                )
            else:
                self.out = NoisyLinear(hidden_dim, 11)

        else:
            self.model = nn.Sequential(
                nn.Linear(observation_space, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            if self.architecture == "Dueling":
                self.value_head =  nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                self.adv_heads = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 11)
                )
            else:
                self.out = nn.Linear(hidden_dim, 11)

    def forward(self, x):
        out = self.model(x)
        if self.architecture == "Dueling":
            value = self.value_head(out)
            advs = self.adv_heads(out)
            q_val = value + (advs - advs.mean())
        else:
            q_val = self.out(out)
        return q_val

    def sample_noise(self):
        self.model[0].sample_noise()
        self.model[2].sample_noise()
        if self.architecture == "Dueling":
            self.value_head.sample_noise()
            self.sample_noise()
        else:
            self.out.sample_noise()
        


class BranchingDQN(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, target_update_freq, learning_rate, gamma, hidden_dim, td_target, device, exploration_method, architecture, multi_steps=3):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.gamma = gamma
        self.exploration_method = exploration_method
        self.architecture = architecture

        self.policy_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture)
        self.target_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture)
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
        x = torch.from_numpy(x).float()
        x = x.to(self.device)
        with torch.no_grad():
            if self.exploration_method == "Noisy":
                self.policy_network.sample_noise()
            out = self.policy_network(x)
            action = torch.argmax(out)

        return (action.detach().cpu().numpy().item(), out.max().detach().cpu().numpy().item())  # action.numpy()

    def update_policy(self, batch, memory, swarm_id):
        # print(batch)
        batch_data, batch_indxs, batch_weights = batch
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_done = []
        for b in batch_data:
            batch_states.append(b[0])
            batch_actions.append(b[1])
            batch_rewards.append(b[2])
            batch_next_states.append(b[3])
            batch_done.append(b[4])
        # print("state", batch_states)
        # print("act", batch_actions)
        # print("reward", batch_rewards)
        # print("next_state", batch_next_states)
        # print("done", batch_done)
        states = []
        # actions = torch.tensor(batch_actions).long().reshape(
        #     states.shape[0], -1, 1).to(self.device)
        rewards = torch.tensor(batch_rewards).float(
        ).reshape(-1, 1).to(self.device)
        next_states = []

        if self.exploration_method == "Noisy":
            self.policy_network.sample_noise()
        
        for idx, s in enumerate(batch_states):
            states.append(state_pre_processing(s)[swarm_id])

        states = torch.tensor(states).float().to(self.device)
       
        current_Q = self.policy_network(states)
        if self.td_target == "mean":
            current_Q = current_Q.mean(1, keepdim=True)
        elif self.td_target == "max":
            current_Q, _ = current_Q.max(1, keepdim=True)
        with torch.no_grad():
            if self.exploration_method == "Noisy":
                self.target_network.sample_noise()

            for idx, s in enumerate(batch_next_states):
                next_states.append(state_pre_processing(s)[swarm_id])

            next_states = torch.tensor(next_states).float().to(self.device)
            
            max_next_Q = self.target_network(next_states)
            if self.td_target == "mean":
                max_next_Q = max_next_Q.mean(1, keepdim=True)
                print("mean", max_next_Q.shape)
            elif self.td_target == "max":
                max_next_Q, _ = max_next_Q.max(1, keepdim=True)
                

        #print("Current Q", current_Q)
        expected_Q = rewards + max_next_Q * (self.gamma)
        errors = torch.abs(expected_Q - current_Q).cpu().data.numpy()

        #print("Expect Q", expected_Q)
        # batch_weights = torch.from_numpy(batch_weights).float()
        # batch_weights = batch_weights.to(self.device)
        loss = (batch_weights *
                F.mse_loss(current_Q, expected_Q)).mean()
        # print(batch_weights)
        self.optim.zero_grad()
        loss.backward()
        # print(self.policy_network.parameters())
        for p in self.policy_network.parameters():
            p.grad.data.clamp_(-1., 1.)
        self.optim.step()
        memory.update_priorities(batch_indxs, errors)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            #print("Update target net")
            self.update_counter = 0
            self.target_network.load_state_dict(
                self.policy_network.state_dict())

        return loss.detach().cpu()
