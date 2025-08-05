import torch
import torch.nn.functional as F
import numpy as np

from learning.strategy.sac.replay_buffer import ReplayBuffer
from learning.strategy.sac.networks import ActorNetwork, CriticNetwork

class SACAgent:
    def __init__(self, state_dim, action_dim, device, 
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, buffer_size=100_000, batch_size=64):

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_2 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic_2 = CriticNetwork(state_dim, action_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self._update_target_networks(tau=1.0)

    def _update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(q1, target_value)
        critic_2_loss = F.mse_loss(q2, target_value)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._update_target_networks()


